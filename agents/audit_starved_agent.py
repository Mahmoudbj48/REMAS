# Starvation audit (read-only) → CSVs with ready-to-send suggestions
import os
import csv
from tqdm import tqdm
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional , Tuple
from qdrant_client.models import Record
import pandas as pd


from utils.qdrant_connection import (
    client,
    OWNER_COLLECTION,
    USER_COLLECTION,
    OWNER_PROFILES_COLLECTION,
    USER_PROFILES_COLLECTION
)
# ----------------- helpers -----------------
def _parse_date(s: Optional[str]) -> Optional[datetime]:
    if not s or not isinstance(s, str):
        return None
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except Exception:
        return None

def _days_since(d: Optional[datetime]) -> Optional[int]:
    if not d:
        return None
    return (datetime.now(timezone.utc) - d).days

def _as_list(x):
    if x is None:
        return []
    return x if isinstance(x, list) else [x]

def _hard_match_user_to_owner(user_p: dict, owner_p: dict) -> bool:
    """
    User seeks (budget >= price, min bedrooms <= offered, state overlap).
    Availability omitted here (can add later).
    """
    u_states = set(map(lambda s: str(s).strip().lower(), _as_list(user_p.get("state"))))
    o_states = set(map(lambda s: str(s).strip().lower(), _as_list(owner_p.get("state"))))
    state_ok = (not u_states or not o_states) or bool(u_states & o_states)

    try:
        price_ok = (owner_p.get("price") is None) or (user_p.get("price") is None) \
                   or (float(owner_p["price"]) <= float(user_p["price"]))
    except Exception:
        price_ok = True

    try:
        beds_ok = (owner_p.get("bedrooms") is None) or (user_p.get("bedrooms") is None) \
                  or (int(owner_p["bedrooms"]) >= int(user_p["bedrooms"]))
    except Exception:
        beds_ok = True

    return bool(state_ok and price_ok and beds_ok)

def _get_point(collection: str, pid: str, with_payload=True, with_vectors=True) -> Optional[Record]:
    try:
        recs: List[Record] = client.retrieve(
            collection_name=collection,
            ids=[str(pid)],
            with_payload=with_payload,
            with_vectors=with_vectors,
        )
        return recs[0] if recs else None
    except Exception:
        return None

def _search_top(collection: str, vector, topn: int = 300) -> List[Record]:
    if vector is None:
        return []
    try:
        return client.search(
            collection_name=collection,
            query_vector=vector,
            limit=topn,
        )
    except Exception:
        return []

def _count_matches_user(user_payload: dict, owners: List[Record]) -> int:
    cnt = 0
    for r in owners:
        p = r.payload or {}
        if _hard_match_user_to_owner(user_payload, p):
            cnt += 1
    return cnt

def _count_matches_owner(owner_payload: dict, users: List[Record]) -> int:
    cnt = 0
    for r in users:
        p = r.payload or {}
        if _hard_match_user_to_owner(p, owner_payload):  # inverse rule
            cnt += 1
    return cnt

def _scroll_profiles(collection: str, page: int = 1000):
    next_page = None
    while True:
        recs, next_page = client.scroll(
            collection_name=collection,
            with_payload=True,
            with_vectors=False,
            limit=page,
            offset=next_page
        )
        for r in recs:
            yield r
        if next_page is None:
            break

def _ensure_logs_dir():
    os.makedirs("logs", exist_ok=True)

# ----------------- USERS: audit & CSV -----------------
def audit_starved_users_to_csv(
    *,
    price_delta: float = 200.0,
    room_delta: int = 1,
    user_topn: int = 300,
    days_threshold: int = 30,
    shows_threshold: int = 1,
    limit_users: Optional[int] = None,
) -> str:
    """
    Find starved users and simulate:
      - bedrooms need - room_delta
      - budget + price_delta
    Writes a CSV with ready-to-send suggestion messages.
    Returns CSV path.
    """
    _ensure_logs_dir()
    path = f"logs/starved_users_suggestions_{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv"

    fieldnames = [
        "user_id",
        "days_since_application",
        "number_of_shows",
        "baseline_matches",
        "relax_bedrooms_minus_matches",
        "delta_bedrooms",
        "suggested_bedrooms",
        "bedrooms_message",
        "relax_budget_plus_matches",
        "delta_budget",
        "suggested_budget",
        "budget_message",
    ]

    written = 0
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        processed = 0
        profiles = list(_scroll_profiles(USER_PROFILES_COLLECTION))
        print(f"🔎 Auditing {len(profiles)} user profiles...")

        for prof in tqdm(profiles, desc="Users audited", unit="user"):
            p = prof.payload or {}
            days = _days_since(_parse_date(p.get("application_date")))
            try:
                shows = int(p.get("number_of_shows", 0))
            except Exception:
                shows = 0

            # starved?
            if days is None or days < days_threshold or shows > shows_threshold:
                continue

            user_id = str(prof.id)
            urec = _get_point(USER_COLLECTION, user_id, with_payload=True, with_vectors=True)
            if not urec:
                continue

            up = urec.payload or {}
            uvec = urec.vector

            owner_cands = _search_top(OWNER_COLLECTION, uvec, topn=user_topn)
            base_cnt = _count_matches_user(up, owner_cands)

            # bedrooms suggestion
            suggested_bedrooms = None
            up_relax_bed = dict(up)
            try:
                if up_relax_bed.get("bedrooms") is not None:
                    suggested_bedrooms = max(0, int(up_relax_bed["bedrooms"]) - room_delta)
                    up_relax_bed["bedrooms"] = suggested_bedrooms
            except Exception:
                pass
            cnt_bed = _count_matches_user(up_relax_bed, owner_cands) if suggested_bedrooms is not None else base_cnt
            delta_bed = cnt_bed - base_cnt

            bedrooms_message = ""
            if suggested_bedrooms is not None and delta_bed > 0:
                bedrooms_message = (
                    f"Hi! We noticed your application has had limited opportunities. "
                    f"If you can consider places with {suggested_bedrooms} bedroom(s), "
                    f"your potential matches could increase from {base_cnt} to ~{cnt_bed}. "
                    f"This small adjustment might improve your chances of getting invited to a showing."
                )

            # budget suggestion
            suggested_budget = None
            up_relax_price = dict(up)
            try:
                if up_relax_price.get("price") is not None:
                    suggested_budget = float(up_relax_price["price"]) + float(price_delta)
                    up_relax_price["price"] = suggested_budget
            except Exception:
                pass
            cnt_price = _count_matches_user(up_relax_price, owner_cands) if suggested_budget is not None else base_cnt
            delta_budget = cnt_price - base_cnt

            budget_message = ""
            if suggested_budget is not None and delta_budget > 0:
                budget_message = (
                    f"Hi! We noticed your application has had limited opportunities. "
                    f"If you can extend your budget to about ${int(suggested_budget)} per month, "
                    f"your potential matches could increase from {base_cnt} to ~{cnt_price}. "
                    f"This may help you receive more viewing invitations."
                )

            w.writerow({
                "user_id": user_id,
                "days_since_application": days,
                "number_of_shows": shows,
                "baseline_matches": base_cnt,
                "relax_bedrooms_minus_matches": cnt_bed,
                "delta_bedrooms": delta_bed,
                "suggested_bedrooms": suggested_bedrooms,
                "bedrooms_message": bedrooms_message,
                "relax_budget_plus_matches": cnt_price,
                "delta_budget": delta_budget,
                "suggested_budget": suggested_budget,
                "budget_message": budget_message,
            })
            written += 1

            processed += 1
            if limit_users is not None and processed >= limit_users:
                break

    print(f"✅ Wrote starved users suggestions → {path} (rows={written})")
    return path

# ----------------- OWNERS: audit & CSV -----------------
def audit_starved_owners_to_csv(
    *,
    price_delta: float = 200.0,
    owner_topn: int = 300,
    days_threshold: int = 30,
    shows_threshold: int = 1,
    limit_owners: Optional[int] = None,
) -> str:
    """
    Find starved owners and simulate:
      - price - price_delta (more affordable)
    Writes a CSV with ready-to-send suggestion messages.
    Returns CSV path.
    """
    _ensure_logs_dir()
    path = f"logs/starved_owners_suggestions_{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv"

    fieldnames = [
        "owner_id",
        "days_since_application",
        "number_of_shows",
        "baseline_matches",
        "relax_price_minus_matches",
        "delta_price",
        "suggested_price",
        "price_message",
    ]

    written = 0
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        processed = 0
        profiles = list(_scroll_profiles(OWNER_PROFILES_COLLECTION))
        print(f"🔎 Auditing {len(profiles)} owner profiles...")

        for prof in tqdm(profiles, desc="Owners audited", unit="owner"):
            p = prof.payload or {}
            days = _days_since(_parse_date(p.get("application_date")))
            try:
                shows = int(p.get("number_of_shows", 0))
            except Exception:
                shows = 0

            if days is None or days < days_threshold or shows > shows_threshold:
                continue

            owner_id = str(prof.id)
            orec = _get_point(OWNER_COLLECTION, owner_id, with_payload=True, with_vectors=True)
            if not orec:
                continue

            op = orec.payload or {}
            ovec = orec.vector

            user_cands = _search_top(USER_COLLECTION, ovec, topn=owner_topn)
            base_cnt = _count_matches_owner(op, user_cands)

            suggested_price = None
            op_relax_price = dict(op)
            try:
                if op_relax_price.get("price") is not None:
                    suggested_price = max(0.0, float(op_relax_price["price"]) - float(price_delta))
                    op_relax_price["price"] = suggested_price
            except Exception:
                pass
            cnt_price = _count_matches_owner(op_relax_price, user_cands) if suggested_price is not None else base_cnt
            delta_price = cnt_price - base_cnt

            price_message = ""
            if suggested_price is not None and delta_price > 0:
                price_message = (
                    f"Hi! Your listing has had limited showings recently. "
                    f"If you consider reducing the monthly price to around ${int(suggested_price)}, "
                    f"your potential matches could increase from {base_cnt} to ~{cnt_price}. "
                    f"This may help attract more qualified applicants and accelerate scheduling."
                )

            w.writerow({
                "owner_id": owner_id,
                "days_since_application": days,
                "number_of_shows": shows,
                "baseline_matches": base_cnt,
                "relax_price_minus_matches": cnt_price,
                "delta_price": delta_price,
                "suggested_price": suggested_price,
                "price_message": price_message,
            })
            written += 1

            processed += 1
            if limit_owners is not None and processed >= limit_owners:
                break

    print(f"✅ Wrote starved owners suggestions → {path} (rows={written})")
    return path



def _safe_int(x):
    try: return int(x)
    except: return 0

def _read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)

def run_owner_starvation_audit_simple(
    *,
    price_delta: float = 200.0,
    owner_topn: int = 300,
    days_threshold: int = 30,
    shows_threshold: int = 1,
    limit_owners: int | None = None,
) -> Tuple[str, str]:
    """
    Runs the owners starvation audit using your existing engine,
    then condenses to a 'messages' CSV and a short summary text.

    Returns:
      messages_csv_path, summary_text
    """
    # 1) Run your existing (full) audit
    raw_csv = audit_starved_owners_to_csv(
        price_delta=price_delta,
        owner_topn=owner_topn,
        days_threshold=days_threshold,
        shows_threshold=shows_threshold,
        limit_owners=limit_owners,
    )
    raw = _read_csv(raw_csv)

    # Empty case
    if raw.empty:
        summary = (
            "Looking through the owners in the system to explore starved ones, "
            "we found 0 owners who have not been shown more than once in the last month."
        )
        # Still emit an empty messages CSV so ops has a file
        ts = datetime.now().strftime('%Y%m%d-%H%M%S')
        out_csv = f"logs/starved_owners_messages_{ts}.csv"
        pd.DataFrame(columns=["owner_id","days_since_application","number_of_shows","suggested_price","delta_price","message"]).to_csv(out_csv, index=False)
        return out_csv, summary

    # 2) Compute simple insights
    total_starved = len(raw)
    raw["delta_price"] = raw["delta_price"].apply(_safe_int)
    gain_mask = raw["delta_price"] > 0
    improved = int(gain_mask.sum())
    avg_gain_matches = float(raw.loc[gain_mask, "delta_price"].mean() or 0.0)

    # 3) Prepare slim messages CSV
    # keep only actionable rows (those with improvement)
    actionable = raw.loc[gain_mask, [
        "owner_id",
        "days_since_application",
        "number_of_shows",
        "suggested_price",
        "delta_price",
        "price_message",
    ]].copy()
    actionable = actionable.rename(columns={"price_message": "message"})

    ts = datetime.now().strftime('%Y%m%d-%H%M%S')
    out_csv = f"logs/starved_owners_messages_{ts}.csv"
    actionable.to_csv(out_csv, index=False)

    # 4) Build short summary text
    summary = (
    f"Starved Owners Audit\n\n"
    f"We identified {total_starved} owners whose listings have not been shown more than once "
    f"in the past month.\n\n"
    f"For {improved} of these owners, reducing the listing price by {int(price_delta)} dollars "
    f"is expected to increase exposure, with an average improvement of approximately "
    f"{avg_gain_matches:.1f} additional matched candidates.\n\n"
    f"An email will be sent to each relevant owner with the suggested adjustment."
)

    return out_csv, summary


def run_user_starvation_audit_simple(
    *,
    price_delta: float = 200.0,
    room_delta: int = 1,
    user_topn: int = 300,
    days_threshold: int = 30,
    shows_threshold: int = 1,
    limit_users: int | None = None,
) -> Tuple[str, str]:
    """
    Runs the users starvation audit using your existing engine,
    then condenses to a 'messages' CSV and a short summary text.

    Returns:
      messages_csv_path, summary_text
    """
    # 1) Run your existing (full) audit
    raw_csv = audit_starved_users_to_csv(
        price_delta=price_delta,
        room_delta=room_delta,
        user_topn=user_topn,
        days_threshold=days_threshold,
        shows_threshold=shows_threshold,
        limit_users=limit_users,
    )
    raw = _read_csv(raw_csv)

    # Empty case
    if raw.empty:
        summary = (
            "Looking through the renters in the system to explore starved ones, "
            "we found 0 renters who have not been shown more than once in the last month."
        )
        ts = datetime.now().strftime('%Y%m%d-%H%M%S')
        out_csv = f"logs/starved_users_messages_{ts}.csv"
        pd.DataFrame(columns=[
            "user_id","days_since_application","number_of_shows",
            "suggested_bedrooms","delta_bedrooms",
            "suggested_budget","delta_budget",
            "message"
        ]).to_csv(out_csv, index=False)
        return out_csv, summary

    # 2) Compute simple insights
    total_starved = len(raw)
    raw["delta_bedrooms"] = raw["delta_bedrooms"].apply(_safe_int)
    raw["delta_budget"]  = raw["delta_budget"].apply(_safe_int)

    bed_gain_mask = raw["delta_bedrooms"] > 0
    bud_gain_mask = raw["delta_budget"] > 0

    bed_improved = int(bed_gain_mask.sum())
    bud_improved = int(bud_gain_mask.sum())
    avg_bed_gain = float(raw.loc[bed_gain_mask, "delta_bedrooms"].mean() or 0.0)
    avg_bud_gain = float(raw.loc[bud_gain_mask, "delta_budget"].mean() or 0.0)

    # 3) Prepare slim messages CSV (one row per user if any improvement; prefer combined message)
    messages_rows = []
    for _, r in raw.iterrows():
        has_bed = r.get("delta_bedrooms", 0) > 0
        has_bud = r.get("delta_budget", 0) > 0
        if not (has_bed or has_bud):
            continue

        # Prefer a combined note if both apply
        parts = []
        if has_bed:
            parts.append(f"consider {int(r.get('suggested_bedrooms', 0))} bedroom(s), ~+{int(r.get('delta_bedrooms', 0))} matches")
        if has_bud:
            parts.append(f"extend budget to ${int(r.get('suggested_budget', 0))}, ~+{int(r.get('delta_budget', 0))} matches")
        msg = " & ".join(parts)

        messages_rows.append({
            "user_id": r["user_id"],
            "days_since_application": r.get("days_since_application"),
            "number_of_shows": r.get("number_of_shows"),
            "suggested_bedrooms": (int(r["suggested_bedrooms"]) if has_bed else ""),
            "delta_bedrooms": (int(r["delta_bedrooms"]) if has_bed else ""),
            "suggested_budget": (int(r["suggested_budget"]) if has_bud else ""),
            "delta_budget": (int(r["delta_budget"]) if has_bud else ""),
            "message": msg,
        })

    ts = datetime.now().strftime('%Y%m%d-%H%M%S')
    out_csv = f"logs/starved_users_messages_{ts}.csv"
    pd.DataFrame(messages_rows).to_csv(out_csv, index=False)

    # 4) Build short summary text
    summary = (
    f"Starved Renters Audit\n\n"
    f"We identified {total_starved} renters who have not been invited to property show more than once"
    f"in the past month.\n\n"
    f"For {bed_improved} renters, decreasing the minimum bedroom requirement by {int(room_delta)} "
    f"could improve matching, with an average gain of ~{avg_bed_gain:.1f} additional candidates.\n\n"
    f"For {bud_improved} renters, increasing their budget by {int(price_delta)} dollars "
    f"could improve matching, with an average gain of ~{avg_bud_gain:.1f} additional candidates.\n\n"
    f"An email will be sent to each relevant renter with the suggested adjustment."
)

    return out_csv, summary