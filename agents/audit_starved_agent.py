# Starvation audit (read-only) → CSVs with ready-to-send suggestions
import os
import csv
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from qdrant_client.models import Record

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
        for prof in _scroll_profiles(USER_PROFILES_COLLECTION):
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
        for prof in _scroll_profiles(OWNER_PROFILES_COLLECTION):
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