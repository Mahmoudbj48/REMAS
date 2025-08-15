# agents/matching_agent.py
import time
import hashlib
from typing import List, Optional, Union, Dict, Any
from utils.qdrant_connection import (
    OWNER_COLLECTION, USER_COLLECTION,SIM_COLLECTION,
    get_matches_by_owner, get_matches_by_user,
    fetch_payloads_by_ids, get_point_by_id
)

from qdrant_client.models import (
    Filter, FieldCondition, MatchAny, MatchValue, Range, PointStruct, VectorParams, Distance
)
from qdrant_client.http import exceptions as qexc

from utils.qdrant_connection import (
    client,                      # QdrantClient instance from your helper module
    get_point_by_id,             # helper you added earlier
)


# -----------------------------
# Helpers
# -----------------------------

def _ensure_similarity_collection():
    """Create the similarity collection if missing (vector is the similarity score as 1D)."""
    try:
        client.get_collection(collection_name=SIM_COLLECTION)
    except qexc.UnexpectedResponse:
        client.create_collection(
            collection_name=SIM_COLLECTION,
            vectors_config=VectorParams(size=1, distance=Distance.COSINE),
        )

def _normalize_state_list(value) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        v = value.strip()
        return [v] if v else None
    if isinstance(value, list):
        cleaned = [str(x).strip() for x in value if str(x).strip()]
        return cleaned or None
    return None

def _safe_int(v) -> Optional[int]:
    try:
        if v is None: return None
        return int(v)
    except Exception:
        return None

def _safe_float(v) -> Optional[float]:
    try:
        if v is None: return None
        return float(v)
    except Exception:
        return None

def _build_owner_filter_for_user(user_payload: dict) -> Optional[Filter]:
    """
    Build a Qdrant filter to select owner listings compatible with a given user preference.
    user_payload fields:
      - state: List[str]
      - price: user's MAX budget
      - bedrooms: user's MIN bedrooms
      - available_from: optional month name (string)
    """
    must = []
    states = _normalize_state_list(user_payload.get("state"))
    if states:
        must.append(FieldCondition(key="state", match=MatchAny(any=states)))

    max_budget = _safe_float(user_payload.get("price"))
    if max_budget is not None:
        # owner price must be <= user max budget
        must.append(FieldCondition(key="price", range=Range(lte=max_budget)))

    min_beds = _safe_int(user_payload.get("bedrooms"))
    if min_beds is not None:
        # owner bedrooms must be >= user min bedrooms
        must.append(FieldCondition(key="bedrooms", range=Range(gte=min_beds)))

    # # Optional: availability basic equality if provided by user
    # avail = user_payload.get("available_from")
    # if isinstance(avail, str) and avail.strip():
    #     must.append(FieldCondition(key="available_from", match=MatchValue(value=avail.strip())))

    return Filter(must=must) if must else None

def _build_user_filter_for_owner(owner_payload: dict) -> Optional[Filter]:
    """
    Build a Qdrant filter to select user queries compatible with a given owner listing.
    owner_payload fields:
      - state: List[str] or str
      - price: owner's price
      - bedrooms: owner's number of bedrooms
      - available_from: optional month name
    """
    must = []
    states = _normalize_state_list(owner_payload.get("state"))
    if states:
        must.append(FieldCondition(key="state", match=MatchAny(any=states)))

    owner_price = _safe_float(owner_payload.get("price"))
    if owner_price is not None:
        # user budget must be >= owner price
        must.append(FieldCondition(key="price", range=Range(gte=owner_price)))

    owner_bedrooms = _safe_int(owner_payload.get("bedrooms"))
    if owner_bedrooms is not None:
        # user min bedrooms must be <= owner bedrooms
        must.append(FieldCondition(key="bedrooms", range=Range(lte=owner_bedrooms)))

    avail = owner_payload.get("available_from")
    if isinstance(avail, str) and avail.strip():
        must.append(FieldCondition(key="available_from", match=MatchValue(value=avail.strip())))

    return Filter(must=must) if must else None

def _pair_id(user_id: str, owner_id: str) -> str:
    """Deterministic id for the (user, owner) pair to avoid duplicates."""
    s = f"{user_id}::{owner_id}"
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def _save_similarity_pairs(pairs: List[dict]):
    """
    Save pairs to `similarity_collection`.
    Each pair dict: { "user_id": str, "owner_id": str, "score": float, "filter_used": dict, "timestamp": float }
    Vector is [score] for convenient sorting.
    """
    _ensure_similarity_collection()
    points: List[PointStruct] = []
    for p in pairs:
        pid = _pair_id(p["user_id"], p["owner_id"])
        points.append(
            PointStruct(
                id=pid,
                vector=[float(p["score"])],
                payload=p
            )
        )
    # upsert in one go (few dozens) – adjust if you expect thousands
    client.upsert(collection_name=SIM_COLLECTION, points=points, wait=True)

# -----------------------------
# Public APIs
# -----------------------------

def match_for_new_user(user_point_id: Union[str, int], top_k: int = 50) -> List[dict]:
    """
    Given a newly created user point id (from user_parser_agent), find matching owner listings.
    - Filters owner listings using user's structured preferences (state, price, bedrooms, available_from).
    - Performs vector search using the user's vector.
    - Saves top pairs into `similarity_collection`.
    Returns a list of dicts for the top matches.
    """
    # 1) Get user point (vector + payload)
    user_rec = get_point_by_id(USER_COLLECTION, user_point_id, with_payload=True, with_vectors=True)
    if user_rec is None:
        raise ValueError(f"User point '{user_point_id}' not found in '{USER_COLLECTION}'")

    user_vec = user_rec.vector
    user_payload = user_rec.payload or {}
    filt = _build_owner_filter_for_user(user_payload)

    # 2) Search owners by similarity with filter
    results = client.search(
        collection_name=OWNER_COLLECTION,
        query_vector=user_vec,
        limit=top_k,
        query_filter=filt
    )
    # Qdrant returns 'score' with cosine similarity (higher is better) when collection distance is COSINE.

    # 3) Save pairs
    pairs = []
    now = time.time()
    for r in results:
        pairs.append({
            "user_id": str(user_rec.id),
            "owner_id": str(r.id),
            "score": float(r.score),
            "filter_used": {
                "collection": OWNER_COLLECTION,
                "state": user_payload.get("state"),
                "price_max": user_payload.get("price"),
                "bedrooms_min": user_payload.get("bedrooms"),
                "available_from": user_payload.get("available_from"),
            },
            "timestamp": now,
        })
    if pairs:
        _save_similarity_pairs(pairs)

    return pairs

def match_for_new_owner(owner_point_id: Union[str, int], top_k: int = 50) -> List[dict]:
    """
    Given a newly created owner point id (from owner_parser_agent), find matching user queries.
    - Filters users using owner's structured attributes (state, price, bedrooms, available_from).
    - Performs vector search using the owner's vector.
    - Saves top pairs into `similarity_collection`.
    Returns a list of dicts for the top matches.
    """
    # 1) Get owner point
    owner_rec = get_point_by_id(OWNER_COLLECTION, owner_point_id, with_payload=True, with_vectors=True)
    if owner_rec is None:
        raise ValueError(f"Owner point '{owner_point_id}' not found in '{OWNER_COLLECTION}'")

    owner_vec = owner_rec.vector
    owner_payload = owner_rec.payload or {}
    filt = _build_user_filter_for_owner(owner_payload)

    # 2) Search users by similarity with filter
    results = client.search(
        collection_name=USER_COLLECTION,
        query_vector=owner_vec,
        limit=top_k,
        query_filter=filt
    )

    # 3) Save pairs
    pairs = []
    now = time.time()
    for r in results:
        pairs.append({
            "user_id": str(r.id),
            "owner_id": str(owner_rec.id),
            "score": float(r.score),
            "filter_used": {
                "collection": USER_COLLECTION,
                "state": owner_payload.get("state"),
                "price_owner": owner_payload.get("price"),
                "bedrooms_owner": owner_payload.get("bedrooms"),
                "available_from": owner_payload.get("available_from"),
            },
            "timestamp": now,
        })
    if pairs:
        _save_similarity_pairs(pairs)

    return pairs



# === Estimated match summaries for immediate feedback ===
from typing import List, Dict, Any, Optional
from utils.qdrant_connection import (
    OWNER_COLLECTION, USER_COLLECTION,
    get_matches_by_owner, get_matches_by_user,
    fetch_payloads_by_ids, get_point_by_id
)

def _as_list(x):
    if x is None:
        return []
    return x if isinstance(x, list) else [x]

def _availability_ok(user_avail: Optional[str], owner_avail: Optional[str]) -> bool:
    if not user_avail or not owner_avail:
        return True
    return str(user_avail).strip().lower()[:3] == str(owner_avail).strip().lower()[:3]

def _hard_match(user_p: dict, owner_p: dict) -> bool:
    u_states = set(map(lambda s: str(s).strip().lower(), _as_list(user_p.get("state"))))
    o_states = set(map(lambda s: str(s).strip().lower(), _as_list(owner_p.get("state"))))
    state_ok = (not u_states or not o_states) or (u_states & o_states)

    try:
        price_ok = (owner_p.get("price") is None) or (user_p.get("price") is None) or (float(owner_p["price"]) <= float(user_p["price"]))
    except Exception:
        price_ok = True

    try:
        beds_ok = (owner_p.get("bedrooms") is None) or (user_p.get("bedrooms") is None) or (int(owner_p["bedrooms"]) >= int(user_p["bedrooms"]))
    except Exception:
        beds_ok = True

    avail_ok = _availability_ok(user_p.get("available_from"), owner_p.get("available_from"))
    return bool(state_ok and price_ok and beds_ok and avail_ok)

def _norm_id(s: Optional[str]) -> str:
    # Lowercase, strip, remove hyphens
    if s is None:
        return ""
    return str(s).strip().lower().replace("-", "")

def _rank_in_inverse(inverse_rows: List[Dict[str, Any]], key: str, target_id: str) -> Optional[int]:
    """Return 0-based rank if found, else None. Compares normalized IDs."""
    targetN = _norm_id(target_id)
    for i, row in enumerate(inverse_rows):
        cand = row.get(key)
        if _norm_id(cand) == targetN:
            return i
    return None

def summarize_estimated_for_user(user_id: str, user_matches: List[Dict[str, Any]], check_top_k: int = 5) -> str:
    user_rec = get_point_by_id(USER_COLLECTION, user_id)
    user_payload = (user_rec.payload if user_rec else {}) or {}

    owner_ids = [str(m["owner_id"]) for m in user_matches]
    owner_payloads = fetch_payloads_by_ids(OWNER_COLLECTION, owner_ids)

    total_considered = len(user_matches)
    top1_count = 0
    top5_count = 0
    hard_fit_count = 0
    RANK_WINDOW = max(50, check_top_k)

    for m in user_matches:
        oid = str(m["owner_id"])
        inverse = get_matches_by_owner(oid, top_k=RANK_WINDOW)
        rank = _rank_in_inverse(inverse, key="user_id", target_id=user_id)
        if rank is not None:
            if rank == 0:
                top1_count += 1
            if rank < check_top_k:
                top5_count += 1
        if _hard_match(user_payload, owner_payloads.get(oid, {}) or {}):
            hard_fit_count += 1

    lines = []
    lines.append("=== Estimated opportunities for you ===")
    lines.append(f"- You appear as the #1 candidate in ~{top1_count} listing(s).")
    lines.append(f"- You appear in the top {check_top_k} for ~{top5_count} listing(s).")
    lines.append(f"- You have a strong ‘hard-attribute’ fit with ~{hard_fit_count} listing(s).")
    lines.append(f"- Total listings evaluated in this preview: {total_considered}")
    if user_matches:
        best = user_matches[0]
        lines.append(f"- Your current best score: {best.get('score', 0):.4f}")# (owner_id={best.get('owner_id')})")
    lines.append("")
    lines.append("Note: These are early estimates based on current matches.")
    lines.append("Final invitations depend on scheduling, fairness (giving chances to those with fewer shows),")
    lines.append("and listing popularity. You may not be invited to all matched properties.")
    lines.append("")
    return "\n".join(lines)


def summarize_estimated_for_owner(owner_id: str, owner_matches: List[Dict[str, Any]], check_top_k: int = 5) -> str:
    owner_rec = get_point_by_id(OWNER_COLLECTION, owner_id)
    owner_payload = (owner_rec.payload if owner_rec else {}) or {}

    user_ids = [str(m["user_id"]) for m in owner_matches]
    user_payloads = fetch_payloads_by_ids(USER_COLLECTION, user_ids)

    total_considered = len(owner_matches)
    top1_count = 0
    top5_count = 0
    hard_fit_count = 0
    RANK_WINDOW = max(50, check_top_k)

    for m in owner_matches:
        uid = str(m["user_id"])
        inverse = get_matches_by_user(uid, top_k=RANK_WINDOW)
        rank = _rank_in_inverse(inverse, key="owner_id", target_id=owner_id)
        if rank is not None:
            if rank == 0:
                top1_count += 1
            if rank < check_top_k:
                top5_count += 1
        if _hard_match(user_payloads.get(uid, {}) or {}, owner_payload):
            hard_fit_count += 1

    lines = []
    lines.append("=== Estimated demand for your listing ===")
    lines.append(f"- Your listing appears as the #1 match for ~{top1_count} user(s).")
    lines.append(f"- Your listing appears in the top {check_top_k} for ~{top5_count} user(s).")
    lines.append(f"- There are ~{hard_fit_count} user(s) whose requirements strongly fit your listing.")
    lines.append(f"- Total users evaluated in this preview: {total_considered}")
    if owner_matches:
        best = owner_matches[0]
        lines.append(f"- Best current candidate score: {best.get('score', 0):.4f}")# (user_id={best.get('user_id')})")
    lines.append("")
    lines.append("Note: These are early estimates to help you gauge interest.")
    lines.append("Actual showings are scheduled by our agent considering fairness, user availability,")
    lines.append("and platform-wide demand management.")
    lines.append("")
    return "\n".join(lines)

