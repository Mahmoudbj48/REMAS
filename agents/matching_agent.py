# agents/matching_agent.py
import time
import hashlib
from typing import List, Optional, Union

from qdrant_client.models import (
    Filter, FieldCondition, MatchAny, MatchValue, Range, PointStruct, VectorParams, Distance
)
from qdrant_client.http import exceptions as qexc

from utils.qdrant_connection import (
    client,                      # QdrantClient instance from your helper module
    get_point_by_id,             # helper you added earlier
)

OWNER_COLLECTION = "owner_agent_listings"
USER_COLLECTION  = "user_agent_listings"
SIM_COLLECTION   = "similarity_collection"

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

    # Optional: availability basic equality if provided by user
    avail = user_payload.get("available_from")
    if isinstance(avail, str) and avail.strip():
        must.append(FieldCondition(key="available_from", match=MatchValue(value=avail.strip())))

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
