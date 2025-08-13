from qdrant_client import QdrantClient , models
from qdrant_client.models import VectorParams, Distance, Record,Filter, FieldCondition, MatchValue, Record
from qdrant_client.http import exceptions as qdrant_exc
from typing import Iterable, List, Union, Dict, Any, Optional, Tuple
import random
import time
import hashlib


# Qdrant configuration
AQDRANT_API_KEY= "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.79h_Yg9qXYtICf-fs1CMuMdK5Rw13OnE_DJR953fYQ4"
QDRANT_URL = "https://3cf2848d-0574-468d-a996-0efabdea92b9.us-west-1-0.aws.cloud.qdrant.io"
SIM_COLLECTION = "similarity_collection"
OWNER_COLLECTION = "owner_agent_listings"
USER_COLLECTION  = "user_agent_listings"
OWNER_PROFILES_COLLECTION = "owner_profiles"
USER_PROFILES_COLLECTION = "user_profiles"

client = QdrantClient(
    url=QDRANT_URL,
    api_key=AQDRANT_API_KEY,
)

def upload_to_qdrant(points: list,COLLECTION_NAME : str):
    if not points:
        print("No points to upload.")
        return

    vector_size = len(points[0].vector)

    # Create the collection ONLY if it doesn't exist
    try:
        client.get_collection(collection_name=COLLECTION_NAME)
        # Optional: sanity-check vector size / distance here if you want
        # info = client.get_collection(COLLECTION_NAME)
        # assert info.vectors_count is not None
    except qdrant_exc.UnexpectedResponse:  # collection not found
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )

    # Upsert = insert new or replace existing by ID (no reset)
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
        wait=True,          # wait until indexed
    )

    print(f"✅ Upserted {len(points)} points into '{COLLECTION_NAME}' without resetting the collection.")



def scroll_qdrant(COLLECTION_NAME : str, limit: int):
    """
    Scroll through the Qdrant collection to fetch points.
    """
    return client.scroll(
        collection_name=COLLECTION_NAME,
        limit=limit # Number of points to fetch
        
    )


def count_qdrant(COLLECTION_NAME: str) -> int:
    """
    Count the number of points in a Qdrant collection.
    """
    return client.count(
        collection_name=COLLECTION_NAME,
        exact=True  
    ).count


from typing import Iterable, List, Union
from qdrant_client.models import Record

def get_points_by_ids(
    COLLECTION_NAME: str,
    ids: Iterable[Union[int, str]],
    with_payload: bool = True,
    with_vectors: bool = False,
) -> List[Record]:
    """
    Retrieve multiple points by ID from a collection.
    Returns a list of qdrant_client.models.Record (may be empty if IDs not found).
    """
    # Qdrant accepts str or int IDs; pass-through as-is
    return client.retrieve(
        collection_name=COLLECTION_NAME,
        ids=list(ids),
        with_payload=with_payload,
        with_vectors=with_vectors,
    )

def get_point_by_id(
    COLLECTION_NAME: str,
    point_id: Union[int, str],
    with_payload: bool = True,
    with_vectors: bool = False,
) -> Union[Record, None]:
    """
    Retrieve a single point by ID. Returns None if not found.
    """
    records = get_points_by_ids(COLLECTION_NAME, [point_id], with_payload, with_vectors)
    return records[0] if records else None




# ---------- internal helpers ----------

def _scroll_all_filtered(collection: str, flt: Filter, page_size: int = 1000) -> List[Record]:
    """Scroll all records matching a filter."""
    out: List[Record] = []
    next_page = None
    while True:
        records, next_page = client.scroll(
            collection_name=collection,
            with_payload=True,
            with_vectors=False,   # vectors are [score]; payload has 'score' too
            scroll_filter=flt,
            limit=page_size,
            offset=next_page,
        )
        out.extend(records)
        if next_page is None:
            break
    return out

def _as_match_dict(r: Record) -> Dict[str, Any]:
    """Normalize a Record into a simple dict."""
    p = r.payload or {}
    # Prefer payload['score']; if absent, fall back to 1D vector value
    score = p.get("score")
    if score is None and r.vector:
        try:
            score = float(r.vector[0])
        except Exception:
            score = None
    return {
        "pair_id": str(r.id),
        "owner_id": p.get("owner_id"),
        "user_id": p.get("user_id"),
        "score": score,
        "payload": p,
    }

def _pair_id(user_id: str, owner_id: str) -> str:
    """Must match the one used when writing pairs."""
    return hashlib.md5(f"{user_id}::{owner_id}".encode("utf-8")).hexdigest()

# ---------- public API ----------

def get_matches_by_owner(owner_id: str, top_k: Optional[int] = 20) -> List[Dict[str, Any]]:
    """
    Return matches for a given owner_id, sorted by score (desc).
    """
    flt = Filter(must=[FieldCondition(key="owner_id", match=MatchValue(value=owner_id))])
    recs = _scroll_all_filtered(SIM_COLLECTION, flt)
    rows = sorted((_as_match_dict(r) for r in recs), key=lambda x: (x["score"] is not None, x["score"]), reverse=True)
    return rows[:top_k] if top_k is not None else rows

def get_matches_by_user(user_id: str, top_k: Optional[int] = 20) -> List[Dict[str, Any]]:
    """
    Return matches for a given user_id, sorted by score (desc).
    """
    flt = Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))])
    recs = _scroll_all_filtered(SIM_COLLECTION, flt)
    rows = sorted((_as_match_dict(r) for r in recs), key=lambda x: (x["score"] is not None, x["score"]), reverse=True)
    return rows[:top_k] if top_k is not None else rows

def get_owner_user_match(owner_id: str, user_id: str) -> Optional[Dict[str, Any]]:
    """
    Return the single match for (owner_id, user_id) if it exists.
    Uses the deterministic pair id, so it's O(1) retrieve.
    """
    pid = _pair_id(user_id, owner_id)
    recs = client.retrieve(
        collection_name=SIM_COLLECTION,
        ids=[pid],
        with_payload=True,
        with_vectors=True,
    )
    if not recs:
        return None
    return _as_match_dict(recs[0])



def fetch_payloads_by_ids(collection: str, ids: List[str]) -> Dict[str, dict]:
    """Batch-retrieve payloads for a list of point IDs. Returns {id: payload}."""
    if not ids:
        return {}
    recs: List[Record] = client.retrieve(
        collection_name=collection,
        ids=ids,
        with_payload=True,
        with_vectors=False
    )
    return {str(r.id): (r.payload or {}) for r in recs}

def summarize_payload(p: dict) -> str:
    """Compact one-line summary from your schema."""
    state = p.get("state")
    if isinstance(state, list):
        state = ", ".join(state)
    return (
        f"[state: {state or '-'} | price: {p.get('price', '-')} | "
        f"bedrooms: {p.get('bedrooms', '-')} | available_from: {p.get('available_from', '-')}] "
        f"soft: {p.get('soft_attributes', '')[:120]}{'…' if p.get('soft_attributes') and len(p['soft_attributes'])>120 else ''}"
    )

def print_user_matches_with_details(matches: List[dict], top_k: int = 5):
    """Given matches from match_for_new_user (has owner_id), print full owner details."""
    rows = matches[:top_k]
    owner_ids = [str(r["owner_id"]) for r in rows]
    owner_payloads = fetch_payloads_by_ids(OWNER_COLLECTION, owner_ids)

    print(f"Top {len(rows)} owners for this user:")
    for r in rows:
        oid = str(r["owner_id"])
        print(f"  score={r['score']:.4f}  owner_id={oid}")
        print(f"    {summarize_payload(owner_payloads.get(oid, {}))}")
    return rows

def print_owner_matches_with_details(matches: List[dict], top_k: int = 5):
    """Given matches from match_for_new_owner (has user_id), print full user details."""
    rows = matches[:top_k]
    user_ids = [str(r["user_id"]) for r in rows]
    user_payloads = fetch_payloads_by_ids(USER_COLLECTION, user_ids)

    print(f"Top {len(rows)} users for this owner:")
    for r in rows:
        uid = str(r["user_id"])
        print(f"  score={r['score']:.4f}  user_id={uid}")
        print(f"    {summarize_payload(user_payloads.get(uid, {}))}")
    
    return rows


def _retrieve_payload(collection: str, point_id: str) -> Optional[dict]:
    """Fetch a single point's payload by id."""
    try:
        recs: List[Record] = client.retrieve(
            collection_name=collection,
            ids=[point_id],
            with_payload=True,
            with_vectors=False
        )
        return (recs[0].payload or {}) if recs else None
    except qdrant_exc.UnexpectedResponse:
        return None


def _fetch_user_payloads(user_ids: List[str]) -> Dict[str, dict]:
    """Batch-retrieve user payloads by IDs."""
    if not user_ids:
        return {}
    try:
        recs: List[Record] = client.retrieve(
            collection_name=USER_COLLECTION,
            ids=user_ids,
            with_payload=True,
            with_vectors=False
        )
        return {str(r.id): (r.payload or {}) for r in recs}
    except qdrant_exc.UnexpectedResponse:
        return {}
    

def _iter_owner_ids(batch: int = 1000,owner_collection: str = OWNER_COLLECTION):
    """Robust iterator over owner IDs with lightweight backoff on transient errors."""
    next_page = None
    while True:
        try:
            recs, next_page = client.scroll(
                collection_name=owner_collection,
                with_payload=False,
                with_vectors=False,
                limit=batch,
                offset=next_page
            )
        except qdrant_exc.UnexpectedResponse:
            # tiny jittered backoff and retry
            time.sleep(0.5 + random.random())
            continue

        for r in recs:
            yield str(r.id)

        if next_page is None:
            break



def upload_profile(profile, type_):
    """Upload profile dict to the relevant Qdrant collection."""
    collection_name = OWNER_PROFILES_COLLECTION if type_ == "owner" else USER_PROFILES_COLLECTION

    # Ensure collection exists (non-vector, payload-only storage)
    if collection_name not in [c.name for c in client.get_collections().collections]:
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=1, distance=models.Distance.COSINE)
        )

    # Use profile_id as point ID, dummy vector [0.0] since no embeddings are needed
    client.upsert(
        collection_name=collection_name,
        points=[
            models.PointStruct(
                id=profile["profile_id"],
                vector=[0.0],
                payload=profile
            )
        ]
    )
    print(f"✅ Uploaded profile to {collection_name}: {profile['full_name']}")