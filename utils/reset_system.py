# Reset system:
# - Drop & rebuild `similarity_collection` by calling your existing `match_for_new_owner`
# - Reset number_of_shows=0 for *all* user/owner profiles
from typing import List, Optional
from tqdm.auto import tqdm
from qdrant_client.http import exceptions as qexc
from qdrant_client.models import PayloadSchemaType

from utils.qdrant_connection import client, OWNER_COLLECTION,OWNER_PROFILES_COLLECTION, USER_PROFILES_COLLECTION


# Use your existing matching function that persists matches into `similarity_collection`
from agents.matching_agent import match_for_new_owner


SIM_COLLECTION = "similarity_collection"


def _iter_ids(collection: str, page: int = 1000) -> List[str]:
    """Yield ALL point ids from a collection (no payload/vectors)."""
    out = []
    offset = None
    while True:
        recs, offset = client.scroll(
            collection_name=collection,
            with_payload=False,
            with_vectors=False,
            limit=page,
            offset=offset
        )
        if not recs:
            break
        out.extend(str(r.id) for r in recs)
        if offset is None:
            break
    return out


def _reset_profile_shows(collection: str) -> int:
    """Set number_of_shows=0 for every profile in the given collection."""
    updated = 0
    offset = None
    while True:
        recs, offset = client.scroll(
            collection_name=collection,
            with_payload=True,
            with_vectors=False,
            limit=1000,
            offset=offset
        )
        if not recs:
            break
        for r in recs:
            payload = r.payload or {}
            if payload.get("number_of_shows") != 0:
                payload["number_of_shows"] = 0
                client.set_payload(collection_name=collection, payload=payload, points=[r.id])
            updated += 1
        if offset is None:
            break
    return updated


def _drop_similarity_collection():
    try:
        client.delete_collection(collection_name=SIM_COLLECTION)
        print(f"🗑️  Deleted '{SIM_COLLECTION}'.")
    except qexc.UnexpectedResponse as e:
        # It's fine if it didn't exist
        if "doesn't exist" in str(e).lower():
            print(f"ℹ️  '{SIM_COLLECTION}' did not exist; continuing.")
        else:
            raise


def _ensure_similarity_indexes():
    """Create payload indexes required for filtered queries (owner_id, user_id)."""
    for field, schema in [
        ("owner_id", PayloadSchemaType.KEYWORD),
        ("user_id",  PayloadSchemaType.KEYWORD),
        # Optional but useful later:
        # ("score",     PayloadSchemaType.FLOAT),
        # ("timestamp", PayloadSchemaType.FLOAT),
    ]:
        try:
            client.create_payload_index(
                collection_name=SIM_COLLECTION,
                field_name=field,
                field_schema=schema,
            )
            print(f"✅ Created index on '{field}'")
        except qexc.UnexpectedResponse as e:
            # Treat "already exists" as OK
            if "already exists" in str(e).lower():
                print(f"ℹ️  Index on '{field}' already exists")
            elif "Collection" in str(e) and "doesn't exist" in str(e):
                # If collection isn't created yet, we'll retry after first upserts
                print(f"ℹ️  '{SIM_COLLECTION}' not ready for index yet; will retry after upserts.")
                return False
            else:
                raise
    return True


def reset_system(*, top_k_per_owner: int = 50):
    """
    1) Drop `similarity_collection`
    2) Rebuild it by iterating all owners and calling `match_for_new_owner(owner_id, top_k=...)`
       (this uses your project’s existing save path for pairs)
    3) Recreate payload indexes on (owner_id, user_id)
    4) Reset number_of_shows=0 for all owner & user profiles
    """
    print("🚀 Resetting system…")

    # 1) Drop similarity collection (so we start clean)
    _drop_similarity_collection()

    # 2) Rebuild using your existing matching pipeline
    owner_ids = _iter_ids(OWNER_COLLECTION)
    print(f"🔁 Recomputing similarity pairs via match_for_new_owner for {len(owner_ids)} owners...")
    built_ok = False
    for oid in tqdm(owner_ids, desc="Rebuilding similarity_collection", unit="owner"):
        try:
            # This function is expected to:
            #  - search users
            #  - write pairs into `similarity_collection`
            match_for_new_owner(oid, top_k=top_k_per_owner)
            built_ok = True
        except Exception as e:
            # Log error but continue with other owners
            tqdm.write(f"  ⚠️ owner_id={oid} error: {e}")

    if not built_ok:
        print("⚠️ No pairs were written. Ensure `match_for_new_owner` persists to "
              f"'{SIM_COLLECTION}' and that collections are populated.")
    else:
        # 3) Ensure payload indexes exist (needed for filtered queries later)
        ok = _ensure_similarity_indexes()
        if not ok:
            # If the collection didn't exist at index creation time, try again now
            _ensure_similarity_indexes()

    # 4) Reset number_of_shows for profiles
    print("🔄 Resetting number_of_shows to 0 on profiles…")
    owners_reset = _reset_profile_shows(OWNER_PROFILES_COLLECTION)
    users_reset  = _reset_profile_shows(USER_PROFILES_COLLECTION)

    print(f"✅ Done. Rebuilt similarity from {len(owner_ids)} owners.")
    print(f"   Profiles reset → owners: {owners_reset}, users: {users_reset}")
