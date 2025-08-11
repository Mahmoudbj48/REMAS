# utils/bootstrap_indexes.py
from qdrant_client import QdrantClient
from qdrant_client.http import exceptions as qexc
from qdrant_client.models import PayloadSchemaType

from utils.qdrant_connection import client  # your shared client

OWNER_COLLECTION = "owner_agent_listings"
USER_COLLECTION  = "user_agent_listings"

def _create_index(collection: str, field: str, schema: PayloadSchemaType):
    try:
        client.create_payload_index(
            collection_name=collection,
            field_name=field,
            field_schema=schema,
            wait=True,
        )
        print(f"Indexed {collection}.{field} as {schema.value}")
    except qexc.UnexpectedResponse as e:
        # already exists or other non-fatal cases
        if "already exists" in str(e).lower():
            print(f"{collection}.{field} index already exists")
        else:
            raise

def ensure_indexes():
    # Exact-match/string filters
    _create_index(OWNER_COLLECTION, "state",           PayloadSchemaType.KEYWORD)
    _create_index(OWNER_COLLECTION, "available_from",  PayloadSchemaType.KEYWORD)
    # Numeric range filters
    _create_index(OWNER_COLLECTION, "price",           PayloadSchemaType.FLOAT)
    _create_index(OWNER_COLLECTION, "bedrooms",        PayloadSchemaType.INTEGER)

    _create_index(USER_COLLECTION,  "state",           PayloadSchemaType.KEYWORD)
    _create_index(USER_COLLECTION,  "available_from",  PayloadSchemaType.KEYWORD)
    _create_index(USER_COLLECTION,  "price",           PayloadSchemaType.FLOAT)     # max budget
    _create_index(USER_COLLECTION,  "bedrooms",        PayloadSchemaType.INTEGER)   # min bedrooms

if __name__ == "__main__":
    ensure_indexes()
    print("✅ Payload indexes ensured.")
