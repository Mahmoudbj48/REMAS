from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, Record
from qdrant_client.http import exceptions as qdrant_exc
from typing import Iterable, List, Union

# Qdrant configuration
AQDRANT_API_KEY= "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.79h_Yg9qXYtICf-fs1CMuMdK5Rw13OnE_DJR953fYQ4"
QDRANT_URL = "https://3cf2848d-0574-468d-a996-0efabdea92b9.us-west-1-0.aws.cloud.qdrant.io"

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
