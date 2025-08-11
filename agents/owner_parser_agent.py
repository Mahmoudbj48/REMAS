import json
from langchain.schema import HumanMessage, SystemMessage
from langchain_community.callbacks.manager import get_openai_callback
from utils.logger import init_log_file, log_token_usage
from embeddings.embedding_model import embed_texts
from qdrant_client.models import PointStruct
from utils.qdrant_connection import upload_to_qdrant
from config.llm_config import llm
import hashlib


DEFAULT_PROMPT = """
You are an assistant that extracts structured information from property descriptions submitted by owners.

Your task is to:
1. Extract HARD attributes:
   - state
   - picture_url (if given)
   - price
   - num_bedrooms
   - available_from (if given)

2. Extract SOFT attributes: a free-text summary of qualitative aspects (e.g., "quiet neighborhood", "great for students", "no pets")

Return JSON in the following format:
{
  "hard_attributes": {
    "state": ["Astoria", "Queens"],
    "picture_url": "https://example.com/image.jpg",
    "price": 2300,
    "num_bedrooms": 2,
    "available_from": "October"
  },
  "soft_attributes": "quiet neighborhood, ideal for students or remote workers, no pets, no smoking"
}

Note:
- If multiple states are mentioned, return them as a list.
- Keep soft attributes as a single comma-separated string.
- If a field is missing or unclear, use null.

Output rules:
- Return ONLY valid JSON (no backticks, no explanations).
- Types must match exactly:
  - hard_attributes.state: array of strings (always an array, even if one value)
  - hard_attributes.price: number (monthly USD). If a range is given, use the asking price; if unclear, use null.
  - hard_attributes.num_bedrooms: integer
  - hard_attributes.available_from: month name (e.g., "September") or null
  - hard_attributes.picture_url: string URL if explicitly provided; otherwise null (do NOT fabricate).
- Normalize months to English month names in Title Case (e.g., "September").
- Trim whitespace, no trailing commas.
- soft_attributes: single comma-separated string, ≤ 300 characters.
"""

def clean_json_string(raw_output: str) -> str:
    # Remove Markdown code block formatting if present
    if raw_output.strip().startswith("```"):
        lines = raw_output.strip().splitlines()
        # Remove first and last lines (e.g., ```json and ```)
        return "\n".join(line for line in lines if not line.strip().startswith("```"))
    return raw_output

def run_owner_parser_agent(owner_input: str, log_file: str = "logs/token_usage.csv") -> dict:
    init_log_file(log_file)

    messages = [
        SystemMessage(content=DEFAULT_PROMPT.strip()),
        HumanMessage(content=owner_input)
    ]

    with get_openai_callback() as cb:
        response = llm.invoke(messages)
        log_token_usage(log_file, cb, owner_input)

    cleaned_output = clean_json_string(response.content)

    try:
        parsed = json.loads(cleaned_output)
    except json.JSONDecodeError:
        parsed = {
            "hard_attributes": {},
            "soft_attributes": "",
            "error": "Failed to parse LLM output as JSON",
            "raw_output": response.content
        }

    return parsed



def generate_deterministic_id(listing: dict) -> str:
    raw_text = listing.get("soft_attributes", "") + str(listing.get("hard_attributes", {}))
    return hashlib.md5(raw_text.encode('utf-8')).hexdigest()


def prepare_listings_for_qdrant(listings: list) -> list:
    """
    Takes a list of parsed listing dicts (from run_owner_parser_agent).
    Returns a list of Qdrant PointStructs with a unique 'listing_id' in the payload.
    """
    # Extract soft attributes text
    soft_texts = [listing.get("soft_attributes", "") or "" for listing in listings]

    # Get embeddings
    vectors = embed_texts(soft_texts)

    points = []
    for i, listing in enumerate(listings):
        hard = listing.get("hard_attributes", {})
        soft = listing.get("soft_attributes", "")

        # Generate a deterministic ID based on the listing content
        point_id = generate_deterministic_id(listing)

        # Extract fields with fallback and add listing_id
        payload = {
            "listing_id": point_id, # Unique ID for the listing
            "state": hard.get("state"),
            "picture_url": hard.get("picture_url"),
            "price": hard.get("price"),
            "bedrooms": hard.get("num_bedrooms"),
            "available_from": hard.get("available_from"),
            "soft_attributes": soft,
        }

        point = PointStruct(
            id=point_id,
            vector=vectors[i],
            payload=payload
        )
        points.append(point)

    return points


def invoke_owner_parser_agent(owner_input: str, collection: str = "owner_agent_listings") -> str:
    """
    Parse an owner input, embed, and upload to Qdrant. Returns the point ID.
    """
    # 1) Parse
    parsed_listing = run_owner_parser_agent(owner_input)

    # 2) Prepare (creates deterministic ID & payload)
    points = prepare_listings_for_qdrant([parsed_listing])
    if not points:
        raise ValueError("No point created from parsed listing.")

    # 3) Upload to the owner collection
    upload_to_qdrant(points, collection)

    # 4) Report/return the actual ID (same as payload['listing_id'])
    point_id = points[0].id
    print(f"✅ Uploaded owner listing with ID: {point_id} to '{collection}'")
    return point_id




##