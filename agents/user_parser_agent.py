import json
import hashlib
from typing import List
from langchain.schema import HumanMessage, SystemMessage
from langchain_community.callbacks.manager import get_openai_callback
from utils.logger import init_log_file, log_token_usage
from embeddings.embedding_model import embed_texts
from qdrant_client.models import PointStruct
DEFAULT_PROMPT = """
You are a helpful assistant that extracts structured information from user descriptions of their ideal property.

Your task is to:
1. Extract HARD preferences (structured fields) using the SAME KEYS as owner listings:
   - state                        # list of areas/regions mentioned
   - picture_url                  # always null for customers
   - price                        # numeric, interpret as MAX BUDGET
   - num_bedrooms                 # integer, interpret as MIN BEDROOMS
   - available_from (if given)    # desired move-in month or date

2. Extract SOFT preferences: a free-text summary of lifestyle or environmental desires (e.g., "quiet", "pet-friendly", "good for studying")

Return JSON in the following format (keys and types must match the owner schema):
{
  "hard_attributes": {
    "state": ["Brooklyn", "Queens"],
    "picture_url": null,
    "price": 2000,
    "num_bedrooms": 2,
    "available_from": "September"
  },
  "soft_attributes": "quiet, suitable for studying, no smoking, pet-friendly"
}

Notes:
- If multiple states/areas are mentioned, return them as a list.
- Keep soft preferences as a single comma-separated string.
- If a field is missing or unclear, use null.
- Interpret 'price' as the user's maximum budget and 'num_bedrooms' as the minimum bedrooms desired.

Output rules:
- Return ONLY valid JSON (no backticks, no explanations).
- Use the SAME keys and types as owner listings:
  - hard_attributes.state: array of strings (areas/regions mentioned)
  - hard_attributes.picture_url: null
  - hard_attributes.price: number, interpret as MAX MONTHLY BUDGET in USD (strip currency symbols)
  - hard_attributes.num_bedrooms: integer, interpret as MIN BEDROOMS
  - hard_attributes.available_from: desired move-in month (e.g., "September") or null
- Normalize months to English month names in Title Case (e.g., "September").
- If a range is given for budget or bedrooms, use the MAX budget and MIN bedrooms respectively.
- If multiple states/areas are mentioned, include all in the array.
- If a field is missing or unclear, use null.
- soft_attributes: single comma-separated string describing lifestyle preferences, ≤ 300 characters.
"""


# -------------------------
# Parsing helpers (shared)
# -------------------------

def clean_json_string(raw_output: str) -> str:
    # Remove Markdown code block formatting if present
    if raw_output.strip().startswith("```"):
        lines = raw_output.strip().splitlines()
        return "\n".join(line for line in lines if not line.strip().startswith("```"))
    return raw_output

def generate_deterministic_id(obj: dict) -> str:
    """
    Stable ID based on the semantic content (works for owners or users).
    """
    raw = json.dumps({
        "soft": obj.get("soft_attributes", ""),
        "hard": obj.get("hard_attributes", {})
    }, sort_keys=True)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()

# ---------------------------------------------------------
# USER PARSER (mirrors owner parser but with user prompt)
# ---------------------------------------------------------

# Expect you defined DEFAULT_PROMPT_USER elsewhere per your aligned schema
# If not, pass it explicitly via the `prompt` parameter.
def run_user_parser_agent(user_input: str, llm, log_file: str = "logs/token_usage.csv") -> dict:
    """
    Parse a renter's free-text description into the SAME schema as owner:
    hard_attributes: { state: [...], picture_url: null, price: <max budget>, num_bedrooms: <min>, available_from: <month|null> }
    soft_attributes: "<comma-separated string>"
    """
    init_log_file(log_file)

    messages = [
        SystemMessage(content=DEFAULT_PROMPT.strip()),
        HumanMessage(content=user_input)
    ]

    with get_openai_callback() as cb:
        response = llm.invoke(messages)
        log_token_usage(log_file, cb, user_input)

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

    # Normalize: ensure picture_url is explicitly null for users
    hard = parsed.get("hard_attributes", {}) or {}
    if "picture_url" not in hard or hard.get("picture_url") is not None:
        hard["picture_url"] = None
    parsed["hard_attributes"] = hard

    return parsed

# -------------------------------------------------------------------
# PREP for vector DB (if you want to store user queries as vectors)
# -------------------------------------------------------------------

def prepare_users_for_qdrant(user_profiles: List[dict]) -> List[PointStruct]:
    """
    Takes a list of parsed user dicts (from run_user_parser_agent).
    Returns a list of Qdrant PointStructs (embedding on soft_attributes).
    picture_url is always null for users (enforced).
    """
    # Extract soft attributes text
    soft_texts = [(p.get("soft_attributes", "") or "") for p in user_profiles]

    # Embed
    vectors = embed_texts(soft_texts)

    points: List[PointStruct] = []
    for i, profile in enumerate(user_profiles):
        hard = profile.get("hard_attributes", {}) or {}
        soft = profile.get("soft_attributes", "") or ""

        # Enforce null picture_url for users
        hard["picture_url"] = None

        # Deterministic ID for dedupe/upsert
        point_id = generate_deterministic_id(profile)

        payload = {
            "listing_id": point_id,         # consistent with owner payload key
            "state": hard.get("state"),
            "picture_url": None,
            "price": hard.get("price"),     # interpreted as MAX budget per your prompt
            "bedrooms": hard.get("num_bedrooms"),  # interpreted as MIN bedrooms
            "available_from": hard.get("available_from"),
            "soft_attributes": soft,
            "source": "user_query"          # helpful to distinguish in the same collection
        }

        points.append(PointStruct(
            id=point_id,
            vector=vectors[i],
            payload=payload
        ))

    return points
