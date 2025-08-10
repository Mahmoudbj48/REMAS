# pipeline/synthetic_from_openai.py
import os, json, time
import pandas as pd
from typing import List
from openai import OpenAI
from tqdm import tqdm

# ---- project imports you already have ----
from agents.owner_parser_agent import run_owner_parser_agent, prepare_listings_for_qdrant
from utils.qdrant_connection import upload_to_qdrant
from agents.user_parser_agent import run_user_parser_agent, prepare_users_for_qdrant
from config.llm_config import llm  

CSV_PATH = "data/processed/cleaned_airbnb_listings.csv"
SELECTED_PATH = "data/processed/selected_airbnb_listings.csv"
OUT_DIR  = "data/synthetic"
N_SAMPLES = 1000
CITY_FILTER = {"SAN FRANCISCO", "NEW YORK", "NEW YORK CITY", "NYC"}

# -------- OpenAI client (non-Azure) --------
# Prefer env var: OPENAI_API_KEY
api_k = "my-own-key"
client = OpenAI(api_key=api_k)

MODEL_ID = "gpt-4o-mini"
TEMP = 0.3

# -------- prompts --------
OWNER_NORMALIZE_PROMPT = """
You will receive a short, possibly informal property description sourced from Airbnb data.
Your task is to rewrite it as a concise, standard real-estate OWNER listing (2–4 sentences), 
suitable for a property rental website.

Requirements:
- Include the location (city and neighborhood if available).
- Mention the number of bedrooms.
- Highlight notable perks, features, or constraints (e.g., "pet-friendly", "near subway", "no smoking").
- Include monthly price if present.
- Include availability date or time if present.
- Keep it professional, clear, and free of unnecessary fluff.

Example:
Input:
"Cozy 2BR in SF close to the park, great for students. $2500, move in July."
Output:
"Spacious 2-bedroom apartment in San Francisco, close to Golden Gate Park. Ideal for students and working professionals. $2,500 per month, available starting July."
Output plain text only.
"""

CUSTOMER_SYNTH_PROMPT = """
You will receive a standard real-estate OWNER listing.
Your task is to rewrite it as a RENTER request (1–3 sentences) that naturally matches the property described.

Requirements:
- Mention the location.
- Mention the minimum number of bedrooms required.
- Include lifestyle or environmental preferences hinted in the original (e.g., "pet-friendly", "quiet neighborhood").
- State a maximum monthly budget using the same number from the original price.
- Include desired move-in date if present.
- Make it sound natural, as if written by a potential tenant.

Example:
Input:
"Spacious 2-bedroom apartment in San Francisco, close to Golden Gate Park. Ideal for students and working professionals. $2,500 per month, available starting July."
Output:
"I'm looking for a 2-bedroom apartment in San Francisco near Golden Gate Park. My budget is $2,500 per month, and I'd like to move in by July."
Output plain text only.
"""


# -------- helpers --------
def _chat(messages: List[dict]) -> str:
    r = client.chat.completions.create(
        model=MODEL_ID,
        messages=messages,
        temperature=TEMP
    )
    return r.choices[0].message.content.strip()

def select_seed_listings(csv_path: str, n: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["state"] = df["state"].astype(str).str.upper().str.strip()
    df = df[df["state"].isin(CITY_FILTER)]
    if len(df) > n:
        df = df.sample(n=n, random_state=42)
    return df.reset_index(drop=True)

def build_airbnb_source_text(row: pd.Series) -> str:
    """Compose a compact source text from structured fields (since Airbnb text may be odd)."""
    parts = []
    b = row.get("bedrooms")
    if pd.notna(b):
        parts.append(f"{int(b)}-bedroom")
    city = str(row.get("state") or "").title()
    if city:
        parts.append(f"in {city}")
    soft = (row.get("soft_attributes") or "").strip()
    price = row.get("price")
    avail = (row.get("available_from") or "")
    s = f"{' '.join(parts)}. {soft}".strip()
    if pd.notna(price):
        s += f" Price: ${int(float(price))}/month."
    if isinstance(avail, str) and avail:
        s += f" Available from {avail.title()}."
    return s[:800]

def owner_normalize(text: str) -> str:
    return _chat([
        {"role": "system", "content": OWNER_NORMALIZE_PROMPT},
        {"role": "user", "content": text}
    ])

def customer_from_owner(owner_text: str) -> str:
    return _chat([
        {"role": "system", "content": CUSTOMER_SYNTH_PROMPT},
        {"role": "user", "content": owner_text}
    ])

def save_jsonl(path: str, rows: List[dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            json.dump(r, f, ensure_ascii=False)
            f.write("\n")


# -------- main pipeline --------
def main():
    df = select_seed_listings(CSV_PATH, N_SAMPLES)
    if df.empty:
        print("No rows after filtering."); return
    print(f"Selected {len(df)} rows from {sorted(CITY_FILTER)}")
    df.to_csv(SELECTED_PATH, index=False)

    # Step 1: build source (Airbnb-ish) text → normalize to owner style
    owner_inputs: List[str] = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Normalize owner texts"):
        src = build_airbnb_source_text(row)
        owner_inputs.append(owner_normalize(src))
        time.sleep(0.02)  # gentle pacing

    save_jsonl(f"{OUT_DIR}/owner_inputs.jsonl", [{"owner_input": t} for t in owner_inputs])

    # Step 2: generate matching customer queries
    customer_inputs: List[str] = []
    for owner_text in tqdm(owner_inputs, total=len(owner_inputs), desc="Generate customer queries"):
        customer_inputs.append(customer_from_owner(owner_text))
        time.sleep(0.02)

    save_jsonl(f"{OUT_DIR}/customer_inputs.jsonl", [{"customer_input": t} for t in customer_inputs])

    # Step 3: parse → prepare → upload (owners)
    print("Parsing owners (Azure llm) …")
    parsed_owners = [run_owner_parser_agent(t, llm, log_file="logs/token_usage.csv") for t in tqdm(owner_inputs)]
    owner_points = prepare_listings_for_qdrant(parsed_owners)
    upload_to_qdrant(owner_points,"owner_agent_listings")

    # Step 4: parse → prepare → upload (customers)
    print("Parsing customers (Azure llm) …")
    parsed_customers = [run_user_parser_agent(t, llm, log_file="logs/token_usage.csv")
                        for t in tqdm(customer_inputs)]
    customer_points = prepare_users_for_qdrant(parsed_customers)
    upload_to_qdrant(customer_points,"user_agent_listings")

    print("✅ Done: uploaded normalized owners + synthetic customers.")




if __name__ == "__main__":
    main()
