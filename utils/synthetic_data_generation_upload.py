# pipeline/synthetic_from_openai.py
import  time
from typing import List
from tqdm import tqdm
import random
from datetime import datetime, timedelta

# ---- project imports you already have ----
from utils.qdrant_connection import upload_to_qdrant , upload_profile

# --- 1. Create synthetic data pools ---
first_names = [
    "John", "Alice", "Michael", "Sara", "David", "Emma", "Robert", "Olivia",
    "Daniel", "Sophia", "James", "Mia", "William", "Ava", "Joseph", "Isabella",
    "Thomas", "Emily", "Matthew", "Charlotte"
]
last_names = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
    "Thomas", "Taylor", "Moore", "Jackson", "Martin"
]
emails = [
    "mahmoudbj48@gmail.com", "mhmod.0586367379@gmail.com", "moodmath48@gmail.com",
]
# Add 7 synthetic emails
emails += [f"user{i}@example.com" for i in range(1, 8)]

phones = [f"+1-555-{random.randint(1000,9999)}" for _ in range(20)]

# Generate list of dates between 2025-07-01 and 2025-08-12
start_date = datetime(2025, 7, 10)
end_date = datetime(2025, 8, 12)
date_list = [
    (start_date + timedelta(days=i)).strftime("%Y-%m-%d") 
    for i in range((end_date - start_date).days + 1)
]

# --- 2. Create random profile payload ---
def make_profile(profile_id, type_):
    return {
        "profile_id": profile_id,
        "type": type_,
        "full_name": f"{random.choice(first_names)} {random.choice(last_names)}",
        "email": random.choice(emails),
        "phone": random.choice(phones),
        "application_date": random.choice(date_list),
        "number_of_shows": "0"
    }

# --- 3. For each new entity, create & upload profile ---
def create_and_upload_profile(profile_id, type_):
    """Generate a synthetic profile and upload to Qdrant."""
    profile = make_profile(profile_id, type_)
    upload_profile(profile, type_)
    return profile



