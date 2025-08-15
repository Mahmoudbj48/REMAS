# app.py
import streamlit as st
from datetime import datetime
from utils.qdrant_connection import client , organize_dataset_after_showings
from agents.user_parser_agent import invoke_user_parser_agent
from agents.owner_parser_agent import invoke_owner_parser_agent
from agents.matching_agent import (
    match_for_new_user, match_for_new_owner,
    summarize_estimated_for_user, summarize_estimated_for_owner,
)
from agents.manage_showings_agent import run_daily_decisions
from utils.reset_system import reset_system  # wrap your reset_system() here

ALLOWED_STATES = ["New York", "San Francisco", "Los Angeles", "Chicago", "Miami"]
MONTHS = ["January","February","March","April","May","June","July","August","September","October","November","December"]

st.set_page_config(page_title="REMAS", page_icon="🏠", layout="centered")

# ---- helpers for one-click locking ----
def lock_once(key: str):
    st.session_state[key] = True

def is_locked(key: str) -> bool:
    return bool(st.session_state.get(key, False))

def unlock_all():
    # reset locks when navigating back to home
    for k in ["renter_locked", "owner_locked"]:
        st.session_state[k] = False

def main_page():
    st.title("REMAS")
    st.write("Choose how you want to enter:")
    col1, col2, col3 = st.columns(3)
    if col1.button("Enter as Realtor 🧑‍💼", type="primary"):
        st.session_state.page = "realtor"
    if col2.button("Enter as Renter 🧑‍🦱"):
        st.session_state.page = "renter"
    if col3.button("Enter as Property Owner 🏡"):
        st.session_state.page = "owner"

def renter_page():
    st.header("Renter — Tell us what you need")

    # No forms → Enter key will NOT submit anything
    state = st.selectbox("State", ALLOWED_STATES, index=0)
    num_rooms = st.number_input("Number of rooms (min bedrooms)", min_value=0, max_value=20, step=1, value=2)
    price = st.number_input("Max price (USD/month)", min_value=0, step=50, value=2000)
    month = st.selectbox("Available from", MONTHS, index=8)  # default September
    soft = st.text_area("Soft preferences (comma-separated)", placeholder="quiet, near cafes, gym, pool")

    run_btn = st.button("Find matches", disabled=is_locked("renter_locked"))
    if run_btn and not is_locked("renter_locked"):
        user_input = f"""
        Looking for a {num_rooms}-bedroom in {state}.
        Budget up to ${price}/month. Move-in around {month}.
        Preferences: {soft}
        """.strip()
        with st.spinner("Uploading & matching…"):
            try:
                user_id = invoke_user_parser_agent(user_input)
                matches = match_for_new_user(user_id)
                st.subheader("Summary")
                summary_text = summarize_estimated_for_user(user_id, matches, check_top_k=5)
                st.code(summary_text)
                lock_once("renter_locked")  # lock after successful run
            except Exception as e:
                st.error(f"Error: {e}")

    if st.button("← Back"):
        unlock_all()
        st.session_state.page = "home"

def owner_page():
    st.header("Owner — Describe your property")

    # No forms → Enter key will NOT submit anything
    state = st.selectbox("State", ALLOWED_STATES, index=4)  # default Miami
    num_rooms = st.number_input("Number of rooms (bedrooms)", min_value=0, max_value=20, step=1, value=3)
    price = st.number_input("Asking price (USD/month)", min_value=0, step=50, value=2100)
    month = st.selectbox("Available from", MONTHS, index=8)
    soft = st.text_area("Soft description (comma-separated)", placeholder="quiet street, pool, gym, remote-work friendly")

    run_btn = st.button("Find candidates", disabled=is_locked("owner_locked"))
    if run_btn and not is_locked("owner_locked"):
        owner_input = f"""
        {num_rooms}-bedroom apartment in {state}.
        ${price}/month. Available in {month}.
        Features: {soft}
        """.strip()
        with st.spinner("Uploading & matching…"):
            try:
                owner_id = invoke_owner_parser_agent(owner_input)
                matches = match_for_new_owner(owner_id)
                st.subheader("Summary")
                summary_text = summarize_estimated_for_owner(owner_id, matches, check_top_k=5)
                st.code(summary_text)
                lock_once("owner_locked")  # lock after successful run
            except Exception as e:
                st.error(f"Error: {e}")

    if st.button("← Back"):
        unlock_all()
        st.session_state.page = "home"

def realtor_page():
    st.header("Realtor Tools")
    st.caption("Admin utilities for managing shows and audits.")

    # 1) Manage Show Day Agent
    st.subheader("1) Manage Show Day Agent")
    max_invites = st.selectbox("Maximum invitations to schedule", options=[5, 10, 15, 20], index=0)
    if st.button("Run daily decisions and create CSV"):
        with st.spinner("Running daily decisions…"):
            try:
                results, csv_path = run_daily_decisions(max_invites=int(max_invites))
                organize_dataset_after_showings(results=results)
                st.success(f"Done. Scheduled up to {max_invites} invitations. Returned {len(results)} rows.")
                st.download_button("Download CSV", data=open(csv_path, "rb").read(),
                                   file_name=csv_path.split("/")[-1], mime="text/csv")
            except Exception as e:
                st.error(f"Error: {e}")

    # 2) Audit user starvation (placeholder)
    st.subheader("2) Audit User Starvation")
    if st.button("Run user starvation audit"):
        with st.spinner("Computing…"):
            try:
                st.info("TODO: implement audit—e.g., users with high days_since_application and number_of_shows=0.")
            except Exception as e:
                st.error(f"Error: {e}")

    # 3) Audit owner starvation
    st.subheader("3) Audit Owner Starvation")
    if st.button("Run owner starvation audit"):
        with st.spinner("Computing…"):
            try:
                st.info("TODO: implement audit—e.g., owners with many days_since_application and no shows.")
            except Exception as e:
                st.error(f"Error: {e}")

    # 4) Reset system
    st.subheader("4) Reset the System")
    if st.button("Reset"):
        with st.spinner("Resetting…"):
            try:
                reset_system(top_k_per_owner=50)
                st.success("Reset complete.")
            except Exception as e:
                st.error(f"Error: {e}")

    if st.button("← Back"):
        unlock_all()
        st.session_state.page = "home"

# ---- Router ----
if "page" not in st.session_state:
    st.session_state.page = "home"
    unlock_all()

if st.session_state.page == "home":
    main_page()
elif st.session_state.page == "renter":
    renter_page()
elif st.session_state.page == "owner":
    owner_page()
elif st.session_state.page == "realtor":
    realtor_page()
