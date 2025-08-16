# app.py
import os
import streamlit as st
from datetime import datetime
from utils.qdrant_connection import summarize_shows_text , organize_dataset_after_showings
from agents.user_parser_agent import invoke_user_parser_agent
from agents.owner_parser_agent import invoke_owner_parser_agent
from agents.matching_agent import (
    match_for_new_user, match_for_new_owner,
    summarize_estimated_for_user, summarize_estimated_for_owner,
)
from agents.audit_starved_agent import (
    run_owner_starvation_audit_simple,
    run_user_starvation_audit_simple,
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

    state = st.selectbox("State", ALLOWED_STATES, index=0)
    num_rooms = st.number_input("Number of rooms (min bedrooms)", min_value=0, max_value=20, step=1, value=2)
    price = st.number_input("Max price (USD/month)", min_value=0, step=50, value=2000)
    month = st.selectbox("Available from", MONTHS, index=8)
    soft = st.text_area("Soft preferences (comma-separated)", placeholder="quiet, near cafes, gym, pool")

    # Lock immediately on click
    run_btn = st.button(
        "Find matches",
        key="btn_find_matches",
        disabled=is_locked("renter_locked"),
        on_click=lock_once,
        args=("renter_locked",),
    )

    if run_btn:  # button was clicked this run
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
            except Exception as e:
                st.error(f"Error: {e}")

    if st.button("← Back"):
        unlock_all()
        st.session_state.page = "home"


def owner_page():
    st.header("Owner — Describe your property")

    state = st.selectbox("State", ALLOWED_STATES, index=4)
    num_rooms = st.number_input("Number of rooms (bedrooms)", min_value=0, max_value=20, step=1, value=3)
    price = st.number_input("Asking price (USD/month)", min_value=0, step=50, value=2100)
    month = st.selectbox("Available from", MONTHS, index=8)
    soft = st.text_area("Soft description (comma-separated)", placeholder="quiet street, pool, gym, remote-work friendly")

    # Lock immediately on click
    run_btn = st.button(
        "Find candidates",
        key="btn_find_candidates",
        disabled=is_locked("owner_locked"),
        on_click=lock_once,
        args=("owner_locked",),
    )

    if run_btn:
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

    if st.button("Run daily decisions and create CSV", key="btn_run_daily"):
        with st.spinner("Running daily decisions…"):
            try:
                results, csv_path = run_daily_decisions(max_invites=int(max_invites))
                organize_dataset_after_showings(results=results)

                summary_text = summarize_shows_text(results)
                csv_bytes = open(csv_path, "rb").read()
                csv_name = os.path.basename(csv_path)

                # 🔒 persist outputs across reruns (e.g., when clicking download buttons)
                st.session_state["daily_results"] = {
                    "summary_text": summary_text,
                    "csv_bytes": csv_bytes,
                    "csv_name": csv_name,
                    "max_invites": int(max_invites),
                    "rows": len(results),
                }

                st.success(f"Done. Scheduled up to {max_invites} invitations. Returned {len(results)} rows.")
            except Exception as e:
                st.error(f"Error: {e}")

    # Always render saved outputs if present (survive reruns)
    if "daily_results" in st.session_state:
        dr = st.session_state["daily_results"]
        st.subheader("Scheduling Summary")
        st.code(dr["summary_text"])

        st.download_button(
            "Download CSV",
            data=dr["csv_bytes"],
            file_name=dr["csv_name"],
            mime="text/csv",
            key="dl_csv",
        )
        st.download_button(
            "Download Summary (TXT)",
            data=dr["summary_text"].encode("utf-8"),
            file_name=f"showings_summary_{datetime.now().strftime('%Y%m%d-%H%M%S')}.txt",
            mime="text/plain",
            key="dl_txt",
        )

        # Optional: let user clear the saved outputs
        if st.button("Clear results", key="btn_clear_daily"):
            del st.session_state["daily_results"]
            st.toast("Results cleared")

    # 2) Audit user starvation
    st.subheader("2) Audit User Starvation")
    if st.button("Run user starvation audit"):
        with st.spinner("Auditing starved users…"):
            try:
                users_csv, users_summary = run_user_starvation_audit_simple()
                st.markdown("#### Summary (Users)")
                st.write(users_summary)
                with open(users_csv, "rb") as fh:
                    st.download_button("Download users messages CSV", data=fh.read(), file_name=os.path.basename(users_csv), mime="text/csv")
            except Exception as e:
                st.error(f"Error: {e}")

    # 3) Audit owner starvation
    st.subheader("3) Audit Owner Starvation")
    if st.button("Run owner starvation audit"):
        with st.spinner("Auditing starved owners…"):
            try:
                owners_csv, owners_summary = run_owner_starvation_audit_simple()
                st.markdown("#### Summary (Owners)")
                st.write(owners_summary)
                with open(owners_csv, "rb") as fh:
                    st.download_button("Download owners messages CSV", data=fh.read(), file_name=os.path.basename(owners_csv), mime="text/csv")
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
