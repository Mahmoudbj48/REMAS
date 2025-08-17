# app.py
import os
import streamlit as st
import json
from datetime import datetime
from utils.qdrant_connection import (
    summarize_shows_text,
    organize_dataset_after_showings,
    _retrieve_payload,
    get_user_profile,
    get_owner_profile,
    get_random_user_owner_ids,
)
from agents.user_parser_agent import invoke_user_parser_agent
from agents.owner_parser_agent import invoke_owner_parser_agent
from agents.matching_agent import (
    match_for_new_user,
    match_for_new_owner,
    summarize_estimated_for_user,
    summarize_estimated_for_owner,
)
from agents.audit_starved_agent import (
    run_owner_starvation_audit_simple,
    run_user_starvation_audit_simple,
)
from agents.manage_showings_agent import run_daily_decisions
from utils.reset_system import reset_system  # wrap your reset_system() here
from feedback.feedback_interface import FeedbackInterface, FeedbackInput

ALLOWED_STATES = ["New York", "San Francisco", "Los Angeles", "Chicago", "Miami"]
MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]

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


def generate_customer_notifications(results, timestamp_str):
    """Generate email messages for scheduled showings"""
    email_messages = []

    for result in results:
        if result.get("error"):
            continue

        decision = result.get("decision", {})
        if decision.get("show") != "1":
            continue

        owner_id = result.get("owner_id")
        num_invites = int(decision.get("num", 0))

        if num_invites == 0:
            continue

        # Get owner information
        try:
            owner_payload = _retrieve_payload("owner_agent_listings", owner_id) or {}
            owner_profile = get_owner_profile(owner_id) or {}
        except Exception:
            owner_payload = {}
            owner_profile = {}

        # Get user information from sample
        sample_users = result.get("sample", [])[:num_invites]

        # Generate email for property owner
        owner_email = generate_owner_email(
            owner_payload, owner_profile, sample_users, timestamp_str
        )
        if owner_email:
            email_messages.append(owner_email)

        # Generate emails for each invited user
        for user_data in sample_users:
            user_id = user_data.get("user_id")
            if not user_id:
                continue

            try:
                user_payload = _retrieve_payload("user_agent_listings", user_id) or {}
                user_profile = get_user_profile(user_id) or {}
            except Exception:
                user_payload = {}
                user_profile = {}

            user_email = generate_user_email(
                user_payload,
                user_profile,
                owner_payload,
                owner_profile,
                user_data.get("score", 0),
                timestamp_str,
            )
            if user_email:
                email_messages.append(user_email)

    return email_messages


def generate_owner_email(owner_payload, owner_profile, invited_users, timestamp_str):
    """Generate email message for property owner"""
    owner_name = owner_profile.get("name", f"Property Owner")
    owner_email = owner_profile.get("email", f"owner@example.com")

    property_desc = f"{owner_payload.get('bedrooms', 'N/A')}-bedroom property"
    if owner_payload.get("price"):
        property_desc += f" (${owner_payload.get('price')}/month)"
    if owner_payload.get("state"):
        property_desc += f" in {owner_payload.get('state')}"

    # Build user list - no scores shown
    user_list = ""
    for i, user_data in enumerate(invited_users, 1):
        user_list += f"{i}. Qualified candidate looking for similar properties\n"

    showing_date = datetime.now().strftime("%A, %B %d, %Y")

    subject = f"Property Showing Scheduled - {len(invited_users)} Interested Candidates"

    body = f"""Dear {owner_name},

Excellent news! We've scheduled a showing for your {property_desc} for {showing_date}.

We've carefully selected {len(invited_users)} qualified candidates who are actively looking for properties like yours:

{user_list}

WHAT'S NEXT:
• Our team will coordinate with interested candidates
• You'll receive confirmation calls from each attendee
• We recommend preparing your property for viewing
• Please ensure the property is accessible on the scheduled date

SHOWING DETAILS:
• Date: {showing_date}
• Candidates Invited: {len(invited_users)}
• Expected Response: Within 24 hours

If you have any questions or need to reschedule, please contact our team immediately.

Best regards,
REMAS Property Management Team
support@remas.com
(555) 123-REMAS

---
Showing Reference: REF{timestamp_str}"""

    return {
        "message_id": f"owner_notification_{owner_profile.get('name', 'unknown')}_{timestamp_str}",
        "recipient": {
            "name": owner_name,
            "email": owner_email,
            "type": "property_owner",
        },
        "email": {"subject": subject, "body": body},
        "metadata": {
            "showing_date": showing_date,
            "candidates_invited": len(invited_users),
            "property_id": owner_payload.get("owner_id", "unknown"),
        },
    }


def generate_user_email(
    user_payload, user_profile, owner_payload, owner_profile, match_score, timestamp_str
):
    """Generate email message for invited user"""
    user_name = user_profile.get("name", f"Valued Customer")
    user_email = user_profile.get("email", f"user@example.com")

    property_desc = f"{owner_payload.get('bedrooms', 'N/A')}-bedroom property"
    if owner_payload.get("price"):
        property_desc += f" for ${owner_payload.get('price')}/month"
    if owner_payload.get("state"):
        property_desc += f" in {owner_payload.get('state')}"

    # Determine message tone based on match score
    if match_score < 0.5:
        opening_line = "We have a property that might interest you! While it may not check every box on your wishlist, it could be a great opportunity worth exploring."
        subject = f"🏠 New Property Opportunity - Worth a Look!"
        encouragement = "Sometimes the best discoveries come from keeping an open mind. We'd love for you to see it in person!"
    elif match_score < 0.7:
        opening_line = "Great news! We found a property that aligns well with what you're looking for."
        subject = f"🏠 Great Property Match Found - Showing Scheduled!"
        encouragement = "This property has caught our attention for you, and we think you'll appreciate what it has to offer."
    else:
        opening_line = "Exciting news! We found an excellent property that matches your preferences beautifully."
        subject = f"🏠 Perfect Property Match - Don't Miss This One!"
        encouragement = (
            "This property really stood out to us as something special for you!"
        )

    # Highlight matching features - more user-friendly
    matching_features = []
    if user_payload.get("bedrooms") == owner_payload.get("bedrooms"):
        matching_features.append(
            f"✓ {owner_payload.get('bedrooms')} bedrooms - exactly what you requested"
        )
    if user_payload.get("state") == owner_payload.get("state"):
        matching_features.append(
            f"✓ Located in {owner_payload.get('state')} - your preferred area"
        )
    if user_payload.get("price") and owner_payload.get("price"):
        if owner_payload.get("price") <= user_payload.get("price"):
            matching_features.append(
                f"✓ Rent at ${owner_payload.get('price')}/month - comfortably within your budget"
            )

    if not matching_features:
        features_text = "• This property offers unique features that might surprise you"
    else:
        features_text = "\n".join(matching_features)

    showing_date = datetime.now().strftime("%A, %B %d, %Y")

    body = f"""Dear {user_name},

{opening_line}

PROPERTY DETAILS:
{property_desc}

WHAT MAKES THIS SPECIAL:
{features_text}

PROPERTY FEATURES:
{owner_payload.get('soft_attributes', 'Charming property with unique character - contact us to learn more!')[:200]}

SHOWING DETAILS:
• Date: {showing_date}
• Property: {owner_payload.get('bedrooms', 'N/A')} bedroom in {owner_payload.get('state', 'N/A')}
• Monthly Rent: ${owner_payload.get('price', 'TBD')}

{encouragement}

TO CONFIRM YOUR ATTENDANCE:
• Reply to this email with "CONFIRM"
• Call us at (555) 123-REMAS
• Text us at (555) 123-REMAS with code: VIEW{timestamp_str[-4:]}

⏰ Please confirm within 24 hours to reserve your viewing time.

We're excited to show you this property!

Warm regards,
The REMAS Team
support@remas.com
(555) 123-REMAS

---
Property Reference: PROP{timestamp_str[-6:]}"""

    return {
        "message_id": f"user_invitation_{user_profile.get('name', 'unknown')}_{timestamp_str}",
        "recipient": {
            "name": user_name,
            "email": user_email,
            "type": "potential_tenant",
        },
        "email": {"subject": subject, "body": body},
        "metadata": {
            "match_score": match_score,
            "showing_date": showing_date,
            "property_price": owner_payload.get("price"),
            "user_budget": user_payload.get("price"),
        },
    }


def save_notifications_json(email_messages, timestamp_str):
    """Save email messages to JSON file"""
    if not email_messages:
        return None

    os.makedirs("logs", exist_ok=True)
    json_path = f"logs/email_messages_{timestamp_str}.json"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at": datetime.now().isoformat(),
                "total_messages": len(email_messages),
                "email_messages": email_messages,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    return json_path


def main_page():
    st.title("🏠 Welcome to REMAS")
    st.markdown("### *Real Estate Matching & Scheduling System*")

    # Welcome message and guide
    st.markdown(
        """
    Welcome to REMAS! Our intelligent platform connects property owners with potential renters through smart matching and automated scheduling.
    
    **How it works:**
    - 🏡 **Property Owners** list their properties and get matched with qualified renters
    - 🧑‍🦱 **Renters** describe their needs and discover perfect property matches  
    - 🧑‍💼 **Realtors** manage the system, schedule showings, and collect feedback
    
    **Getting Started is Easy:**
    """
    )

    # Three columns for user types with descriptions
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 🧑‍💼 Realtor")
        st.markdown("*For real estate professionals*")
        st.markdown("• Run daily matching decisions")
        st.markdown("• Schedule property showings")
        st.markdown("• Audit system performance")
        st.markdown("• Collect feedback")
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Enter as Realtor", type="primary", use_container_width=True):
            st.session_state.page = "realtor"

    with col2:
        st.markdown("#### 🧑‍🦱 Renter")
        st.markdown("*Looking for a place to rent?*")
        st.markdown("• Tell us your preferences")
        st.markdown("• Get matched with properties")
        st.markdown("• View recommendations")
        st.markdown("• Connect with owners")
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Enter as Renter", use_container_width=True):
            st.session_state.page = "renter"

    with col3:
        st.markdown("#### 🏡 Property Owner")
        st.markdown("*Have a property to rent?*")
        st.markdown("• List your property details")
        st.markdown("• Find qualified candidates")
        st.markdown("• Get matching recommendations")
        st.markdown("• Schedule showings easily")
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Enter as Property Owner", use_container_width=True):
            st.session_state.page = "owner"

    st.markdown("---")
    st.markdown(
        "💡 **Tip:** Start by entering your role above to explore the system's capabilities!"
    )


def renter_page():
    st.header("Renter — Tell us what you need")

    state = st.selectbox("State", ALLOWED_STATES, index=0)
    num_rooms = st.number_input(
        "Number of rooms (min bedrooms)", min_value=0, max_value=20, step=1, value=2
    )
    price = st.number_input("Max price (USD/month)", min_value=0, step=50, value=2000)
    month = st.selectbox("Available from", MONTHS, index=8)
    soft = st.text_area(
        "Soft preferences (comma-separated)", placeholder="quiet, near cafes, gym, pool"
    )

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
                summary_text = summarize_estimated_for_user(
                    user_id, matches, check_top_k=5
                )
                st.code(summary_text)
            except Exception as e:
                st.error(f"Error: {e}")

    if st.button("← Back"):
        unlock_all()
        st.session_state.page = "home"


def owner_page():
    st.header("Owner — Describe your property")

    state = st.selectbox("State", ALLOWED_STATES, index=4)
    num_rooms = st.number_input(
        "Number of rooms (bedrooms)", min_value=0, max_value=20, step=1, value=3
    )
    price = st.number_input(
        "Asking price (USD/month)", min_value=0, step=50, value=2100
    )
    month = st.selectbox("Available from", MONTHS, index=8)
    soft = st.text_area(
        "Soft description (comma-separated)",
        placeholder="quiet street, pool, gym, remote-work friendly",
    )

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
                summary_text = summarize_estimated_for_owner(
                    owner_id, matches, check_top_k=5
                )
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
    max_invites = st.selectbox(
        "Maximum invitations to schedule", options=[5, 10, 15, 20], index=0
    )

    st.info(
        "📝 **Note for Lecturer:** This daily decision process would typically run automatically on a weekly schedule in production. For demonstration purposes, we've made it manually triggerable so you can observe the system's functionality in real-time."
    )

    if st.button("Run daily decisions and create CSV", key="btn_run_daily"):
        with st.spinner(
            "🔄 Processing complex matching algorithms and generating recommendations... This may take a moment as we analyze available properties and user preferences."
        ):
            try:
                results, csv_path = run_daily_decisions(max_invites=int(max_invites))
                organize_dataset_after_showings(results=results)

                summary_text = summarize_shows_text(results)
                csv_bytes = open(csv_path, "rb").read()
                csv_name = os.path.basename(csv_path)

                # Generate customer email messages
                timestamp_str = datetime.now().strftime("%Y%m%d-%H%M%S")
                email_messages = generate_customer_notifications(results, timestamp_str)
                json_path = save_notifications_json(email_messages, timestamp_str)

                # Store email messages for preview (no download needed)
                email_data = None
                if json_path and os.path.exists(json_path):
                    with open(json_path, "r", encoding="utf-8") as f:
                        email_data = json.load(f)

                # 🔒 persist outputs across reruns (e.g., when clicking download buttons)
                st.session_state["daily_results"] = {
                    "summary_text": summary_text,
                    "csv_bytes": csv_bytes,
                    "csv_name": csv_name,
                    "email_data": email_data,
                    "email_messages_count": (
                        len(email_messages) if email_messages else 0
                    ),
                    "max_invites": int(max_invites),
                    "rows": len(results),
                }

                st.success(
                    f"Done. Scheduled up to {max_invites} invitations. Generated {len(email_messages) if email_messages else 0} email messages. Returned {len(results)} rows."
                )
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

        # Email messages preview
        if dr.get("email_data") and dr.get("email_messages_count", 0) > 0:
            st.subheader(
                f"📧 Generated Email Messages ({dr.get('email_messages_count', 0)} total)"
            )

            if st.button("Preview Email Messages", key="preview_emails"):
                try:
                    email_data = dr.get("email_data", {})
                    messages = email_data.get("email_messages", [])

                    # Group messages by type
                    owner_messages = [
                        msg
                        for msg in messages
                        if msg.get("recipient", {}).get("type") == "property_owner"
                    ]
                    user_messages = [
                        msg
                        for msg in messages
                        if msg.get("recipient", {}).get("type") == "potential_tenant"
                    ]

                    if owner_messages:
                        st.write("### 🏠 Property Owner Notifications")
                        for i, msg in enumerate(owner_messages[:2]):  # Show first 2
                            with st.expander(
                                f"Owner Email {i+1}: {msg.get('recipient', {}).get('name', 'N/A')}",
                                expanded=(i == 0),
                            ):
                                st.write(
                                    f"**To:** {msg.get('recipient', {}).get('email', 'N/A')}"
                                )
                                st.write(
                                    f"**Subject:** {msg.get('email', {}).get('subject', 'N/A')}"
                                )
                                st.write("**Message Body:**")
                                st.text_area(
                                    "",
                                    value=msg.get("email", {}).get(
                                        "body", "No content"
                                    ),
                                    height=300,
                                    key=f"owner_msg_{i}",
                                    disabled=True,
                                )

                    if user_messages:
                        st.write("### 🏠 User Invitations")
                        for i, msg in enumerate(user_messages[:3]):  # Show first 3
                            with st.expander(
                                f"User Email {i+1}: {msg.get('recipient', {}).get('name', 'N/A')}",
                                expanded=(i == 0),
                            ):
                                st.write(
                                    f"**To:** {msg.get('recipient', {}).get('email', 'N/A')}"
                                )
                                st.write(
                                    f"**Subject:** {msg.get('email', {}).get('subject', 'N/A')}"
                                )
                                st.write("**Message Body:**")
                                st.text_area(
                                    "",
                                    value=msg.get("email", {}).get(
                                        "body", "No content"
                                    ),
                                    height=300,
                                    key=f"user_msg_{i}",
                                    disabled=True,
                                )

                    total_shown = min(2, len(owner_messages)) + min(
                        3, len(user_messages)
                    )
                    if len(messages) > total_shown:
                        st.info(
                            f"Showing {total_shown} of {len(messages)} email messages. Check logs/email_messages_*.json for complete messages."
                        )

                except Exception as e:
                    st.error(f"Error previewing email messages: {e}")

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
                    st.download_button(
                        "Download users messages CSV",
                        data=fh.read(),
                        file_name=os.path.basename(users_csv),
                        mime="text/csv",
                    )
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
                    st.download_button(
                        "Download owners messages CSV",
                        data=fh.read(),
                        file_name=os.path.basename(owners_csv),
                        mime="text/csv",
                    )
            except Exception as e:
                st.error(f"Error: {e}")

    # 4) Feedback Collection
    st.subheader("4) Feedback Collection")
    if st.button("Collect Realtor Feedback"):
        st.session_state.page = "feedback"

    if st.button("← Back"):
        unlock_all()
        st.session_state.page = "home"


def feedback_page():
    st.header("Realtor Feedback Collection")
    st.caption("Provide feedback on user-owner matches to improve the system.")

    # Show success message with dismiss option
    if "feedback_success" in st.session_state:
        col1, col2 = st.columns([4, 1])
        with col1:
            st.success(st.session_state.feedback_success)
        with col2:
            if st.button("✕ Dismiss", key="dismiss_success"):
                del st.session_state.feedback_success
                st.rerun()

    # Initialize feedback interface
    feedback_interface = FeedbackInterface()

    # Auto-populate button for demo
    st.info(
        "📝 **Note for Lecturer:** In production, Owner ID and User ID would be automatically selected from the system interface. For demonstration purposes, we provide a button to randomly populate these fields from existing match data."
    )

    # if st.button("🎲 Auto-fill with Random Match Data (Demo)", key="autofill_ids"):
    try:
        random_user_id, random_owner_id = get_random_user_owner_ids()
        if random_user_id and random_owner_id:
            st.session_state.demo_user_id = random_user_id
            st.session_state.demo_owner_id = random_owner_id
            # st.success(f"✅ Auto-filled with User ID: {random_user_id}, Owner ID: {random_owner_id}")
        else:
            st.warning("⚠️ No match data available for auto-fill")
    except Exception as e:
        st.error(f"Error getting random IDs: {e}")

    # Create form for feedback input with dynamic key for clearing
    if "form_counter" not in st.session_state:
        st.session_state.form_counter = 0

    with st.form(
        f"feedback_form_{st.session_state.form_counter}", clear_on_submit=True
    ):
        st.subheader("Match Information")

        col1, col2 = st.columns(2)
        with col1:
            owner_id = st.text_input(
                "Owner ID",
                placeholder="Enter owner ID",
                value=st.session_state.get("demo_owner_id", ""),
            )
        with col2:
            user_id = st.text_input(
                "User ID",
                placeholder="Enter user ID",
                value=st.session_state.get("demo_user_id", ""),
            )

        st.subheader("Feedback Details")

        col3, col4 = st.columns(2)
        with col3:
            feedback_type = st.selectbox("Feedback Type", ["positive", "negative"])
        with col4:
            feedback_score = st.slider(
                "Feedback Score", min_value=1.0, max_value=5.0, value=3.0, step=0.1
            )

        col5, col6 = st.columns(2)
        with col5:
            match_quality_rating = st.selectbox(
                "Match Quality Rating", [1, 2, 3, 4, 5], index=2
            )
        with col6:
            showing_outcome = st.selectbox(
                "Showing Outcome", ["scheduled", "toured", "leased", "declined"]
            )

        st.subheader("Additional Information")

        realtor_notes = st.text_area(
            "Realtor Notes",
            placeholder="Detailed feedback about the match quality, user satisfaction, etc.",
            height=100,
        )

        suggested_adjustments = st.text_area(
            "Suggested Adjustments (Optional)",
            placeholder="Recommendations for improving future matches",
            height=80,
        )

        # Submit button
        submitted = st.form_submit_button("Submit Feedback")

        if submitted:
            # Validate required fields
            if not owner_id or not user_id or not realtor_notes:
                st.error(
                    "Please fill in all required fields: Owner ID, User ID, and Realtor Notes"
                )
            else:
                try:
                    # Create feedback input object
                    feedback_input = FeedbackInput(
                        owner_id=owner_id,
                        user_id=user_id,
                        feedback_type=feedback_type,
                        feedback_score=feedback_score,
                        match_quality_rating=match_quality_rating,
                        showing_outcome=showing_outcome,
                        realtor_notes=realtor_notes,
                        suggested_adjustments=suggested_adjustments,
                    )

                    # Validate feedback
                    errors = feedback_interface.validate_feedback(feedback_input)
                    if errors:
                        st.error("Validation errors:")
                        for error in errors:
                            st.error(f"• {error}")
                    else:
                        # Save feedback
                        feedback_id = feedback_interface.save_feedback(feedback_input)
                        st.session_state.feedback_success = (
                            f"Feedback saved successfully! ID: {feedback_id}"
                        )
                        st.session_state.form_counter += 1
                        st.rerun()

                except Exception as e:
                    st.error(f"Error saving feedback: {e}")

    # Show recent feedback summary
    st.subheader("Recent Feedback Summary")
    try:
        if st.button("Show Recent Feedback", key="show_recent"):
            feedback_interface.display_feedback_summary(num_recent=5)
    except Exception as e:
        st.error(f"Error displaying summary: {e}")

    if st.button("← Back to Realtor Tools"):
        st.session_state.page = "realtor"


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
elif st.session_state.page == "feedback":
    feedback_page()
