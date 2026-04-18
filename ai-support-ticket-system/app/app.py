import os
import sqlite3
from datetime import datetime

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# =========================================================
# PAGE SETUP
# =========================================================
st.set_page_config(
    page_title="AI Support Ticket System",
    page_icon="🎫",
    layout="wide"
)

# =========================================================
# CONSTANTS
# =========================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

DB_PATH = os.path.join(DATA_DIR, "support_tickets.db")
TRAINING_DATA_PATH = os.path.join(DATA_DIR, "tickets.csv")
DEFAULT_ADMIN_PASSWORD = "admin123"  # change this later if you want

SUPPORT_STAFF_OPTIONS = [
    "Unassigned",
    "Help Desk Team",
    "LMS Support",
    "Zoom Support",
    "Account Support",
    "Software Support"
]

# =========================================================
# STYLING
# =========================================================
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    .sub-text {
        color: #666;
        margin-bottom: 1.2rem;
    }
    .card {
        padding: 1rem;
        border-radius: 14px;
        background: #f8f9fa;
        border: 1px solid #e8e8e8;
        margin-bottom: 1rem;
    }
    .badge {
        display: inline-block;
        padding: 0.35rem 0.65rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.4rem;
        margin-bottom: 0.3rem;
    }
    .badge-high {
        background-color: #fde2e2;
        color: #b42318;
    }
    .badge-medium {
        background-color: #fff1cc;
        color: #b54708;
    }
    .badge-low {
        background-color: #dcfce7;
        color: #15803d;
    }
    .badge-open {
        background-color: #e5e7eb;
        color: #374151;
    }
    .badge-progress {
        background-color: #dbeafe;
        color: #1d4ed8;
    }
    .badge-resolved {
        background-color: #dcfce7;
        color: #15803d;
    }
    .ticket-box {
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid #ececec;
        background-color: white;
        margin-bottom: 0.85rem;
    }
    .small-muted {
        color: #666;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# DATABASE FUNCTIONS
# =========================================================
def get_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tickets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticket_id TEXT UNIQUE,
            submitted_at TEXT,
            updated_at TEXT,
            student_name TEXT,
            school_id TEXT,
            email TEXT,
            issue_area TEXT,
            ticket_text TEXT,
            predicted_category TEXT,
            predicted_urgency TEXT,
            category_confidence REAL,
            urgency_confidence REAL,
            status TEXT,
            assigned_to TEXT,
            internal_notes TEXT,
            notification_sent TEXT
        )
    """)

    conn.commit()
    conn.close()

def insert_ticket(ticket_data):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO tickets (
            ticket_id, submitted_at, updated_at, student_name, school_id, email,
            issue_area, ticket_text, predicted_category, predicted_urgency,
            category_confidence, urgency_confidence, status, assigned_to,
            internal_notes, notification_sent
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        ticket_data["ticket_id"],
        ticket_data["submitted_at"],
        ticket_data["updated_at"],
        ticket_data["student_name"],
        ticket_data["school_id"],
        ticket_data["email"],
        ticket_data["issue_area"],
        ticket_data["ticket_text"],
        ticket_data["predicted_category"],
        ticket_data["predicted_urgency"],
        ticket_data["category_confidence"],
        ticket_data["urgency_confidence"],
        ticket_data["status"],
        ticket_data["assigned_to"],
        ticket_data["internal_notes"],
        ticket_data["notification_sent"]
    ))

    conn.commit()
    conn.close()

def get_all_tickets():
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM tickets ORDER BY id DESC", conn)
    conn.close()
    return df

def update_ticket(ticket_id, status, assigned_to, internal_notes):
    conn = get_connection()
    cursor = conn.cursor()

    updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute("""
        UPDATE tickets
        SET status = ?, assigned_to = ?, internal_notes = ?, updated_at = ?
        WHERE ticket_id = ?
    """, (status, assigned_to, internal_notes, updated_at, ticket_id))

    conn.commit()
    conn.close()

def delete_ticket(ticket_id):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("DELETE FROM tickets WHERE ticket_id = ?", (ticket_id,))

    conn.commit()
    conn.close()

def generate_ticket_id():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM tickets")
    count = cursor.fetchone()[0]

    conn.close()
    return f"TKT-{1001 + count}"

# =========================================================
# ML FUNCTIONS
# =========================================================
@st.cache_resource
def train_models():
    df = pd.read_csv(TRAINING_DATA_PATH)

    X = df["ticket_text"]
    y_category = df["category"]
    y_urgency = df["urgency"]

    vectorizer = TfidfVectorizer()
    X_vectorized = vectorizer.fit_transform(X)

    category_model = LogisticRegression(max_iter=1000)
    category_model.fit(X_vectorized, y_category)

    urgency_model = LogisticRegression(max_iter=1000)
    urgency_model.fit(X_vectorized, y_urgency)

    return vectorizer, category_model, urgency_model

def predict_ticket_details(text, vectorizer, category_model, urgency_model):
    sample_vector = vectorizer.transform([text])

    predicted_category = category_model.predict(sample_vector)[0]
    predicted_urgency = urgency_model.predict(sample_vector)[0]

    category_probs = category_model.predict_proba(sample_vector)[0]
    urgency_probs = urgency_model.predict_proba(sample_vector)[0]

    category_confidence = float(max(category_probs)) * 100
    urgency_confidence = float(max(urgency_probs)) * 100

    return {
        "predicted_category": predicted_category,
        "predicted_urgency": predicted_urgency,
        "category_confidence": round(category_confidence, 2),
        "urgency_confidence": round(urgency_confidence, 2)
    }

# =========================================================
# UI HELPERS
# =========================================================
def urgency_badge(urgency):
    urgency = str(urgency).strip().lower()
    if urgency == "high":
        return '<span class="badge badge-high">High Priority</span>'
    if urgency == "medium":
        return '<span class="badge badge-medium">Medium Priority</span>'
    return '<span class="badge badge-low">Low Priority</span>'

def status_badge(status):
    status_lower = str(status).strip().lower()
    if status_lower == "in progress":
        return '<span class="badge badge-progress">In Progress</span>'
    if status_lower == "resolved":
        return '<span class="badge badge-resolved">Resolved</span>'
    return '<span class="badge badge-open">Open</span>'

def is_valid_email(email):
    return "@" in email and "." in email and len(email.strip()) >= 5

def show_ticket_card(row):
    st.markdown(f"""
        <div class="ticket-box">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap;">
                <div>
                    <strong>{row['ticket_id']}</strong> - {row['student_name']}
                </div>
                <div>
                    {urgency_badge(row['predicted_urgency'])}
                    {status_badge(row['status'])}
                </div>
            </div>
            <div class="small-muted" style="margin-top:0.4rem;">
                Email: {row['email']} | School ID: {row['school_id']} | Assigned To: {row['assigned_to']}
            </div>
            <div class="small-muted" style="margin-top:0.25rem;">
                Submitted: {row['submitted_at']} | Updated: {row['updated_at']}
            </div>
            <div style="margin-top:0.7rem;">
                <strong>Issue Area:</strong> {row['issue_area']}<br>
                <strong>Predicted Category:</strong> {row['predicted_category']} ({row['category_confidence']}%)<br>
                <strong>Predicted Urgency:</strong> {row['predicted_urgency']} ({row['urgency_confidence']}%)
            </div>
            <div style="margin-top:0.7rem;">
                <strong>Ticket:</strong><br>
                {row['ticket_text']}
            </div>
            <div style="margin-top:0.7rem;">
                <strong>Internal Notes:</strong><br>
                {row['internal_notes'] if row['internal_notes'] else 'No notes yet.'}
            </div>
        </div>
    """, unsafe_allow_html=True)

# =========================================================
# APP STARTUP
# =========================================================
init_db()
vectorizer, category_model, urgency_model = train_models()

# =========================================================
# HEADER
# =========================================================
st.markdown('<div class="main-title">AI Support Ticket System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-text">A support workflow app that collects tickets, predicts urgency and category with machine learning, stores requests in a database, and helps support staff manage them through resolution.</div>',
    unsafe_allow_html=True
)

# =========================================================
# SIDEBAR / ROLE SELECTION
# =========================================================
with st.sidebar:
    st.header("Access")
    user_mode = st.radio("Choose view", ["Student Portal", "Tech Support Portal"])

    if user_mode == "Tech Support Portal":
        admin_password = st.text_input("Admin Password", type="password")
        is_admin = admin_password == DEFAULT_ADMIN_PASSWORD
        if is_admin:
            st.success("Admin access granted.")
        elif admin_password:
            st.error("Incorrect password.")
    else:
        is_admin = False

    st.markdown("---")
    st.caption("Demo admin password: admin123")

# =========================================================
# STUDENT PORTAL
# =========================================================
if user_mode == "Student Portal":
    st.markdown("### Submit a Support Ticket")
    st.write("Please enter your contact information and describe the issue so a support staff member can follow up with you.")

    col1, col2 = st.columns(2)

    with col1:
        student_name = st.text_input("Full Name")
        school_id = st.text_input("School ID")

    with col2:
        email = st.text_input("School Email")
        issue_area = st.selectbox(
            "Issue Area",
            ["D2L / LMS", "Zoom", "Password / Login", "Software Installation", "Email Access", "General Support", "Other"]
        )

    ticket_text = st.text_area(
        "Describe your issue",
        placeholder="Example: I forgot my D2L password and cannot log in to submit my assignment."
    )

    submitted = st.button("Submit Ticket")

    if submitted:
        validation_errors = []

        if not student_name.strip():
            validation_errors.append("Full Name is required.")
        if not school_id.strip():
            validation_errors.append("School ID is required.")
        if not email.strip():
            validation_errors.append("Email is required.")
        elif not is_valid_email(email):
            validation_errors.append("Please enter a valid email address.")
        if not ticket_text.strip():
            validation_errors.append("Ticket description is required.")
        elif len(ticket_text.strip()) < 15:
            validation_errors.append("Please provide a more detailed issue description (at least 15 characters).")

        if validation_errors:
            for error in validation_errors:
                st.error(error)
        else:
            prediction = predict_ticket_details(
                ticket_text,
                vectorizer,
                category_model,
                urgency_model
            )

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ticket_id = generate_ticket_id()

            ticket_data = {
                "ticket_id": ticket_id,
                "submitted_at": timestamp,
                "updated_at": timestamp,
                "student_name": student_name.strip(),
                "school_id": school_id.strip(),
                "email": email.strip(),
                "issue_area": issue_area,
                "ticket_text": ticket_text.strip(),
                "predicted_category": prediction["predicted_category"],
                "predicted_urgency": prediction["predicted_urgency"],
                "category_confidence": prediction["category_confidence"],
                "urgency_confidence": prediction["urgency_confidence"],
                "status": "Open",
                "assigned_to": "Unassigned",
                "internal_notes": "",
                "notification_sent": "Yes"
            }

            insert_ticket(ticket_data)

            st.success(f"Your ticket has been submitted successfully. Ticket ID: {ticket_id}")

            result_col1, result_col2 = st.columns(2)
            with result_col1:
                st.metric("Predicted Category", prediction["predicted_category"])
                st.metric("Category Confidence", f"{prediction['category_confidence']}%")

            with result_col2:
                st.metric("Predicted Urgency", prediction["predicted_urgency"])
                st.metric("Urgency Confidence", f"{prediction['urgency_confidence']}%")

            if prediction["predicted_urgency"] == "High":
                st.error("This issue appears to require immediate attention.")
            elif prediction["predicted_urgency"] == "Medium":
                st.warning("This issue should be reviewed soon.")
            else:
                st.success("This appears to be a lower priority issue.")

            st.info("Notification sent to tech support team. A staff member can now review your ticket and contact you using the submitted information.")

# =========================================================
# TECH SUPPORT PORTAL
# =========================================================
else:
    st.markdown("### Tech Support Dashboard")

    if not is_admin:
        st.warning("Enter the correct admin password in the sidebar to access the dashboard.")
    else:
        tickets_df = get_all_tickets()

        if tickets_df.empty:
            st.info("No tickets have been submitted yet.")
        else:
            # -----------------------------
            # METRICS
            # -----------------------------
            total_tickets = len(tickets_df)
            open_count = (tickets_df["status"] == "Open").sum()
            in_progress_count = (tickets_df["status"] == "In Progress").sum()
            resolved_count = (tickets_df["status"] == "Resolved").sum()
            high_count = (tickets_df["predicted_urgency"] == "High").sum()

            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            metric_col1.metric("Total Tickets", total_tickets)
            metric_col2.metric("Open", int(open_count))
            metric_col3.metric("In Progress", int(in_progress_count))
            metric_col4.metric("High Priority", int(high_count))

            # -----------------------------
            # FILTERS
            # -----------------------------
            st.markdown("#### Search and Filter")

            filter_col1, filter_col2, filter_col3 = st.columns(3)

            with filter_col1:
                search_text = st.text_input("Search by name, email, school ID, or ticket text")

            with filter_col2:
                urgency_filter = st.selectbox(
                    "Filter by Urgency",
                    ["All"] + sorted(tickets_df["predicted_urgency"].dropna().unique().tolist())
                )

            with filter_col3:
                status_filter = st.selectbox(
                    "Filter by Status",
                    ["All"] + sorted(tickets_df["status"].dropna().unique().tolist())
                )

            filter_col4, filter_col5 = st.columns(2)

            with filter_col4:
                category_filter = st.selectbox(
                    "Filter by Category",
                    ["All"] + sorted(tickets_df["predicted_category"].dropna().unique().tolist())
                )

            with filter_col5:
                assigned_filter = st.selectbox(
                    "Filter by Assigned Staff",
                    ["All"] + sorted(tickets_df["assigned_to"].dropna().unique().tolist())
                )

            filtered_df = tickets_df.copy()

            if search_text.strip():
                search_lower = search_text.lower()
                filtered_df = filtered_df[
                    filtered_df["student_name"].str.lower().str.contains(search_lower, na=False) |
                    filtered_df["email"].str.lower().str.contains(search_lower, na=False) |
                    filtered_df["school_id"].str.lower().str.contains(search_lower, na=False) |
                    filtered_df["ticket_text"].str.lower().str.contains(search_lower, na=False)
                ]

            if urgency_filter != "All":
                filtered_df = filtered_df[filtered_df["predicted_urgency"] == urgency_filter]

            if status_filter != "All":
                filtered_df = filtered_df[filtered_df["status"] == status_filter]

            if category_filter != "All":
                filtered_df = filtered_df[filtered_df["predicted_category"] == category_filter]

            if assigned_filter != "All":
                filtered_df = filtered_df[filtered_df["assigned_to"] == assigned_filter]

            # -----------------------------
            # ANALYTICS
            # -----------------------------
            st.markdown("#### Analytics")

            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                st.write("Tickets by Category")
                category_counts = tickets_df["predicted_category"].value_counts()
                st.bar_chart(category_counts)

            with chart_col2:
                st.write("Tickets by Urgency")
                urgency_counts = tickets_df["predicted_urgency"].value_counts()
                st.bar_chart(urgency_counts)

            chart_col3, chart_col4 = st.columns(2)

            with chart_col3:
                st.write("Tickets by Status")
                status_counts = tickets_df["status"].value_counts()
                st.bar_chart(status_counts)

            with chart_col4:
                st.write("Assigned Support Staff")
                assigned_counts = tickets_df["assigned_to"].value_counts()
                st.bar_chart(assigned_counts)

            # -----------------------------
            # HIGH PRIORITY SECTION
            # -----------------------------
            st.markdown("#### High Priority Tickets")
            high_priority_df = filtered_df[filtered_df["predicted_urgency"] == "High"]

            if high_priority_df.empty:
                st.info("No high priority tickets match the current filters.")
            else:
                for _, row in high_priority_df.head(5).iterrows():
                    show_ticket_card(row)

            # -----------------------------
            # ALL TICKETS
            # -----------------------------
            st.markdown("#### All Matching Tickets")
            st.dataframe(
                filtered_df[[
                    "ticket_id", "student_name", "school_id", "email", "issue_area",
                    "predicted_category", "predicted_urgency", "status", "assigned_to",
                    "submitted_at", "updated_at"
                ]],
                use_container_width=True
            )

            # -----------------------------
            # ADMIN ACTIONS
            # -----------------------------
            st.markdown("#### Manage a Ticket")

            if not filtered_df.empty:
                selected_ticket_id = st.selectbox(
                    "Select a Ticket ID",
                    filtered_df["ticket_id"].tolist()
                )

                selected_ticket = tickets_df[tickets_df["ticket_id"] == selected_ticket_id].iloc[0]

                manage_col1, manage_col2 = st.columns(2)

                with manage_col1:
                    updated_status = st.selectbox(
                        "Update Status",
                        ["Open", "In Progress", "Resolved"],
                        index=["Open", "In Progress", "Resolved"].index(selected_ticket["status"])
                        if selected_ticket["status"] in ["Open", "In Progress", "Resolved"] else 0
                    )

                    updated_assigned_to = st.selectbox(
                        "Assign Support Staff",
                        SUPPORT_STAFF_OPTIONS,
                        index=SUPPORT_STAFF_OPTIONS.index(selected_ticket["assigned_to"])
                        if selected_ticket["assigned_to"] in SUPPORT_STAFF_OPTIONS else 0
                    )

                with manage_col2:
                    updated_notes = st.text_area(
                        "Internal Notes",
                        value=selected_ticket["internal_notes"] if pd.notna(selected_ticket["internal_notes"]) else "",
                        height=150
                    )

                action_col1, action_col2 = st.columns(2)

                with action_col1:
                    if st.button("Save Ticket Updates"):
                        update_ticket(
                            selected_ticket_id,
                            updated_status,
                            updated_assigned_to,
                            updated_notes
                        )
                        st.success(f"{selected_ticket_id} updated successfully.")
                        st.rerun()

                with action_col2:
                    if st.button("Delete Ticket"):
                        delete_ticket(selected_ticket_id)
                        st.warning(f"{selected_ticket_id} deleted.")
                        st.rerun()

                st.markdown("#### Selected Ticket Details")
                show_ticket_card(selected_ticket)

                st.markdown("#### Contact Recommendation")
                st.info(
                    f"Reach out to {selected_ticket['student_name']} at {selected_ticket['email']} "
                    f"regarding ticket {selected_ticket['ticket_id']}."
                )