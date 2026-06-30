import streamlit as st
import pandas as pd
from datetime import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="OpsPilot AI",
    page_icon="🛠️",
    layout="wide"
)


# --------------------------------------------------
# Load dataset
# --------------------------------------------------
@st.cache_data
def load_data():
    data = pd.read_csv("data/tickets.csv")

    required_columns = [
        "ticket_id",
        "description",
        "category",
        "priority",
        "team",
        "suggested_action"
    ]

    missing_columns = [col for col in required_columns if col not in data.columns]

    if missing_columns:
        st.error(f"Missing required columns in tickets.csv: {missing_columns}")
        st.stop()

    return data


tickets = load_data()


# --------------------------------------------------
# Train machine learning models
# --------------------------------------------------
@st.cache_resource
def train_models(data):
    category_model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2))),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    priority_model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2))),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    category_model.fit(data["description"], data["category"])
    priority_model.fit(data["description"], data["priority"])

    similarity_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    ticket_vectors = similarity_vectorizer.fit_transform(data["description"])

    return category_model, priority_model, similarity_vectorizer, ticket_vectors


category_model, priority_model, similarity_vectorizer, ticket_vectors = train_models(tickets)


# --------------------------------------------------
# Session state for ticket history
# --------------------------------------------------
if "ticket_history" not in st.session_state:
    st.session_state.ticket_history = []


# --------------------------------------------------
# Business logic
# --------------------------------------------------
def calculate_sla_risk(priority):
    if priority == "Urgent":
        return "Critical"
    elif priority == "High":
        return "High"
    elif priority == "Medium":
        return "Medium"
    return "Low"


def get_team_for_category(category):
    category_team_map = tickets.groupby("category")["team"].agg(lambda x: x.mode()[0]).to_dict()
    return category_team_map.get(category, "IT Support")


def get_default_action(category):
    action_map = {
        "Network": "Check network connectivity, restart VPN/client, verify credentials, and review network status.",
        "Access": "Verify user identity, check account status, reset password, and review access permissions.",
        "Hardware": "Check device status, restart hardware, review updates, and reinstall drivers if needed.",
        "Cloud": "Check cloud resource status, review monitoring logs, restart service, and escalate if outage is detected.",
        "Data": "Check data source connection, refresh schedule, permissions, gateway status, and database availability.",
        "Software": "Collect error logs, restart application, check recent updates, and escalate if issue repeats.",
        "Security": "Isolate affected account or device, review security logs, reset credentials if needed, and escalate to security operations.",
        "General IT": "Review ticket details and assign to the correct support team."
    }

    return action_map.get(category, "Review ticket details and assign to the correct support team.")


def get_matched_keywords(description):
    text = description.lower()

    keyword_groups = {
        "Security": ["phishing", "malware", "suspicious", "unauthorized", "security", "clicked", "link"],
        "Network": ["vpn", "wi-fi", "wifi", "network", "internet", "firewall", "website"],
        "Access": ["password", "login", "access", "permission", "mfa", "account", "locked"],
        "Hardware": ["laptop", "printer", "device", "keyboard", "mouse", "screen", "monitor", "battery"],
        "Cloud": ["azure", "cloud", "vm", "virtual machine", "storage", "server", "function", "deployment"],
        "Data": ["database", "power bi", "dashboard", "excel", "data", "sql", "pipeline", "gateway"],
        "Software": ["application", "app", "teams", "software", "crash", "error", "outlook", "crm", "erp"]
    }

    matched = []

    for category, keywords in keyword_groups.items():
        for keyword in keywords:
            if keyword in text:
                matched.append(keyword)

    return matched


def predict_ticket(description):
    predicted_category = category_model.predict([description])[0]
    predicted_priority = priority_model.predict([description])[0]

    category_probability = max(category_model.predict_proba([description])[0])
    priority_probability = max(priority_model.predict_proba([description])[0])

    confidence_score = round(((category_probability + priority_probability) / 2) * 100, 1)

    assigned_team = get_team_for_category(predicted_category)
    sla_risk = calculate_sla_risk(predicted_priority)
    suggested_action = get_default_action(predicted_category)
    matched_keywords = get_matched_keywords(description)

    return {
        "category": predicted_category,
        "priority": predicted_priority,
        "team": assigned_team,
        "sla_risk": sla_risk,
        "confidence": confidence_score,
        "suggested_action": suggested_action,
        "matched_keywords": matched_keywords
    }


def find_similar_tickets(description, top_n=3):
    user_vector = similarity_vectorizer.transform([description])
    similarity_scores = cosine_similarity(user_vector, ticket_vectors).flatten()

    top_indices = similarity_scores.argsort()[::-1][:top_n]

    similar = tickets.iloc[top_indices].copy()
    similar["similarity_score"] = [round(similarity_scores[index] * 100, 1) for index in top_indices]

    return similar


def create_explanation(result):
    keywords = result["matched_keywords"]

    if keywords:
        keyword_text = ", ".join(keywords)
    else:
        keyword_text = "no strong keyword match"

    explanation = (
        f"The system predicted this ticket as **{result['category']}** because the text contains patterns "
        f"similar to previous {result['category']} tickets in the dataset. "
        f"Matched keyword signals: **{keyword_text}**."
    )

    priority_explanation = (
        f"The predicted priority is **{result['priority']}**, which creates an SLA risk level of "
        f"**{result['sla_risk']}**. This helps support teams decide how quickly the issue should be handled."
    )

    return explanation, priority_explanation


# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.title("🛠️ OpsPilot AI")
st.sidebar.write("Enterprise IT Service Desk Intelligence Platform")

st.sidebar.markdown("---")
st.sidebar.markdown("### Features")
st.sidebar.write("✅ ML ticket classification")
st.sidebar.write("✅ Priority prediction")
st.sidebar.write("✅ SLA risk detection")
st.sidebar.write("✅ Team assignment")
st.sidebar.write("✅ Explainable results")
st.sidebar.write("✅ Similar ticket search")
st.sidebar.write("✅ Dashboard filters")
st.sidebar.write("✅ CSV export")

st.sidebar.markdown("---")
st.sidebar.info("Portfolio MVP: Machine Learning + Analytics")


# --------------------------------------------------
# Header
# --------------------------------------------------
st.title("OpsPilot AI")
st.subheader("AI-Powered IT Service Desk Intelligence Platform")

st.write(
    "OpsPilot AI helps enterprise IT teams classify support tickets, predict urgency, assign the correct support team, "
    "identify SLA risk, and recommend first troubleshooting actions using machine learning and data analysis."
)


# --------------------------------------------------
# Tabs
# --------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎫 Ticket Analyzer",
    "📊 Dashboard",
    "🔎 Ticket Search",
    "📁 Dataset",
    "ℹ️ Project Info"
])


# --------------------------------------------------
# Tab 1: Ticket Analyzer
# --------------------------------------------------
with tab1:
    st.markdown("## Analyze a Support Ticket")

    ticket_description = st.text_area(
        "Enter the IT support ticket description:",
        placeholder="Example: My Azure virtual machine is not responding.",
        height=150
    )

    if st.button("Analyze Ticket", type="primary"):
        if ticket_description.strip() == "":
            st.warning("Please enter a ticket description first.")
        else:
            result = predict_ticket(ticket_description)
            explanation, priority_explanation = create_explanation(result)
            similar_tickets = find_similar_tickets(ticket_description)

            history_item = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "description": ticket_description,
                "category": result["category"],
                "priority": result["priority"],
                "team": result["team"],
                "sla_risk": result["sla_risk"],
                "confidence": result["confidence"]
            }

            st.session_state.ticket_history.append(history_item)

            st.markdown("## Ticket Analysis Result")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Category", result["category"])

            with col2:
                st.metric("Priority", result["priority"])

            with col3:
                st.metric("Assigned Team", result["team"])

            with col4:
                st.metric("SLA Risk", result["sla_risk"])

            st.markdown("### Confidence Score")
            st.progress(result["confidence"] / 100)
            st.write(f"{result['confidence']}% model confidence based on learned ticket patterns.")

            st.markdown("### Matched Keywords")
            if result["matched_keywords"]:
                st.write(", ".join(result["matched_keywords"]))
            else:
                st.write("No strong keyword match found. The prediction is mainly based on similarity to training examples.")

            st.markdown("### Why this result?")
            st.info(explanation)

            st.markdown("### Priority and SLA Explanation")
            st.warning(priority_explanation)

            st.markdown("### Suggested First Action")
            st.success(result["suggested_action"])

            st.markdown("### Similar Past Tickets")
            st.dataframe(
                similar_tickets[
                    ["ticket_id", "description", "category", "priority", "team", "similarity_score"]
                ],
                use_container_width=True
            )

    st.markdown("---")
    st.markdown("## Recent Ticket History")

    if len(st.session_state.ticket_history) == 0:
        st.info("No tickets analyzed yet.")
    else:
        history_df = pd.DataFrame(st.session_state.ticket_history)
        st.dataframe(history_df, use_container_width=True)


# --------------------------------------------------
# Tab 2: Dashboard
# --------------------------------------------------
with tab2:
    st.markdown("## Ticket Analytics Dashboard")

    dashboard_data = tickets.copy()
    dashboard_data["sla_risk"] = dashboard_data["priority"].apply(calculate_sla_risk)

    st.markdown("### Filters")

    col1, col2, col3 = st.columns(3)

    with col1:
        selected_category = st.selectbox(
            "Category",
            ["All"] + sorted(dashboard_data["category"].unique().tolist())
        )

    with col2:
        selected_priority = st.selectbox(
            "Priority",
            ["All"] + sorted(dashboard_data["priority"].unique().tolist())
        )

    with col3:
        selected_team = st.selectbox(
            "Support Team",
            ["All"] + sorted(dashboard_data["team"].unique().tolist())
        )

    filtered_data = dashboard_data.copy()

    if selected_category != "All":
        filtered_data = filtered_data[filtered_data["category"] == selected_category]

    if selected_priority != "All":
        filtered_data = filtered_data[filtered_data["priority"] == selected_priority]

    if selected_team != "All":
        filtered_data = filtered_data[filtered_data["team"] == selected_team]

    st.markdown("---")

    metric1, metric2, metric3, metric4 = st.columns(4)

    with metric1:
        st.metric("Total Tickets", len(filtered_data))

    with metric2:
        st.metric("High Priority", len(filtered_data[filtered_data["priority"] == "High"]))

    with metric3:
        st.metric("Urgent Tickets", len(filtered_data[filtered_data["priority"] == "Urgent"]))

    with metric4:
        st.metric("Critical SLA Risk", len(filtered_data[filtered_data["sla_risk"] == "Critical"]))

    st.markdown("---")

    if filtered_data.empty:
        st.warning("No tickets match the selected filters.")
    else:
        chart1, chart2 = st.columns(2)

        with chart1:
            st.markdown("### Tickets by Category")
            st.bar_chart(filtered_data["category"].value_counts())

        with chart2:
            st.markdown("### Tickets by Priority")
            st.bar_chart(filtered_data["priority"].value_counts())

        st.markdown("### Tickets by Support Team")
        st.bar_chart(filtered_data["team"].value_counts())

        st.markdown("### SLA Risk Distribution")
        st.bar_chart(filtered_data["sla_risk"].value_counts())

        st.markdown("---")
        st.markdown("## Filtered Ticket Records")
        st.dataframe(filtered_data, use_container_width=True)

        csv_data = filtered_data.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Filtered Tickets as CSV",
            data=csv_data,
            file_name="filtered_tickets.csv",
            mime="text/csv"
        )


# --------------------------------------------------
# Tab 3: Ticket Search
# --------------------------------------------------
with tab3:
    st.markdown("## Search Existing Tickets")

    search_query = st.text_input(
        "Search by keyword:",
        placeholder="Example: Azure, VPN, password, phishing, Power BI"
    )

    search_results = tickets.copy()

    if search_query.strip() != "":
        query = search_query.lower()

        search_results = search_results[
            search_results["description"].str.lower().str.contains(query)
            | search_results["category"].str.lower().str.contains(query)
            | search_results["priority"].str.lower().str.contains(query)
            | search_results["team"].str.lower().str.contains(query)
        ]

    st.write(f"Showing {len(search_results)} matching tickets.")
    st.dataframe(search_results, use_container_width=True)


# --------------------------------------------------
# Tab 4: Dataset
# --------------------------------------------------
with tab4:
    st.markdown("## Training Dataset")

    st.write(
        "This dataset is used to train the machine learning models that predict ticket category and priority."
    )

    st.dataframe(tickets, use_container_width=True)

    st.markdown("### Dataset Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Records", len(tickets))

    with col2:
        st.metric("Categories", tickets["category"].nunique())

    with col3:
        st.metric("Support Teams", tickets["team"].nunique())


# --------------------------------------------------
# Tab 5: Project Info
# --------------------------------------------------
with tab5:
    st.markdown("## About OpsPilot AI")

    st.markdown("### Business Problem")
    st.write(
        "Enterprise IT service desks receive many support tickets every day. Manual ticket triage can be slow, "
        "inconsistent, and expensive. Incorrect routing can delay resolution and increase SLA risk."
    )

    st.markdown("### Solution")
    st.write(
        "OpsPilot AI uses machine learning and analytics to classify support tickets, predict priority, assign support teams, "
        "identify SLA risk, explain predictions, and recommend first troubleshooting actions."
    )

    st.markdown("### How the Machine Learning Works")
    st.write(
        "The project converts ticket descriptions into numerical text features using TF-IDF. "
        "Then Logistic Regression models learn patterns from previous tickets to predict category and priority."
    )

    st.markdown("### Technologies Used")
    st.write("- Python")
    st.write("- Streamlit")
    st.write("- Pandas")
    st.write("- Scikit-learn")
    st.write("- TF-IDF text vectorization")
    st.write("- Logistic Regression")
    st.write("- Data visualization")
    st.write("- IT service management logic")

    st.markdown("### Why This Project Is Valuable")
    st.write(
        "This project demonstrates practical skills in software development, machine learning, data analysis, "
        "enterprise IT operations, explainable AI, and cloud-ready application design."
    )

    st.markdown("### Future Improvements")
    st.write("- Deploy app online")
    st.write("- Add Azure cloud deployment")
    st.write("- Store tickets in a database")
    st.write("- Add user authentication")
    st.write("- Add Power BI dashboard")
    st.write("- Add AI-generated troubleshooting responses")