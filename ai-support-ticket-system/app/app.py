import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("data/tickets.csv")

# Prepare data
X = df["ticket_text"]
y_category = df["category"]
y_urgency = df["urgency"]

# Vectorize
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Train models
category_model = LogisticRegression()
category_model.fit(X_vectorized, y_category)

urgency_model = LogisticRegression()
urgency_model.fit(X_vectorized, y_urgency)

# UI
st.title("AI Support Ticket System")
st.write("Submit a support issue and the system will predict the ticket category and urgency.")
st.caption("Try examples like: 'I forgot my D2L password', 'Zoom audio is not working', or 'How do I install Zoom on my laptop?'")

user_input = st.text_area("Enter your support ticket:")

if st.button("Analyze Ticket"):
    if user_input.strip() == "":
        st.error("Please enter a support ticket before clicking Analyze.")
    else:
        sample_vector = vectorizer.transform([user_input])

        category = category_model.predict(sample_vector)[0]
        urgency = urgency_model.predict(sample_vector)[0]

        st.subheader("Prediction Results")
        col1, col2 = st.columns(2)
        with col1:
         st.metric(label="Category", value=category)
        with col2:
         st.metric(label="Urgency", value=urgency)
        st.info("This prediction is based on a small training dataset and should be treated as a support recommendation, not a final decision.")
        if urgency == "High":
         st.error("This issue requires immediate attention.")
        elif urgency == "Medium":
         st.warning("This issue should be addressed soon.")
        else:
         st.success("This is a low priority issue.")