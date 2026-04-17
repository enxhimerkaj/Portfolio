import pandas as pd
df = pd.read_csv("data/tickets.csv")
print(df.head())
X = df["ticket_text"]
y_category = df["category"]
y_urgency = df["urgency"]

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
# Takes your text (X), Converts it into numbers, Creates a matrix the model can learn from
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)
#train the category model
category_model = LogisticRegression()
category_model.fit(X_vectorized, y_category)
"""
# test the model
sample_ticket = ["I can't access my account and I need it urgently"]
sample_vector = vectorizer.transform(sample_ticket)
prediction = category_model.predict(sample_vector)
print("Predicted category:", prediction[0])

#test the urgency model
urgency_model = LogisticRegression()
urgency_model.fit(X_vectorized, y_urgency)
urgency_prediction = urgency_model.predict(sample_vector)
#test the model for the same ticket
print("Predicted urgency:", urgency_prediction[0])"""

user_input = input("Enter your support ticket: ")
sample_ticket = [user_input]
sample_vector = vectorizer.transform(sample_ticket)
prediction = category_model.predict(sample_vector)
print("Predicted category:", prediction[0])
urgency_model = LogisticRegression()
urgency_model.fit(X_vectorized, y_urgency)
urgency_prediction = urgency_model.predict(sample_vector)
print("Predicted urgency:", urgency_prediction[0])

