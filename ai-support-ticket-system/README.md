My personal AI Support Ticket System

A full-stack support ticket application built with Python and Streamlit that uses machine learning to classify user issues and helps support staff manage tickets through a dashboard.

---

## Project Overview

This project simulates a real-world IT help desk system.

Users (students) can submit support requests with their contact information and issue description. The system automatically analyzes the request using machine learning to predict both the **category** and **urgency** of the issue.

All tickets are stored in a database and can be reviewed, managed, and updated through a dedicated admin dashboard.

---

## Key Features

### Student Portal

* Submit support tickets with:

  * Full name
  * School ID
  * Email
  * Issue description
* Automatic AI predictions:

  * Issue category
  * Urgency level (Low / Medium / High)
* Confidence scores for predictions
* Unique ticket ID generation
* Input validation to ensure clean and usable data

---

### Tech Support Dashboard

* Password-protected admin access
* View all submitted tickets
* Search tickets by:

  * Name
  * Email
  * School ID
  * Ticket content
* Filter tickets by:

  * Urgency
  * Status
  * Category
  * Assigned support team
* Update ticket details:

  * Status (Open, In Progress, Resolved)
  * Assigned support staff
  * Internal notes
* Delete tickets if needed
* Highlight high-priority tickets for faster response

---

### Analytics & Insights

* Visual breakdown of:

  * Tickets by category
  * Tickets by urgency
  * Ticket status distribution
  * Assigned support workload
* Built-in charts for quick decision-making

---

## Machine Learning Approach

The application uses a simple but effective NLP pipeline:

* **TF-IDF Vectorization**

  * Converts text input into numerical features

* **Logistic Regression Models**

  * One model predicts the ticket category
  * Another model predicts urgency level

* **Confidence Scores**

  * Shows how certain the model is about each prediction

The model is trained using a labeled dataset (`tickets.csv`) containing example support requests.

---

## Tech Stack

* **Python**
* **Streamlit** – UI and app framework
* **Pandas** – data manipulation
* **SQLite** – database storage
* **Scikit-learn** – machine learning (TF-IDF + Logistic Regression)

---

## Project Structure

```
ai-support-ticket-system/
│
├── app/
│   └── app.py                # main application
│
├── data/
│   ├── tickets.csv           # training dataset
│   └── support_tickets.db    # SQLite database (auto-created)
│
├── train_model.py            # optional training script
├── README.md
```

---

## How to Run the Project

1. Clone the repository:

```bash
git clone <your-repo-link>
cd ai-support-ticket-system
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the application:

```bash
streamlit run app/app.py
```

---

## Demo Access

* Student Portal → available by default
* Tech Support Portal → requires password

**Admin Password:**

```
admin123
```

---

## What This Project Demonstrates

* Building an end-to-end application using Python
* Integrating machine learning into a real workflow
* Designing user input forms and dashboards
* Working with a relational database (SQLite)
* Implementing filtering, search, and data visualization
* Structuring a project for real-world use cases

---

## Future Improvements

* Real email notifications (SMTP / SendGrid)
* Multi-user authentication (separate staff accounts)
* Improved ML model with larger dataset
* UI/UX enhancements
* Deployment (Streamlit Cloud or AWS)

---

## Summary

This project goes beyond a simple machine learning demo. It combines data processing, model training, database management, and an interactive interface to simulate a realistic support ticket system.

It demonstrates how AI can be integrated into everyday workflows to improve efficiency and decision-making.

---
## Author

Enxhi Merkaj
git