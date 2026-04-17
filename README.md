# Portfolio
My personal AI Support Ticket System

This project is a machine learning-based web application that classifies IT support tickets by category and urgency.

## Overview

The goal of this project is to simulate a real-world support system where incoming tickets are automatically analyzed and categorized. The model predicts both the type of issue and how urgent it is.

## Features

* Classifies support tickets into categories (Account Access, LMS, Zoom, etc.)
* Predicts urgency level (Low, Medium, High)
* Simple web interface using Streamlit
* Real-time predictions based on user input

## Technologies Used

* Python
* Pandas
* Scikit-learn
* TF-IDF Vectorization
* Logistic Regression
* Streamlit

## How It Works

1. The user enters a support ticket
2. The text is converted into numerical features using TF-IDF
3. Two models predict:

   * Category
   * Urgency
4. The results are displayed in the web interface

## Running the Project

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

## Example

Input:
"I forgot my D2L password"

Output:

* Category: LMS / Brightspace
* Urgency: High

## Notes

This project uses a small dataset and is meant for demonstration purposes.

## Author

Enxhi Merkaj
