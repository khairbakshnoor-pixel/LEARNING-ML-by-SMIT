# Customer Churn Prediction & Multi-Model Streamlit Dashboard

**Live App:** _Add your Streamlit Cloud/Render/HF Space URL here_

## Overview
This project predicts whether a Telco customer will **Churn (leave)** or **Retain**.

It implements **6 machine learning models**:
1. Logistic Regression
2. Decision Tree
3. Random Forest (tuned)
4. K-Nearest Neighbors (KNN)
5. Support Vector Machine (SVM)
6. XGBoost / Gradient Boosting (tuned)

It provides an interactive Streamlit dashboard with:
- Dataset Explorer
- Plotly-based EDA Dashboard
- Model Training section
- Model Comparison leaderboard + visualizations
- Prediction System form with probability-based confidence score

## Dataset
Recommended dataset: **Telco Customer Churn** (`Telco-Customer-Churn.csv`).

Place the CSV at:
- `project/data/Telco-Customer-Churn.csv`

## Project Structure
```
project/
├── data/
├── models/
├── visuals/
├── notebooks/
├── app.py
├── utils.py
├── requirements.txt
└── README.md
```

## Install & Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Train Models
Training occurs inside the Streamlit app via the **Model Training** tab.
Trained pipelines are saved into `project/models/` as `.pkl` files.

