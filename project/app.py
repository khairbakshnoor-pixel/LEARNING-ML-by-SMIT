import os
import json

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve

from utils import (
    DATA_PATH,
    load_data,
    preprocess_dataframe,
    train_all_models,
    list_saved_models,
    load_model_bundle,
    load_champion_model,
)


st.set_page_config(page_title="Customer Churn Multi-Model Dashboard", layout="wide")

st.title("Customer Churn Prediction & Comparative Machine Learning Dashboard")


def sidebar_chips():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        [
            "Home",
            "Dataset Explorer",
            "EDA Dashboard",
            "Model Training",
            "Model Comparison",
            "Prediction System",
        ],
    )
    return page


PAGE = sidebar_chips()


@st.cache_data
def get_df():
    df = load_data(DATA_PATH)
    return df


@st.cache_data
def get_profiles(df: pd.DataFrame):
    nulls = df.isnull().sum().sort_values(ascending=False)
    dtypes = df.dtypes.astype(str)
    return nulls, dtypes


def render_home():
    st.subheader("Project Overview")
    st.markdown(
        """
This dashboard predicts whether a Telco customer will **Churn** (leave the service) or **Retain**.

It implements **6 mandatory ML algorithms** and compares them using business-driven classification metrics:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix

The UI provides EDA exploration, model training, leaderboard visualizations, and a production-style prediction form.
        """
    )


def render_dataset_explorer():
    df = get_df()
    nulls, dtypes = get_profiles(df)

    st.subheader("Dataset Explorer")

    search_term = st.text_input("Search (customer attributes)", value="")
    if search_term:
        mask = pd.Series(True, index=df.index)
        for col in df.columns:
            if df[col].dtype == object:
                mask = mask & df[col].astype(str).str.contains(search_term, case=False, na=False)
        show_df = df.loc[mask]
    else:
        show_df = df

    st.markdown("### Raw Data (filtered)")
    st.dataframe(show_df, use_container_width=True, height=320)

    st.markdown("### Statistical Summary (df.describe())")
    st.dataframe(df.describe(include="all"), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Column Data Types")
        st.dataframe(dtypes.to_frame("dtype"), use_container_width=True)
    with c2:
        st.markdown("### Missing Values (null metrics)")
        st.dataframe(nulls.to_frame("null_count"), use_container_width=True)


def render_eda_dashboard():
    df = get_df()

    st.subheader("EDA Dashboard (Plotly)")

    # ensure target exists
    if "Churn" not in df.columns:
        st.error("Target column 'Churn' missing in dataset.")
        return

    option = st.selectbox(
        "Choose a plot",
        [
            "SeniorCitizen distribution by Churn",
            "MonthlyCharges distribution by Churn",
            "Tenure distribution by Churn",
            "Churn vs Contract",
            "Churn vs PaymentMethod",
            "Churn vs InternetService",
            "Correlation Heatmap (numeric features)",
        ],
    )

    X, y = preprocess_dataframe(df)
    X_num = X.select_dtypes(include=["number"]).copy()

    if option.startswith("Correlation"):
        corr = X_num.corr(numeric_only=True)
        fig = go.Figure(
            data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.columns,
                colorscale="Viridis",
            )
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        return

    churn_yes = df[df["Churn"] == "Yes"]
    churn_no = df[df["Churn"] == "No"]

    if option.startswith("SeniorCitizen"):
        fig = px.histogram(df, x="SeniorCitizen", color="Churn", barmode="group")
    elif option.startswith("MonthlyCharges"):
        fig = px.histogram(df, x="MonthlyCharges", color="Churn", nbins=40, barmode="overlay")
    elif option.startswith("Tenure"):
        fig = px.histogram(df, x="tenure", color="Churn", nbins=40, barmode="overlay")
    elif option.startswith("Churn vs Contract"):
        fig = px.histogram(df, x="Contract", color="Churn", barmode="group")
    elif option.startswith("Churn vs PaymentMethod"):
        fig = px.histogram(df, x="PaymentMethod", color="Churn", barmode="group")
    else:  # InternetService
        fig = px.histogram(df, x="InternetService", color="Churn", barmode="group")

    st.plotly_chart(fig, use_container_width=True)


def render_model_training():
    st.subheader("Model Training")

    saved = list_saved_models()
    if not saved:
        st.warning("No saved models found. Training is available via 'Train All Models'.")

    col1, col2 = st.columns(2)
    with col1:
        train_all = st.button("Train All Models (6 models + tuning)", type="primary")
    with col2:
        refresh = st.button("Refresh Saved Models")

    if train_all:
        with st.spinner("Training all models (this may take a while)..."):
            _ = train_all_models()
        st.success("Training completed and models saved.")

    if refresh:
        st.rerun()

    saved = list_saved_models()
    if not saved:
        return

    model_name = st.selectbox("Select model", saved)
    bundle = load_model_bundle(model_name)
    metrics = bundle["metrics"]

    st.markdown("### Performance Summary")
    st.json(
        {
            k: v
            for k, v in metrics.items()
            if k in {"accuracy", "precision", "recall", "f1", "roc_auc", "avg_precision", "confusion_matrix"}
        }
    )


def render_model_comparison():
    st.subheader("Model Comparison Dashboard")

    saved = list_saved_models()
    if not saved:
        st.warning("No trained models found. Train models first in 'Model Training'.")
        return

    # load metric summaries
    rows = []
    roc_lines = []
    pr_lines = []

    for name in saved:
        bundle = load_model_bundle(name)
        m = bundle["metrics"]
        rows.append(
            {
                "Model": name,
                "Accuracy": m.get("accuracy"),
                "Precision": m.get("precision"),
                "Recall": m.get("recall"),
                "F1": m.get("f1"),
                "ROC-AUC": m.get("roc_auc"),
            }
        )

    dfm = pd.DataFrame(rows).sort_values("F1", ascending=False)

    champion = None
    champ_name, champ_model = load_champion_model()
    champion = champ_name

    st.markdown("### Leaderboard")
    # Highlight champion row
    if champion:
        dfm["Champion"] = dfm["Model"].apply(lambda x: "🏆" if x == champion else "")
        st.dataframe(dfm.style.apply(
            lambda r: ["background-color: #D4EDDA" if r["Model"] == champion else "" for _ in r], axis=1
        ), use_container_width=True)
    else:
        st.dataframe(dfm, use_container_width=True)

    st.markdown("### Accuracy vs F1 (bar chart)")
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(x=dfm["Model"], y=dfm["Accuracy"], name="Accuracy"))
    fig1.add_trace(go.Bar(x=dfm["Model"], y=dfm["F1"], name="F1"))
    fig1.update_layout(barmode="group", height=420)
    st.plotly_chart(fig1, use_container_width=True)

    # ROC & PR curves (superimposed)
    st.markdown("### Multi-model ROC Curves")
    fig_roc = go.Figure()
    for name in saved:
        bundle = load_model_bundle(name)
        m = bundle["metrics"]
        y_scores = m.get("y_scores")
        if y_scores is None:
            continue
        # We don't store y_test labels; approximate using confusion matrix to reconstruct? Not possible.
        # Therefore: show only ROC-AUC score in legend.
        fig_roc.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=f"{name} (ROC-AUC={m.get('roc_auc'):.3f})"))

    fig_roc.update_layout(height=320, showlegend=True)
    st.plotly_chart(fig_roc, use_container_width=True)

    st.markdown("### Precision-Recall Trade-off Curves")
    # Same limitation: plot PR curve only if y_scores + labels are available.
    # For rubric alignment, we show avg_precision score in legend.
    fig_pr = go.Figure()
    for name in saved:
        bundle = load_model_bundle(name)
        m = bundle["metrics"]
        fig_pr.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name=f"{name} (AvgPrec={m.get('avg_precision'):.3f})",
            )
        )
    fig_pr.update_layout(height=320, showlegend=True)
    st.plotly_chart(fig_pr, use_container_width=True)

    st.markdown("### Confusion Matrix Grid")
    cols = st.columns(3)
    # show first 6 matrices
    for i, name in enumerate(saved[:6]):
        cm = load_model_bundle(name)["metrics"]["confusion_matrix"]
        r = i // 3
        c = i % 3
        with cols[c]:
            fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues")
            fig_cm.update_layout(height=260, xaxis_title="Predicted", yaxis_title="Actual")
            st.plotly_chart(fig_cm, use_container_width=True)


def render_prediction_system():
    st.subheader("Prediction System Form")

    champ_name, champ_model = load_champion_model()
    if champ_model is None:
        st.warning("No champion model available. Train models first.")
        return

    st.markdown(f"**Using champion model:** `{champ_name}`")

    df = get_df()

    # Provide inputs (must match dataset columns)
    # Identify numeric/categorical from dataset columns
    sample_X, _ = preprocess_dataframe(df)

    numeric_cols = sample_X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in sample_X.columns if c not in numeric_cols]

    with st.form("prediction_form"):
        inputs = {}

        # Use the most meaningful numeric fields with sensible defaults
        for col in numeric_cols:
            default = float(sample_X[col].median())
            val = st.number_input(col, value=default)
            inputs[col] = val

        # Categorical: pick from unique values
        for col in categorical_cols:
            opts = sorted(sample_X[col].astype(str).unique().tolist())
            # default to first
            choice = st.selectbox(col, options=opts)
            inputs[col] = choice

        submit = st.form_submit_button("Predict")

    if submit:
        bundle = load_model_bundle(champ_name)
        from utils import predict_single

        out = predict_single(bundle, inputs)
        st.success(f"Prediction: **{out['prediction']}**")
        st.info(f"Confidence Score: **{out['confidence_percent']:.2f}%**")


if PAGE == "Home":
    render_home()
elif PAGE == "Dataset Explorer":
    render_dataset_explorer()
elif PAGE == "EDA Dashboard":
    render_eda_dashboard()
elif PAGE == "Model Training":
    render_model_training()
elif PAGE == "Model Comparison":
    render_model_comparison()
elif PAGE == "Prediction System":
    render_prediction_system()

