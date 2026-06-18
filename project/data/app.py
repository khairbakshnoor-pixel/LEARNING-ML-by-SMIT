import os
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from utils import (
    list_saved_models,
    load_champion_model,
    load_data,
    load_model_bundle,
    preprocess_dataframe,
    train_all_models,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Customer Churn Multi-Model Dashboard", layout="wide")
st.title("Customer Churn Prediction & Comparative Machine Learning Dashboard")


# ── Sidebar navigation ────────────────────────────────────────────────────────
def sidebar_chips():
    st.sidebar.title("Navigation")
    return st.sidebar.radio(
        "Go to",
        ["Home", "Dataset Explorer", "EDA Dashboard", "Model Training", "Model Comparison", "Prediction System"],
    )


PAGE = sidebar_chips()


# ── Cached data helpers ───────────────────────────────────────────────────────
@st.cache_data
def get_df() -> pd.DataFrame:
    """Load CSV from the same directory as app.py."""
    for name in ("Telco-Customer-Churn.csv", "Telco-Customer-Churn copy.csv"):
        candidate = os.path.join(BASE_DIR, name)
        if os.path.exists(candidate):
            return load_data(candidate)
    st.error(
        "Dataset not found. Please place 'Telco-Customer-Churn.csv' "
        "in the same folder as app.py."
    )
    st.stop()


@st.cache_data
def get_profiles(df: pd.DataFrame):
    return df.isnull().sum().sort_values(ascending=False), df.dtypes.astype(str)


# ── Page renderers ────────────────────────────────────────────────────────────

def render_home():
    st.subheader("Project Overview")
    st.markdown(
        """
        This dashboard predicts whether a Telco customer will **Churn** or **Retain**.

        It implements **6 ML algorithms** — Logistic Regression, Decision Tree, Random Forest,
        KNN, SVM, and XGBoost / Gradient Boosting — and evaluates them live using key
        classification metrics (Accuracy, Precision, Recall, F1, ROC-AUC).

        **Navigate using the sidebar** to explore the dataset, run EDA, train models,
        compare their performance, or make a single-customer prediction.
        """
    )
    st.info("Start with **Model Training** to train all models before using Model Comparison or Prediction System.")


# ── Dataset Explorer ──────────────────────────────────────────────────────────

def render_dataset_explorer():
    df = get_df()
    nulls, dtypes = get_profiles(df)

    st.subheader("Dataset Explorer")
    st.markdown(f"**Shape:** {df.shape[0]:,} rows × {df.shape[1]} columns")

    search_term = st.text_input("🔍 Search rows (matches any text column)", value="")
    if search_term:
        mask = pd.Series(False, index=df.index)
        for col in df.select_dtypes(include="object").columns:
            mask |= df[col].astype(str).str.contains(search_term, case=False, na=False)
        show_df = df.loc[mask]
        st.caption(f"{len(show_df):,} rows match '{search_term}'")
    else:
        show_df = df

    st.dataframe(show_df, use_container_width=True, height=280)

    with st.expander("Statistical Summary"):
        st.dataframe(df.describe(include="all"), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Null Counts**")
        st.dataframe(nulls[nulls > 0].rename("Nulls"), use_container_width=True)
    with col2:
        st.markdown("**Data Types**")
        st.dataframe(dtypes.rename("dtype"), use_container_width=True)


# ── EDA Dashboard ─────────────────────────────────────────────────────────────

def render_eda_dashboard():
    df = get_df()
    st.subheader("EDA Dashboard")

    plot_options = [
        "Churn Distribution",
        "MonthlyCharges distribution by Churn",
        "Tenure distribution by Churn",
        "Churn vs Contract Type",
        "Churn vs Payment Method",
        "Churn vs Internet Service",
        "Churn vs Senior Citizen",
        "Correlation Heatmap (numeric features)",
    ]
    option = st.selectbox("Choose a plot", plot_options)

    if option == "Churn Distribution":
        counts = df["Churn"].value_counts().reset_index()
        counts.columns = ["Churn", "Count"]
        fig = px.pie(counts, names="Churn", values="Count", title="Churn vs Retain")

    elif option == "MonthlyCharges distribution by Churn":
        fig = px.histogram(
            df, x="MonthlyCharges", color="Churn", barmode="overlay",
            title="Monthly Charges Distribution by Churn",
            opacity=0.75,
        )

    elif option == "Tenure distribution by Churn":
        fig = px.histogram(
            df, x="tenure", color="Churn", barmode="overlay",
            title="Tenure Distribution by Churn",
            opacity=0.75,
        )

    elif option == "Churn vs Contract Type":
        fig = px.histogram(
            df, x="Contract", color="Churn", barmode="group",
            title="Churn by Contract Type",
        )

    elif option == "Churn vs Payment Method":
        fig = px.histogram(
            df, x="PaymentMethod", color="Churn", barmode="group",
            title="Churn by Payment Method",
        )

    elif option == "Churn vs Internet Service":
        fig = px.histogram(
            df, x="InternetService", color="Churn", barmode="group",
            title="Churn by Internet Service Type",
        )

    elif option == "Churn vs Senior Citizen":
        fig = px.histogram(
            df, x="SeniorCitizen", color="Churn", barmode="group",
            title="Churn by Senior Citizen Status",
        )

    else:  # Correlation Heatmap
        numeric_df = df.select_dtypes(include="number")
        corr = numeric_df.corr()
        fig = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            title="Correlation Heatmap (Numeric Features)",
            aspect="auto",
        )

    st.plotly_chart(fig, use_container_width=True)


# ── Model Training ────────────────────────────────────────────────────────────

def render_model_training():
    st.subheader("Model Training")

    saved = list_saved_models()
    if not saved:
        st.warning("No saved models found. Click **Train All Models** to begin.")

    col1, col2 = st.columns(2)

    if col1.button("🚀 Train All Models", type="primary"):
        with st.spinner("Training 6 models — this may take a few minutes…"):
            try:
                train_all_models()
                st.success("✅ Training completed successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Training failed: {e}")

    if col2.button("🔄 Refresh"):
        st.rerun()

    saved = list_saved_models()
    if saved:
        st.markdown(f"**{len(saved)} saved model(s):** {', '.join(saved)}")
        model_name = st.selectbox("Inspect a model", saved)
        try:
            bundle = load_model_bundle(model_name)
            display_metrics = {k: v for k, v in bundle["metrics"].items() if k != "y_scores"}
            st.json(display_metrics)
            if bundle.get("best_params"):
                st.markdown("**Best Hyperparameters**")
                st.json(bundle["best_params"])
        except Exception as e:
            st.error(f"Could not load model: {e}")


# ── Model Comparison ──────────────────────────────────────────────────────────

def render_model_comparison():
    st.subheader("Model Comparison Dashboard")

    saved = list_saved_models()
    if not saved:
        st.warning("Train models first in **Model Training**.")
        return

    rows = []
    roc_fig = go.Figure()

    for name in saved:
        try:
            bundle = load_model_bundle(name)
            m = bundle["metrics"]
            rows.append({
                "Model":     name,
                "Accuracy":  round(m.get("accuracy") or 0, 4),
                "Precision": round(m.get("precision") or 0, 4),
                "Recall":    round(m.get("recall") or 0, 4),
                "F1":        round(m.get("f1") or 0, 4),
                "ROC-AUC":   round(m.get("roc_auc") or 0, 4),
            })
        except Exception:
            continue

    dfm = pd.DataFrame(rows).sort_values("F1", ascending=False)
    st.markdown("### Metrics Leaderboard")
    st.dataframe(dfm, use_container_width=True)

    # ── Bar chart: Accuracy vs F1 ─────────────────────────────────────────
    st.markdown("### Accuracy vs F1 Score")
    bar_df = dfm.melt(id_vars="Model", value_vars=["Accuracy", "F1"], var_name="Metric", value_name="Score")
    fig_bar = px.bar(bar_df, x="Model", y="Score", color="Metric", barmode="group", range_y=[0, 1])
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── ROC Curves ────────────────────────────────────────────────────────
    st.markdown("### ROC Curves")
    from sklearn.metrics import roc_curve
    df_raw = get_df()
    X_full, y_full = preprocess_dataframe(df_raw)

    roc_fig = go.Figure()
    roc_fig.add_shape(type="line", line=dict(dash="dash", color="grey"), x0=0, x1=1, y0=0, y1=1)

    for name in saved:
        try:
            bundle = load_model_bundle(name)
            model  = bundle["model"]
            m      = bundle["metrics"]
            y_scores = m.get("y_scores")
            if y_scores is None:
                continue
            # We stored scores only for the test split — use them with y_full tail
            # Instead, recompute on full data for the ROC plot
            if hasattr(model, "predict_proba"):
                scores = model.predict_proba(X_full)[:, 1]
            elif hasattr(model, "decision_function"):
                s = model.decision_function(X_full)
                scores = (s - s.min()) / (s.max() - s.min() + 1e-12)
            else:
                continue
            fpr, tpr, _ = roc_curve(y_full, scores)
            auc = m.get("roc_auc") or 0
            roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={auc:.3f})"))
        except Exception:
            continue

    roc_fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", legend_title="Model")
    st.plotly_chart(roc_fig, use_container_width=True)

    # ── Champion callout ─────────────────────────────────────────────────
    champion = dfm.iloc[0]["Model"]
    st.success(f"🏆 Champion Model: **{champion}** (highest F1 = {dfm.iloc[0]['F1']})")


# ── Prediction System ─────────────────────────────────────────────────────────

def render_prediction_system():
    st.subheader("Single Customer Churn Prediction")

    champ_name, champ_model = load_champion_model()
    if champ_model is None:
        st.warning("Train models first in **Model Training**.")
        return

    st.info(f"🏆 Active Model: **{champ_name}**")

    with st.form("prediction_form"):
        st.markdown("#### Customer Details")
        col1, col2, col3 = st.columns(3)

        with col1:
            gender          = st.selectbox("Gender", ["Female", "Male"])
            senior          = st.selectbox("Senior Citizen", [0, 1])
            partner         = st.selectbox("Partner", ["Yes", "No"])
            dependents      = st.selectbox("Dependents", ["No", "Yes"])
            tenure          = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)

        with col2:
            phone_service   = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines  = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            online_backup   = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])

        with col3:
            device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
            tech_support      = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            streaming_tv      = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            streaming_movies  = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

        st.markdown("#### Contract & Billing")
        col4, col5, col6 = st.columns(3)
        with col4:
            contract        = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        with col5:
            paperless       = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment_method  = st.selectbox(
                "Payment Method",
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            )
        with col6:
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=500.0, value=50.0, step=0.5)
            total_charges   = st.number_input("Total Charges ($)", min_value=0.0, value=float(tenure * 50), step=1.0)

        submitted = st.form_submit_button("🔮 Predict Churn", type="primary")

    if submitted:
        input_row = pd.DataFrame([{
            "gender":            gender,
            "SeniorCitizen":     senior,
            "Partner":           partner,
            "Dependents":        dependents,
            "tenure":            tenure,
            "PhoneService":      phone_service,
            "MultipleLines":     multiple_lines,
            "InternetService":   internet_service,
            "OnlineSecurity":    online_security,
            "OnlineBackup":      online_backup,
            "DeviceProtection":  device_protection,
            "TechSupport":       tech_support,
            "StreamingTV":       streaming_tv,
            "StreamingMovies":   streaming_movies,
            "Contract":          contract,
            "PaperlessBilling":  paperless,
            "PaymentMethod":     payment_method,
            "MonthlyCharges":    monthly_charges,
            "TotalCharges":      total_charges,
        }])

        try:
            df_raw = get_df()
            if "Churn" in df_raw.columns:
                df_raw = df_raw.drop(columns=["Churn"])

            # Append input row to align column structure, then take last row
            combined = pd.concat([df_raw, input_row], ignore_index=True)
            X_processed, _ = preprocess_dataframe(combined)
            last_row = X_processed.iloc[[-1]]

            pred = champ_model.predict(last_row)[0]

            # Confidence score
            confidence = None
            if hasattr(champ_model, "predict_proba"):
                proba = champ_model.predict_proba(last_row)[0]
                confidence = proba[1] if pred == 1 else proba[0]

            if pred == 1:
                msg = "⚠️ Prediction: Customer will **CHURN**"
                if confidence is not None:
                    msg += f"\n\n**Churn Probability: {confidence:.1%}**"
                st.error(msg)
            else:
                msg = "✅ Prediction: Customer will **RETAIN**"
                if confidence is not None:
                    msg += f"\n\n**Retention Confidence: {confidence:.1%}**"
                st.success(msg)

        except Exception as e:
            st.error(f"Prediction error: {e}")


# ── Router ────────────────────────────────────────────────────────────────────
if   PAGE == "Home":              render_home()
elif PAGE == "Dataset Explorer":  render_dataset_explorer()
elif PAGE == "EDA Dashboard":     render_eda_dashboard()
elif PAGE == "Model Training":    render_model_training()
elif PAGE == "Model Comparison":  render_model_comparison()
elif PAGE == "Prediction System": render_prediction_system()