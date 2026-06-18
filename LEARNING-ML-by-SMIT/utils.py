import os
import json
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    average_precision_score,
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

# ── Base Directory Setup ──────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


@dataclass
class ModelSpec:
    name: str
    pipeline: Pipeline
    best_params: Dict[str, Any]


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_data(path: str = None) -> pd.DataFrame:
    """Load the Telco churn CSV.

    If *path* is supplied (and exists), use it directly.
    Otherwise fall back to the directory that contains this file.
    """
    if path is not None and os.path.exists(path):
        csv_path = path
    else:
        # Fallback: look next to this script
        for name in ("Telco-Customer-Churn.csv", "Telco-Customer-Churn copy.csv"):
            candidate = os.path.join(BASE_DIR, name)
            if os.path.exists(candidate):
                csv_path = candidate
                break
        else:
            raise FileNotFoundError(
                "Could not locate 'Telco-Customer-Churn.csv'. "
                "Please place it in the same folder as app.py / utils.py."
            )

    df = pd.read_csv(csv_path)
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
    return df


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Return (X, y).

    Works whether or not the 'Churn' target column is present.
    When 'Churn' is absent (e.g. live prediction input), y is an empty Series.
    """
    df = df.copy()

    # Fix TotalCharges stored as string
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    if "Churn" in df.columns:
        y = (df["Churn"] == "Yes").astype(int)
        X = df.drop(columns=["Churn"])
    else:
        y = pd.Series(dtype=int)
        X = df

    return X, y


# ── Preprocessor Builder ──────────────────────────────────────────────────────

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", drop=None)

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_classifier(model, X_test, y_test) -> Dict[str, Any]:
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy":         float(accuracy_score(y_test, y_pred)),
        "precision":        float(precision_score(y_test, y_pred, zero_division=0)),
        "recall":           float(recall_score(y_test, y_pred, zero_division=0)),
        "f1":               float(f1_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
        proba = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)

    if proba is not None:
        metrics["roc_auc"]       = float(roc_auc_score(y_test, proba))
        metrics["avg_precision"] = float(average_precision_score(y_test, proba))
        metrics["y_scores"]      = proba.tolist()
    else:
        metrics["roc_auc"]       = None
        metrics["avg_precision"] = None
        metrics["y_scores"]      = None

    return metrics


# ── Save / Load Helpers ───────────────────────────────────────────────────────

def fit_and_save(
    name: str,
    pipeline: Pipeline,
    X_train, y_train,
    X_test, y_test,
    best_params: Dict[str, Any],
) -> Dict[str, Any]:
    pipeline.fit(X_train, y_train)
    metrics = evaluate_classifier(pipeline, X_test, y_test)
    bundle = {"model": pipeline, "best_params": best_params, "metrics": metrics}
    joblib.dump(bundle, os.path.join(MODELS_DIR, f"{name}.pkl"))
    return metrics


def list_saved_models() -> List[str]:
    if not os.path.exists(MODELS_DIR):
        return []
    return [f.replace(".pkl", "") for f in os.listdir(MODELS_DIR) if f.endswith(".pkl")]


def load_model_bundle(name: str) -> Dict[str, Any]:
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model bundle '{name}' not found at {path}.")
    return joblib.load(path)


def load_champion_model() -> Tuple[str, Any]:
    """Return (name, model) for the saved model with the highest F1 score."""
    models = list_saved_models()
    if not models:
        return "None", None

    best_f1, best_name, best_model = -1.0, "None", None
    for name in models:
        try:
            bundle = load_model_bundle(name)
            f1 = bundle["metrics"].get("f1", 0) or 0
            if f1 > best_f1:
                best_f1 = f1
                best_name = name
                best_model = bundle["model"]
        except Exception:
            continue

    return best_name, best_model


# ── Training ──────────────────────────────────────────────────────────────────

def train_all_models(random_state: int = 42) -> Dict[str, Any]:
    """Train all 6 models, apply GridSearchCV where appropriate, and save bundles."""
    df = load_data()
    X, y = preprocess_dataframe(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    preprocessor = build_preprocessor(X_train)

    # ── Model registry ────────────────────────────────────────────────────────
    specs: List[Tuple[str, Any]] = [
        ("logistic_regression", LogisticRegression(max_iter=2000, random_state=random_state)),
        ("decision_tree",       DecisionTreeClassifier(random_state=random_state)),
        ("random_forest",       RandomForestClassifier(random_state=random_state)),
        ("knn",                 KNeighborsClassifier()),
        ("svm",                 SVC(probability=True, random_state=random_state)),
    ]

    if XGBClassifier is not None:
        specs.append((
            "xgboost",
            XGBClassifier(random_state=random_state, eval_metric="logloss", use_label_encoder=False),
        ))
    else:
        specs.append((
            "gradient_boosting",
            GradientBoostingClassifier(random_state=random_state),
        ))

    # ── Hyper-parameter grids ─────────────────────────────────────────────────
    tuned_grids = {
        "random_forest": {
            "model__n_estimators": [100, 200],
            "model__max_depth":    [None, 5, 10],
        },
        "xgboost": {
            "model__n_estimators": [100, 200],
            "model__max_depth":    [3, 5],
        },
        "gradient_boosting": {
            "model__n_estimators": [100, 200],
            "model__max_depth":    [3, 5],
        },
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
    results: Dict[str, Any] = {}

    for name, estimator in specs:
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])

        if name in tuned_grids:
            search = GridSearchCV(
                pipeline,
                param_grid=tuned_grids[name],
                cv=cv,
                scoring="f1",
                n_jobs=-1,
            )
            search.fit(X_train, y_train)
            best_pipeline = search.best_estimator_
            best_params   = search.best_params_
        else:
            pipeline.fit(X_train, y_train)
            best_pipeline = pipeline
            best_params   = {}

        metrics = evaluate_classifier(best_pipeline, X_test, y_test)
        joblib.dump(
            {"model": best_pipeline, "best_params": best_params, "metrics": metrics},
            os.path.join(MODELS_DIR, f"{name}.pkl"),
        )
        results[name] = {"metrics": metrics, "best_params": best_params}

    return results