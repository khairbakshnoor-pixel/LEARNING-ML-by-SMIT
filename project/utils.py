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
    RocCurveDisplay,
    PrecisionRecallDisplay,
    average_precision_score,
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None


DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "Telco-Customer-Churn.csv")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")


os.makedirs(MODELS_DIR, exist_ok=True)


@dataclass
class ModelSpec:
    name: str
    pipeline: Pipeline
    best_params: Dict[str, Any]


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Drop customerID per common telco preprocessing
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
    return df


def preprocess_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    df = df.copy()

    # Fix TotalCharges type
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # Target
    if "Churn" not in df.columns:
        raise ValueError("Expected target column 'Churn' not found.")

    y = (df["Churn"] == "Yes").astype(int)
    X = df.drop(columns=["Churn"])

    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

    categorical_transformer = OneHotEncoder(handle_unknown="ignore", drop=None)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )

    return preprocessor


def evaluate_classifier(model, X_test, y_test) -> Dict[str, Any]:
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    # ROC-AUC requires probabilities
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        # map decision function to pseudo-prob
        scores = model.decision_function(X_test)
        proba = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)

    if proba is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_test, proba))
        metrics["avg_precision"] = float(average_precision_score(y_test, proba))
        metrics["y_scores"] = proba.tolist()
    else:
        metrics["roc_auc"] = None
        metrics["avg_precision"] = None
        metrics["y_scores"] = None

    return metrics


def fit_and_save(
    name: str,
    pipeline: Pipeline,
    X_train,
    y_train,
    X_test,
    y_test,
    best_params: Dict[str, Any],
) -> Dict[str, Any]:
    pipeline.fit(X_train, y_train)

    metrics = evaluate_classifier(pipeline, X_test, y_test)
    bundle = {"model": pipeline, "best_params": best_params, "metrics": metrics}
    out_path = os.path.join(MODELS_DIR, f"{name}.pkl")
    joblib.dump(bundle, out_path)
    return metrics


def train_all_models(random_state: int = 42) -> Dict[str, Any]:
    df = load_data()
    X, y = preprocess_dataframe(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    preprocessor = build_preprocessor(X)

    specs: List[Tuple[str, Any]] = []

    # 1. Logistic Regression
    lr = LogisticRegression(max_iter=2000)
    specs.append(("logistic_regression", lr))

    # 2. Decision Tree
    dt = DecisionTreeClassifier(random_state=random_state)
    specs.append(("decision_tree", dt))

    # 3. Random Forest (tuned)
    rf = RandomForestClassifier(random_state=random_state)
    specs.append(("random_forest", rf))

    # 4. KNN (scaled via preprocessor)
    knn = KNeighborsClassifier()
    specs.append(("knn", knn))

    # 5. SVM (probabilities)
    svm = SVC(probability=True, random_state=random_state)
    specs.append(("svm", svm))

    # 6. Gradient Boosting / XGBoost (tuned)
    if XGBClassifier is not None:
        xgb = XGBClassifier(
            random_state=random_state,
            eval_metric="logloss",
            use_label_encoder=False,
        )
        specs.append(("xgboost", xgb))
    else:
        gb = GradientBoostingClassifier(random_state=random_state)
        specs.append(("gradient_boosting", gb))

    results: Dict[str, Any] = {}

    for name, estimator in specs:
        if name == "random_forest":
            base = Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])
            param_grid = {
                "model__n_estimators": [100, 200],
                "model__max_depth": [3, 5, None],
                "model__min_samples_split": [2, 5],
            }
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
            search = GridSearchCV(
                base,
                param_grid=param_grid,
                cv=cv,
                scoring="f1",
                n_jobs=-1,
            )
            search.fit(X_train, y_train)
            best_pipeline = search.best_estimator_
            best_params = search.best_params_
            metrics = evaluate_classifier(best_pipeline, X_test, y_test)
            bundle = {"model": best_pipeline, "best_params": best_params, "metrics": metrics}
            joblib.dump(bundle, os.path.join(MODELS_DIR, f"{name}.pkl"))
            results[name] = {"metrics": metrics, "best_params": best_params}

        elif name in {"xgboost", "gradient_boosting"}:
            base = Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])
            if name == "xgboost":
                param_grid = {
                    "model__n_estimators": [100, 200],
                    "model__max_depth": [3, 5, 7],
                    "model__learning_rate": [0.05, 0.1],
                    "model__subsample": [0.8, 1.0],
                    "model__colsample_bytree": [0.8, 1.0],
                }
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
                search = GridSearchCV(
                    base,
                    param_grid=param_grid,
                    cv=cv,
                    scoring="f1",
                    n_jobs=-1,
                )
                search.fit(X_train, y_train)
                best_pipeline = search.best_estimator_
                best_params = search.best_params_
            else:
                # GradientBoostingClassifier doesn't accept many hyperparameters; do a small grid
                param_grid = {
                    "model__n_estimators": [100, 200],
                    "model__learning_rate": [0.05, 0.1],
                    "model__max_depth": [2, 3],
                }
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
                search = GridSearchCV(
                    base,
                    param_grid=param_grid,
                    cv=cv,
                    scoring="f1",
                    n_jobs=-1,
                )
                search.fit(X_train, y_train)
                best_pipeline = search.best_estimator_
                best_params = search.best_params_

            metrics = evaluate_classifier(best_pipeline, X_test, y_test)
            bundle = {"model": best_pipeline, "best_params": best_params, "metrics": metrics}
            joblib.dump(bundle, os.path.join(MODELS_DIR, f"{name}.pkl"))
            results[name] = {"metrics": metrics, "best_params": best_params}

        else:
            pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])
            metrics = fit_and_save(
                name=name,
                pipeline=pipeline,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                best_params={},
            )
            results[name] = {"metrics": metrics, "best_params": {}}

    # Choose champion by best F1, then ROC-AUC
    def score_item(item):
        m = item["metrics"]
        return (
            m.get("f1", -1),
            m.get("roc_auc", -1) if m.get("roc_auc") is not None else -1,
        )

    champion_name = max(results.keys(), key=lambda k: score_item(results[k]))
    results["champion"] = champion_name
    return results


def list_saved_models() -> List[str]:
    if not os.path.isdir(MODELS_DIR):
        return []
    out = []
    for fn in os.listdir(MODELS_DIR):
        if fn.endswith(".pkl"):
            out.append(fn.replace(".pkl", ""))
    return sorted(out)


def load_model_bundle(model_name: str) -> Dict[str, Any]:
    path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)


def load_champion_model() -> Tuple[str, Any]:
    saved = list_saved_models()
    if not saved:
        return None, None

    # If champion file exists, prefer it
    champ_path = os.path.join(MODELS_DIR, "champion.json")
    if os.path.exists(champ_path):
        with open(champ_path, "r", encoding="utf-8") as f:
            champ_name = json.load(f).get("champion")
        if champ_name:
            return champ_name, load_model_bundle(champ_name)["model"]

    # else: choose by stored metrics
    best_name = None
    best = (-1, -1)
    for name in saved:
        bundle = load_model_bundle(name)
        m = bundle.get("metrics", {})
        key = (m.get("f1", -1), m.get("roc_auc") if m.get("roc_auc") is not None else -1)
        if key > best:
            best = key
            best_name = name

    if best_name is None:
        return None, None

    # persist champion
    with open(champ_path, "w", encoding="utf-8") as f:
        json.dump({"champion": best_name}, f)

    return best_name, load_model_bundle(best_name)["model"]


def predict_single(model_bundle, input_dict: Dict[str, Any]) -> Dict[str, Any]:
    model = model_bundle["model"] if isinstance(model_bundle, dict) else model_bundle
    X = pd.DataFrame([input_dict])
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1][0]
    else:
        scores = model.decision_function(X)
        scores = np.array(scores)
        proba = float((scores - scores.min()) / (scores.max() - scores.min() + 1e-12))

    pred = int(proba >= 0.5)
    label = "Churn Risk" if pred == 1 else "Retained Account"
    return {
        "prediction": label,
        "probability": float(proba),
        "confidence_percent": float(proba * 100.0),
    }

