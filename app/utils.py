import io
import base64
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from typing import Dict, Any, Tuple
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, f1_score, roc_auc_score, roc_curve, ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split
from app.models import REGRESSION_MODELS, CLASSIFICATION_MODELS


def load_dataset(file_bytes: bytes | None, url: str | None): # pd.DataFrame
    if file_bytes:
        return pd.read_csv(io.BytesIO(file_bytes))
    if url:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return pd.read_csv(io.StringIO(resp.text))
    raise ValueError("Debes proveer un archivo o una URL")


def build_etl_pipeline(df: pd.DataFrame, target: str): #-> Tuple[Pipeline, pd.DataFrame, pd.Series]
    if target not in df.columns:
        raise ValueError(f"La columna objetivo '{target}' no existe en el dataset")

    X = df.drop(columns=[target])
    y = df[target]

    # Detectar tipos de columnas
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    # Transformadores
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ],
        remainder="drop"
    )

    return preprocessor, X, y


def evaluate_regression(models: Dict[str, Any], pipeline: ColumnTransformer, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results = {}
    preds_store = {}

    for name, model in models.items():
        full_pipe = Pipeline(steps=[("pre", pipeline), ("model", model)])
        full_pipe.fit(X_train, y_train)
        preds = full_pipe.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        results[name] = {"RMSE": rmse, "MAE": mae, "R2": r2}
        preds_store[name] = {"y_test": y_test, "preds": preds, "pipe": full_pipe}

    # Ganador por RMSE
    winner = min(results.items(), key=lambda x: x[1]["RMSE"])[0]
    return results, winner, preds_store[winner]


def evaluate_classification(models: Dict[str, Any], pipeline: ColumnTransformer, X, y):
    # Si y es numerica continua, esto no es clasificación
    if pd.api.types.is_numeric_dtype(y) and y.nunique() > 20:
        raise ValueError("La columna objetivo parece continua. Usa 'regression' o discretízala.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    results = {}
    preds_store = {}

    # Detectar si es binaria para calcular ROC-AUC
    classes = sorted(y.unique().tolist())
    is_binary = len(classes) == 2

    for name, model in models.items():
        full_pipe = Pipeline(steps=[("pre", pipeline), ("model", model)])
        full_pipe.fit(X_train, y_train)
        preds = full_pipe.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro")

        metrics = {"Accuracy": acc, "F1_macro": f1}
        if is_binary:
            # proba para AUC
            try:
                proba = full_pipe.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, proba)
                metrics["ROC_AUC"] = auc
            except Exception:
                metrics["ROC_AUC"] = None

        results[name] = metrics
        preds_store[name] = {"y_test": y_test, "preds": preds, "pipe": full_pipe, "is_binary": is_binary}

    # Ganador por F1_macro (mayor mejor)
    winner = max(results.items(), key=lambda x: x[1]["F1_macro"])[0]
    return results, winner, preds_store[winner]


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def plot_regression_all_metrics(results: Dict[str, Dict[str, float]]):
    models = list(results.keys())
    rmse = [results[m]["RMSE"] for m in models]
    mae = [results[m]["MAE"] for m in models]
    r2 = [results[m]["R2"] for m in models]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    sns.barplot(x=models, y=rmse, ax=axes[0], palette="Blues")
    axes[0].set_title("RMSE menor es mejor")
    axes[0].tick_params(axis='x', rotation=20)

    sns.barplot(x=models, y=mae, ax=axes[1], palette="Greens")
    axes[1].set_title("MAE menor es mejor")
    axes[1].tick_params(axis='x', rotation=20)

    sns.barplot(x=models, y=r2, ax=axes[2], palette="Oranges")
    axes[2].set_title("R2 mayor es mejor")
    axes[2].tick_params(axis='x', rotation=20)

    fig.tight_layout()
    return fig_to_base64(fig)


def plot_regression_winner_scatter(y_test, preds, title="Predicción vs Real"):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(y_test, preds, alpha=0.6)
    ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], "r--", label="Ideal")
    ax.set_xlabel("Valor real")
    ax.set_ylabel("Predicción")
    ax.set_title(title)
    ax.legend()
    return fig_to_base64(fig)


def plot_regression_residuals(y_test, preds):
    residuals = y_test - preds
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.histplot(residuals, kde=True, ax=ax, color="purple")
    ax.set_title("Distribución de residuos")
    ax.set_xlabel("Error (y - y_pred)")
    return fig_to_base64(fig)


def plot_classification_metrics(results: Dict[str, Dict[str, float]]):
    models = list(results.keys())
    f1 = [results[m]["F1_macro"] for m in models]
    acc = [results[m]["Accuracy"] for m in models]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    sns.barplot(x=models, y=f1, ax=axes[0], palette="Blues")
    axes[0].set_title("F1 macro (mayor es mejor)")
    axes[0].tick_params(axis='x', rotation=20)

    sns.barplot(x=models, y=acc, ax=axes[1], palette="Greens")
    axes[1].set_title("Accuracy (mayor es mejor)")
    axes[1].tick_params(axis='x', rotation=20)

    fig.tight_layout()
    return fig_to_base64(fig)


def plot_confusion_matrix(y_test, preds):
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, preds, ax=ax, cmap="Blues")
    ax.set_title("Matriz de confusión")
    return fig_to_base64(fig)


def plot_roc_curve_binary(pipe, X_test, y_test):
    try:
        proba = pipe.predict_proba(X_test)[:, 1]
    except Exception:
        return None
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"ROC AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "r--")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("Curva ROC (binaria)")
    ax.legend()
    return fig_to_base64(fig)
