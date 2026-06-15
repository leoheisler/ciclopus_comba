"""
evaluation.py
=============
Avaliação e comparação dos modelos.

Calcula Accuracy, Precision, Recall, F1 (macro e por classe), matriz de
confusão e o classification report. Também implementa a seleção automática
do melhor modelo e a importância de features.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

import config


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]
) -> dict:
    """
    Calcula o conjunto de métricas para um modelo.

    Retorna um dicionário serializável com métricas agregadas (macro/weighted),
    métricas por classe, matriz de confusão e classification report.
    """
    class_indices = list(range(len(labels)))
    report = classification_report(
        y_true,
        y_pred,
        labels=class_indices,
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=class_indices)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "confusion_matrix": cm.tolist(),
        "labels": labels,
        "per_class": {
            label: {
                "precision": report[label]["precision"],
                "recall": report[label]["recall"],
                "f1": report[label]["f1-score"],
                "support": report[label]["support"],
            }
            for label in labels
        },
    }


def metrics_to_frame(metrics: dict) -> pd.DataFrame:
    """Tabela comparativa (uma linha por modelo) para o dashboard."""
    rows = []
    for name, m in metrics.items():
        rows.append(
            {
                "Modelo": name,
                "Accuracy": m["accuracy"],
                "Precision (macro)": m["precision_macro"],
                "Recall (macro)": m["recall_macro"],
                "F1 (macro)": m["f1_macro"],
                "F1 (weighted)": m["f1_weighted"],
            }
        )
    return pd.DataFrame(rows).sort_values("F1 (macro)", ascending=False)


def select_best_model(metrics: dict, metric: str = config.SELECTION_METRIC) -> str:
    """Retorna o nome do modelo com maior valor da métrica de seleção."""
    return max(metrics.items(), key=lambda kv: kv[1][metric])[0]


def feature_importance(model, feature_names: list[str]) -> pd.DataFrame:
    """
    Extrai a importância de features de modelos baseados em árvore.

    Funciona para Decision Tree, Random Forest e XGBoost (todos expõem
    `feature_importances_`).
    """
    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame(columns=["feature", "importance"])
    imp = pd.DataFrame(
        {"feature": feature_names, "importance": model.feature_importances_}
    )
    return imp.sort_values("importance", ascending=False).reset_index(drop=True)


def save_metrics(metrics: dict, best_model_name: str, path=config.METRICS_JSON) -> None:
    """Persiste as métricas em JSON (auditoria/relatório)."""
    payload = {"best_model": best_model_name, "metrics": metrics}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    from src import data_loader, feature_engineering, preprocessing, training

    raw, _ = data_loader.load_dataset()
    raw = feature_engineering.add_synthetic_timestamp(raw)
    X, y, art = preprocessing.preprocess(raw)
    res = training.train_all(X, y, art)
    print(metrics_to_frame(res.metrics).to_string(index=False))
    print("\nMelhor modelo:", res.best_model_name)
