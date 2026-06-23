"""
training.py
===========
Treinamento dos modelos de classificação de tráfego de rede.

Modelos:
    * Decision Tree
    * Random Forest
    * XGBoost

Fluxo:
    * separa treino/teste de forma estratificada;
    * treina os três algoritmos com os hiperparâmetros de config;
    * avalia cada um (delegando ao módulo evaluation);
    * seleciona automaticamente o melhor modelo pela métrica configurada;
    * serializa um "bundle" com modelos, métricas e artefatos de preprocesso.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

import config
from src import evaluation
from src.preprocessing import PreprocessArtifacts


@dataclass
class TrainResult:
    """Resultado completo do treinamento, pronto para o dashboard."""
    models: dict
    metrics: dict
    best_model_name: str
    feature_names: list[str]
    artifacts: PreprocessArtifacts
    X_test: pd.DataFrame
    y_test: np.ndarray


def build_models() -> dict:
    """Instancia os três classificadores com os parâmetros de config."""
    return {
        "Decision Tree": DecisionTreeClassifier(**config.MODEL_PARAMS["Decision Tree"]),
        "Random Forest": RandomForestClassifier(**config.MODEL_PARAMS["Random Forest"]),
        "XGBoost": XGBClassifier(
            **config.MODEL_PARAMS["XGBoost"],
            num_class=len(config.TARGET_CLASSES),
            objective="multi:softprob",
            eval_metric="mlogloss",
        ),
    }


def train_all(
    X: pd.DataFrame,
    y: np.ndarray,
    artifacts: PreprocessArtifacts,
) -> TrainResult:
    """
    Treina todos os modelos e seleciona o melhor automaticamente.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )

    models = build_models()
    metrics: dict = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics[name] = evaluation.compute_metrics(
            y_test, y_pred, labels=list(artifacts.label_encoder.classes_)
        )

    best_model_name = evaluation.select_best_model(metrics, config.SELECTION_METRIC)

    return TrainResult(
        models=models,
        metrics=metrics,
        best_model_name=best_model_name,
        feature_names=artifacts.feature_names,
        artifacts=artifacts,
        X_test=X_test,
        y_test=y_test,
    )


def save_bundle(result: TrainResult, path=config.MODELS_BUNDLE) -> None:
    """Serializa modelos + artefatos + métricas em disco (joblib)."""
    import joblib

    bundle = {
        "models": result.models,
        "metrics": result.metrics,
        "best_model_name": result.best_model_name,
        "feature_names": result.feature_names,
        "label_encoder": result.artifacts.label_encoder,
        "scaler": result.artifacts.scaler,
        "feature_stats": result.artifacts.feature_stats,
    }
    joblib.dump(bundle, path)

    evaluation.save_metrics(result.metrics, result.best_model_name)


def load_bundle(path=config.MODELS_BUNDLE) -> dict | None:
    """Carrega o bundle treinado, ou None se ainda não existir."""
    import joblib

    if not path.exists():
        return None
    return joblib.load(path)


if __name__ == "__main__":
    from src import data_loader, feature_engineering, preprocessing

    raw = data_loader.load_dataset()
    raw = feature_engineering.add_synthetic_timestamp(raw)
    X, y, art = preprocessing.preprocess(raw)
    res = train_all(X, y, art)
    print("Melhor modelo:", res.best_model_name)
    for name, m in res.metrics.items():
        print(f"  {name}: f1_macro={m['f1_macro']:.4f} acc={m['accuracy']:.4f}")
    save_bundle(res)
    print("Bundle salvo em", config.MODELS_BUNDLE)
