"""
pipeline.py
===========
Orquestração reutilizável pelo dashboard.

Fornece `get_pipeline()`, que devolve um dicionário com tudo que a aplicação
precisa: dataset processado (com Timestamp sintético), modelos treinados,
métricas, encoder, etc. Reaproveita artefatos em disco quando existem e, caso
contrário, executa o pipeline completo e os persiste.
"""
from __future__ import annotations

import pandas as pd

import config
from src import data_loader, feature_engineering, preprocessing, training


def _prepare_dataframe() -> pd.DataFrame:
    """Carrega o dataset, amostra e adiciona timestamps sintéticos."""
    raw = data_loader.load_dataset()

    if config.MAX_SAMPLES and len(raw) > config.MAX_SAMPLES:
        raw = data_loader.balanced_sample_by_label(
            raw,
            max_samples=config.MAX_SAMPLES,
            random_state=config.RANDOM_STATE,
        )

    raw = feature_engineering.add_synthetic_timestamp(raw)
    return raw


def build_pipeline(save: bool = True) -> dict:
    """
    Executa o pipeline completo (carregar → timestamp → preprocessar → treinar)
    e retorna o dicionário de artefatos.
    """
    df = _prepare_dataframe()

    X, y, artifacts = preprocessing.preprocess(df)
    result = training.train_all(X, y, artifacts)

    if save:
        df.to_parquet(config.PROCESSED_PARQUET, index=False)
        training.save_bundle(result)

    return {
        "df": df,
        "models": result.models,
        "metrics": result.metrics,
        "best_model_name": result.best_model_name,
        "feature_names": result.feature_names,
        "label_encoder": artifacts.label_encoder,
        "scaler": artifacts.scaler,
        "feature_stats": artifacts.feature_stats,
    }


def get_pipeline(force_retrain: bool = False) -> dict:
    """
    Ponto de entrada principal: tenta reaproveitar artefatos em disco;
    se não houver (ou force_retrain=True), constrói tudo do zero.
    """
    if force_retrain:
        return build_pipeline()

    bundle = training.load_bundle()
    if bundle is not None and config.PROCESSED_PARQUET.exists():
        df = pd.read_parquet(config.PROCESSED_PARQUET)
        return {
            "df": df,
            "models": bundle["models"],
            "metrics": bundle["metrics"],
            "best_model_name": bundle["best_model_name"],
            "feature_names": bundle["feature_names"],
            "label_encoder": bundle["label_encoder"],
            "scaler": bundle["scaler"],
            "feature_stats": bundle["feature_stats"],
        }

    # nada em disco → constrói
    return build_pipeline()
