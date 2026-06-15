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


def _prepare_dataframe(force_synthetic: bool = False) -> tuple[pd.DataFrame, bool]:
    """Carrega o dataset, amostra e adiciona timestamps sintéticos."""
    raw, synthetic = data_loader.load_dataset(prefer_real=not force_synthetic)

    if config.MAX_SAMPLES and len(raw) > config.MAX_SAMPLES:
        raw = raw.groupby(config.LABEL_COL, group_keys=False).apply(
            lambda g: g.sample(
                n=max(1, int(len(g) / len(raw) * config.MAX_SAMPLES)),
                random_state=config.RANDOM_STATE,
            )
        )

    raw = feature_engineering.add_synthetic_timestamp(raw)
    return raw, synthetic


def build_pipeline(force_synthetic: bool = False, save: bool = True) -> dict:
    """
    Executa o pipeline completo (carregar → timestamp → preprocessar → treinar)
    e retorna o dicionário de artefatos.
    """
    df, synthetic = _prepare_dataframe(force_synthetic)

    X, y, artifacts = preprocessing.preprocess(df)
    result = training.train_all(X, y, artifacts)

    if save:
        df.to_parquet(config.PROCESSED_PARQUET, index=False)
        training.save_bundle(result)

    return {
        "df": df,
        "is_synthetic": synthetic,
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
        # detecta origem pelos nomes de arquivo em data/raw
        is_synthetic = not data_loader.list_raw_files()
        return {
            "df": df,
            "is_synthetic": is_synthetic,
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
