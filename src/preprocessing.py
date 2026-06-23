"""
preprocessing.py
================
Limpeza e preparação do dataset CICIDS-2017 para modelagem.

Etapas:
    * conversão de colunas de features para numérico;
    * tratamento de valores infinitos (comuns em Flow Bytes/s) e ausentes;
    * remoção de duplicatas e de colunas constantes/sem variância;
    * separação de features (X) e rótulo (y);
    * codificação de rótulos (LabelEncoder) e escalonamento opcional.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

import config


@dataclass
class PreprocessArtifacts:
    """Objetos ajustados no preprocessamento, reutilizados na predição."""
    feature_names: list[str]
    label_encoder: LabelEncoder
    scaler: StandardScaler | None = None
    feature_stats: pd.DataFrame = field(default_factory=pd.DataFrame)


def _coerce_numeric(df: pd.DataFrame, exclude: list[str]) -> pd.DataFrame:
    """Força colunas de feature a numérico, transformando erros em NaN."""
    df = df.copy()
    for col in df.columns:
        if col in exclude:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpeza básica do DataFrame bruto.

    Mantém a coluna de rótulo (e Timestamp, se já existir) e devolve um
    DataFrame numérico pronto para feature engineering.
    """
    keep = [config.LABEL_COL]
    if config.TIMESTAMP_COL in df.columns:
        keep.append(config.TIMESTAMP_COL)

    df = _coerce_numeric(df, exclude=keep)

    # infinitos -> NaN
    feature_cols = [c for c in df.columns if c not in keep]
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)

    # remove colunas totalmente vazias
    df = df.dropna(axis=1, how="all")

    # preenche NaN restantes com a mediana da coluna (robusto a outliers)
    feature_cols = [c for c in df.columns if c not in keep]
    medians = df[feature_cols].median(numeric_only=True)
    df[feature_cols] = df[feature_cols].fillna(medians)

    # remove duplicatas exatas
    df = df.drop_duplicates().reset_index(drop=True)

    # remove colunas constantes (variância zero): não ajudam o modelo
    nunique = df[feature_cols].nunique()
    constant_cols = nunique[nunique <= 1].index.tolist()
    if constant_cols:
        df = df.drop(columns=constant_cols)

    return df


def split_features_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa a matriz de features X do vetor de rótulos y."""
    drop_cols = [config.LABEL_COL]
    if config.TIMESTAMP_COL in df.columns:
        drop_cols.append(config.TIMESTAMP_COL)
    X = df.drop(columns=drop_cols)
    # garante apenas colunas numéricas em X
    X = X.select_dtypes(include=[np.number])
    y = df[config.LABEL_COL]
    return X, y


def fit_label_encoder(y: pd.Series) -> LabelEncoder:
    """Ajusta o LabelEncoder respeitando a ordem canônica das classes."""
    le = LabelEncoder()
    # força a ordem definida em config para consistência de cores/índices
    le.fit(config.TARGET_CLASSES)
    return le


def build_feature_stats(X: pd.DataFrame) -> pd.DataFrame:
    """Estatísticas descritivas por feature (usadas no simulador de predição)."""
    stats = X.describe(percentiles=[0.05, 0.5, 0.95]).T
    stats = stats.rename(columns={"50%": "median"})
    return stats


def preprocess(
    df: pd.DataFrame, scale: bool = False
) -> tuple[pd.DataFrame, np.ndarray, PreprocessArtifacts]:
    """
    Pipeline completo de preprocessamento.

    Parâmetros
    ----------
    df : DataFrame bruto (com coluna Label e, opcionalmente, Timestamp).
    scale : se True, aplica StandardScaler (útil para modelos lineares;
            árvores não precisam, então o padrão é False).

    Retorna
    -------
    X : DataFrame de features (escalonado ou não).
    y_encoded : np.ndarray de rótulos inteiros.
    artifacts : PreprocessArtifacts com encoder/scaler/estatísticas.
    """
    clean = clean_dataframe(df)
    X, y = split_features_labels(clean)

    le = fit_label_encoder(y)
    y_encoded = le.transform(y)

    feature_stats = build_feature_stats(X)

    scaler = None
    if scale:
        scaler = StandardScaler()
        X = pd.DataFrame(
            scaler.fit_transform(X), columns=X.columns, index=X.index
        )

    artifacts = PreprocessArtifacts(
        feature_names=list(X.columns),
        label_encoder=le,
        scaler=scaler,
        feature_stats=feature_stats,
    )
    return X, y_encoded, artifacts


if __name__ == "__main__":
    from src import data_loader

    raw, _ = data_loader.load_dataset()
    X, y, art = preprocess(raw)
    print("Features:", len(art.feature_names))
    print("Shape X:", X.shape)
    print("Classes:", list(art.label_encoder.classes_))
