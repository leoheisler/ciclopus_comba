"""
data_loader.py
==============
Carregamento do dataset CICIDS-2017.

Responsabilidades:
    * localizar e ler os CSVs originais do CICIDS-2017 (pasta data/raw);
    * concatenar os arquivos em um único DataFrame;
    * normalizar nomes de colunas (os arquivos vêm com espaços iniciais);
    * normalizar e filtrar os rótulos para as classes-alvo do projeto
      (BENIGN, DDoS, PortScan);
    * caso nenhum CSV seja encontrado, gerar um dataset SINTÉTICO realista
      para que a aplicação seja 100% executável sem download.

Os arquivos esperados do Kaggle (CICIDS-2017 / MachineLearningCVE) incluem,
entre outros:
    - Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
    - Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
    - Monday-WorkingHours.pcap_ISCX.csv   (tráfego BENIGN)
"""
from __future__ import annotations

import glob
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import config


# ------------------------------------------------------------------
# Leitura dos CSVs reais
# ------------------------------------------------------------------
def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove espaços e padroniza os nomes das colunas."""
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace("﻿", "", regex=False)  # BOM ocasional
    )
    return df


def _find_label_column(df: pd.DataFrame) -> str | None:
    """Identifica a coluna de rótulo independentemente da grafia."""
    for col in df.columns:
        if col.strip().lower() == "label":
            return col
    return None


def list_raw_files(raw_dir: Path | None = None) -> list[Path]:
    """Lista os CSVs disponíveis em data/raw."""
    raw_dir = Path(raw_dir) if raw_dir else config.RAW_DIR
    return sorted(Path(p) for p in glob.glob(str(raw_dir / "*.csv")))


def load_raw_csvs(raw_dir: Path | None = None) -> pd.DataFrame:
    """
    Lê e concatena todos os CSVs do CICIDS-2017 encontrados em data/raw.

    Retorna um DataFrame com a coluna de rótulo padronizada como `Label`
    e filtrado para as classes-alvo. Lança FileNotFoundError se nada for
    encontrado (o chamador decide usar o fallback sintético).
    """
    files = list_raw_files(raw_dir)
    if not files:
        raise FileNotFoundError("Nenhum CSV do CICIDS-2017 encontrado em data/raw.")

    frames: list[pd.DataFrame] = []
    for path in files:
        try:
            df = pd.read_csv(path, low_memory=False, encoding="latin-1")
        except Exception as exc:  # pragma: no cover - robustez de I/O
            warnings.warn(f"Falha ao ler {path.name}: {exc}")
            continue

        df = _normalize_columns(df)
        label_col = _find_label_column(df)
        if label_col is None:
            warnings.warn(f"{path.name} não possui coluna de rótulo; ignorado.")
            continue

        df = df.rename(columns={label_col: config.LABEL_COL})
        frames.append(df)

    if not frames:
        raise FileNotFoundError("Arquivos encontrados, mas nenhum válido foi lido.")

    data = pd.concat(frames, ignore_index=True)
    data = _normalize_and_filter_labels(data)
    return data


def _normalize_and_filter_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza grafias dos rótulos e mantém somente as classes-alvo."""
    df = df.copy()
    df[config.LABEL_COL] = (
        df[config.LABEL_COL].astype(str).str.strip().map(
            lambda v: config.LABEL_NORMALIZATION.get(v, v)
        )
    )
    df = df[df[config.LABEL_COL].isin(config.TARGET_CLASSES)].reset_index(drop=True)
    return df


def balanced_sample_by_label(
    df: pd.DataFrame,
    max_samples: int,
    random_state: int,
) -> pd.DataFrame:
    """Amostra preservando a proporção das classes sem perder a coluna Label."""
    if max_samples <= 0 or len(df) <= max_samples:
        return df.reset_index(drop=True)

    sampled_frames: list[pd.DataFrame] = []
    total = len(df)
    for _, group in df.groupby(config.LABEL_COL, sort=False):
        n = max(1, int(len(group) / total * max_samples))
        sampled_frames.append(group.sample(n=n, random_state=random_state))

    sampled = pd.concat(sampled_frames, ignore_index=True)
    return sampled.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

# ------------------------------------------------------------------
# Interface pública
# ------------------------------------------------------------------
def load_dataset() -> pd.DataFrame:
    """
    Carrega o dataset para o pipeline.

    Retorna DataFrame. Tenta os CSVs reais
    """
    try:
        return load_raw_csvs()
    except FileNotFoundError:
        warnings.warn(
            "Dataset real não encontrado em data/raw"
        )
        raise


if __name__ == "__main__":
    df = load_dataset()
    print(f"Shape: {df.shape}")
    print(df[config.LABEL_COL].value_counts())
