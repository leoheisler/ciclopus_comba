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


# ------------------------------------------------------------------
# Fallback: dataset sintético realista
# ------------------------------------------------------------------
def generate_synthetic_dataset(
    n_per_class: int | None = None,
    seed: int = config.RANDOM_STATE,
) -> pd.DataFrame:
    """
    Gera um dataset SINTÉTICO que imita as principais características de fluxo
    do CICIDS-2017 para BENIGN, DDoS e PortScan.

    NÃO substitui o dataset real - serve apenas para que a aplicação rode
    de ponta a ponta sem o download. As distribuições foram escolhidas para
    que os modelos consigam separar as classes de forma plausível.
    """
    rng = np.random.default_rng(seed)
    n_per_class = n_per_class or {"BENIGN": 30_000, "DDoS": 12_000, "PortScan": 12_000}
    if isinstance(n_per_class, int):
        n_per_class = {c: n_per_class for c in config.TARGET_CLASSES}

    def _block(label: str, n: int, profile: dict) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Flow Duration": rng.gamma(*profile["flow_duration"], n),
                "Total Fwd Packets": rng.poisson(profile["fwd_packets"], n),
                "Total Backward Packets": rng.poisson(profile["bwd_packets"], n),
                "Flow Bytes/s": np.abs(rng.normal(*profile["flow_bytes"], n)),
                "Flow Packets/s": np.abs(rng.normal(*profile["flow_packets"], n)),
                "Fwd Packet Length Mean": np.abs(rng.normal(*profile["fwd_len"], n)),
                "Bwd Packet Length Mean": np.abs(rng.normal(*profile["bwd_len"], n)),
                "Flow IAT Mean": np.abs(rng.normal(*profile["iat_mean"], n)),
                "Fwd IAT Mean": np.abs(rng.normal(*profile["iat_mean"], n)),
                "SYN Flag Count": rng.poisson(profile["syn"], n),
                "ACK Flag Count": rng.poisson(profile["ack"], n),
                "Init_Win_bytes_forward": np.abs(rng.normal(*profile["win"], n)),
                "Destination Port": rng.integers(*profile["dst_port"], n),
                "Average Packet Size": np.abs(rng.normal(*profile["pkt_size"], n)),
                config.LABEL_COL: label,
            }
        )

    profiles = {
        # tráfego normal: fluxos longos, equilíbrio fwd/bwd, ACKs altos
        "BENIGN": dict(
            flow_duration=(2.0, 60000), fwd_packets=12, bwd_packets=11,
            flow_bytes=(8000, 4000), flow_packets=(40, 25), fwd_len=(450, 150),
            bwd_len=(420, 150), iat_mean=(15000, 8000), syn=1, ack=9,
            win=(8192, 2000), dst_port=(20, 1024), pkt_size=(500, 150),
        ),
        # DDoS: muitos pacotes/s, fluxos curtos, payload alto, SYN alto
        "DDoS": dict(
            flow_duration=(1.2, 5000), fwd_packets=60, bwd_packets=3,
            flow_bytes=(90000, 30000), flow_packets=(600, 200), fwd_len=(900, 200),
            bwd_len=(40, 30), iat_mean=(200, 150), syn=20, ack=2,
            win=(256, 100), dst_port=(80, 81), pkt_size=(950, 200),
        ),
        # PortScan: fluxos minúsculos, 1-2 pacotes, SYN alto, portas variadas
        "PortScan": dict(
            flow_duration=(0.8, 800), fwd_packets=2, bwd_packets=1,
            flow_bytes=(50, 30), flow_packets=(900, 300), fwd_len=(40, 15),
            bwd_len=(0, 5), iat_mean=(80, 60), syn=2, ack=0,
            win=(1024, 400), dst_port=(1, 65535), pkt_size=(40, 20),
        ),
    }

    blocks = [_block(c, n_per_class[c], profiles[c]) for c in config.TARGET_CLASSES]
    data = pd.concat(blocks, ignore_index=True)

    # --- Realismo: introduz sobreposição entre classes ---
    # Sem isso, as classes ficam separáveis demais (métricas = 1.0), o que não
    # reflete um problema real. Aplicamos ruído gaussiano proporcional ao desvio
    # de cada feature e um pequeno percentual de rótulos ambíguos.
    feature_cols = [c for c in data.columns if c != config.LABEL_COL]
    noise_scale = 0.55  # intensidade do ruído (≈ fração do desvio da feature)
    stds = data[feature_cols].std().replace(0, 1.0)
    noise = rng.normal(0, 1, size=(len(data), len(feature_cols))) * stds.values
    data[feature_cols] = np.abs(data[feature_cols].values + noise_scale * noise)

    # rótulos ambíguos: 3% dos fluxos recebem rótulo trocado (ataques furtivos /
    # falsos positivos), forçando matrizes de confusão não-triviais.
    n_flip = int(len(data) * 0.03)
    flip_idx = rng.choice(len(data), size=n_flip, replace=False)
    other = {c: [x for x in config.TARGET_CLASSES if x != c] for c in config.TARGET_CLASSES}
    data.loc[flip_idx, config.LABEL_COL] = [
        rng.choice(other[lbl]) for lbl in data.loc[flip_idx, config.LABEL_COL]
    ]

    # embaralha para não ficar ordenado por classe
    data = data.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return data


# ------------------------------------------------------------------
# Interface pública
# ------------------------------------------------------------------
def load_dataset(prefer_real: bool = True) -> tuple[pd.DataFrame, bool]:
    """
    Carrega o dataset para o pipeline.

    Retorna (DataFrame, is_synthetic). Tenta os CSVs reais primeiro; se não
    houver dados (ou prefer_real=False), cai no gerador sintético.
    """
    if prefer_real:
        try:
            return load_raw_csvs(), False
        except FileNotFoundError:
            warnings.warn(
                "Dataset real não encontrado em data/raw: usando dados SINTÉTICOS."
            )
    return generate_synthetic_dataset(), True


if __name__ == "__main__":
    df, synthetic = load_dataset()
    origem = "SINTÉTICO" if synthetic else "REAL (CICIDS-2017)"
    print(f"Origem dos dados: {origem}")
    print(f"Shape: {df.shape}")
    print(df[config.LABEL_COL].value_counts())
