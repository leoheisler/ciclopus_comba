"""
config.py
=========
Configurações globais do projeto CICLOPUS COMBA (SOC Analytics Platform).

Centraliza caminhos, constantes, parâmetros de modelos e definições de cores
para que todos os módulos compartilhem a mesma fonte de verdade.
"""
from __future__ import annotations

from pathlib import Path

# ------------------------------------------------------------------
# Caminhos do projeto
# ------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"             # CSVs originais do CICIDS-2017
PROCESSED_DIR = DATA_DIR / "processed"  # dataset tratado em parquet
MODELS_DIR = ROOT_DIR / "models"        # modelos serializados
ASSETS_DIR = ROOT_DIR / "assets"

for _d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, ASSETS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Arquivos gerados pelo pipeline
PROCESSED_PARQUET = PROCESSED_DIR / "cicids_processed.parquet"
MODELS_BUNDLE = MODELS_DIR / "models_bundle.joblib"
METRICS_JSON = MODELS_DIR / "metrics.json"

# ------------------------------------------------------------------
# Definição do problema
# ------------------------------------------------------------------
# Classes-alvo utilizadas (subconjunto do CICIDS-2017)
TARGET_CLASSES = ["BENIGN", "DDoS", "PortScan"]

# Nome da coluna de rótulo após a padronização de colunas
LABEL_COL = "Label"
TIMESTAMP_COL = "Timestamp"

# Mapeamento de rótulos brutos -> rótulos canônicos.
# O CICIDS-2017 traz variações de grafia entre arquivos/fontes.
LABEL_NORMALIZATION = {
    "BENIGN": "BENIGN",
    "Benign": "BENIGN",
    "DDOS": "DDoS",
    "DDoS": "DDoS",
    "DDos": "DDoS",
    "PortScan": "PortScan",
    "Portscan": "PortScan",
    "PORTSCAN": "PortScan",
}

# ------------------------------------------------------------------
# Geração de timestamps sintéticos
# ------------------------------------------------------------------
# IMPORTANTE: o CICIDS-2017 não possui uma linha temporal contínua
# adequada para monitoramento operacional. A coluna Timestamp é
# SINTÉTICA e serve apenas para fins de visualização/simulação de SOC.
SYNTHETIC_TIME = {
    "n_days": 5,                       # nº de dias simulados
    "start_date": "2017-07-03",        # segunda-feira (semana do CICIDS-2017)
    "business_hours": (8, 18),         # janela de pico de tráfego benigno
    "seed": 42,
    # Janelas de incidente (dia_offset, hora_inicio, hora_fim) por classe.
    # Concentram ataques para gerar "picos" plausíveis na timeline.
    "attack_windows": {
        "DDoS": [(1, 10, 12), (3, 14, 16)],
        "PortScan": [(0, 9, 11), (2, 13, 15), (4, 16, 18)],
    },
}

# ------------------------------------------------------------------
# Modelagem
# ------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.25

# Amostragem para manter o app responsivo (None = usar tudo).
MAX_SAMPLES = 120_000

MODEL_PARAMS = {
    "Decision Tree": {
        "max_depth": 18,
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
    },
    "Random Forest": {
        "n_estimators": 200,
        "max_depth": 22,
        "n_jobs": -1,
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
    },
    "XGBoost": {
        "n_estimators": 300,
        "max_depth": 8,
        "learning_rate": 0.2,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "tree_method": "hist",
        "n_jobs": -1,
        "random_state": RANDOM_STATE,
    },
}

# Métrica usada para selecionar automaticamente o melhor modelo
SELECTION_METRIC = "f1_macro"

# ------------------------------------------------------------------
# Identidade visual (tema SOC corporativo, dark)
# ------------------------------------------------------------------
COLORS = {
    "BENIGN": "#2DD4BF",     # teal
    "DDoS": "#F87171",       # vermelho
    "PortScan": "#FBBF24",   # âmbar
    "bg": "#0E1117",
    "panel": "#161B26",
    "accent": "#3B82F6",
    "ok": "#22C55E",
    "warn": "#F59E0B",
    "crit": "#EF4444",
    "text": "#E5E7EB",
    "muted": "#9CA3AF",
}

CLASS_COLOR_MAP = {c: COLORS[c] for c in TARGET_CLASSES}

PLOTLY_TEMPLATE = "plotly_dark"
