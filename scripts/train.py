"""
scripts/train.py
================
Script de linha de comando que executa o pipeline completo de Data Science
de ponta a ponta e persiste os artefatos para o dashboard:

    1. carrega o dataset (real ou sintético)            -> data_loader
    2. gera timestamps sintéticos                        -> feature_engineering
    3. limpa e prepara as features                       -> preprocessing
    4. treina Decision Tree, Random Forest e XGBoost     -> training
    5. avalia e seleciona o melhor modelo                -> evaluation
    6. salva dataset processado (parquet) + bundle (.joblib)

Uso:
    python -m scripts.train
    python -m scripts.train --synthetic   # força dados sintéticos
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# permite rodar como `python scripts/train.py` ou `python -m scripts.train`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src import data_loader, evaluation, feature_engineering, preprocessing, training


def run(force_synthetic: bool = False) -> None:
    t0 = time.time()
    print("=" * 60)
    print("CICLOPUS COMBA — Pipeline de treinamento")
    print("=" * 60)

    # 1. Carregar
    raw, synthetic = data_loader.load_dataset(prefer_real=not force_synthetic)
    origem = "SINTÉTICO" if synthetic else "REAL (CICIDS-2017)"
    print(f"[1/6] Dados carregados ({origem}): {raw.shape}")

    # Amostragem para manter o treino ágil, preservando proporção de classes
    if config.MAX_SAMPLES and len(raw) > config.MAX_SAMPLES:
        raw = raw.groupby(config.LABEL_COL, group_keys=False).apply(
            lambda g: g.sample(
                n=max(1, int(len(g) / len(raw) * config.MAX_SAMPLES)),
                random_state=config.RANDOM_STATE,
            )
        )
        print(f"      Amostrado para {len(raw)} registros (MAX_SAMPLES).")

    # 2. Timestamps sintéticos
    raw = feature_engineering.add_synthetic_timestamp(raw)
    print(f"[2/6] Timestamps sintéticos gerados "
          f"({raw[config.TIMESTAMP_COL].min()} → {raw[config.TIMESTAMP_COL].max()})")

    # 3. Preprocessamento
    X, y, artifacts = preprocessing.preprocess(raw)
    print(f"[3/6] Preprocessamento concluído: {len(artifacts.feature_names)} features")

    # salva dataset processado (com Timestamp) para o dashboard
    raw.to_parquet(config.PROCESSED_PARQUET, index=False)
    print(f"      Dataset processado salvo em {config.PROCESSED_PARQUET}")

    # 4 + 5. Treino e avaliação
    print("[4/6] Treinando modelos (Decision Tree, Random Forest, XGBoost)...")
    result = training.train_all(X, y, artifacts)
    print("[5/6] Avaliação:")
    print(evaluation.metrics_to_frame(result.metrics).to_string(index=False))
    print(f"      >> Melhor modelo: {result.best_model_name}")

    # 6. Persistência
    training.save_bundle(result)
    print(f"[6/6] Bundle salvo em {config.MODELS_BUNDLE}")
    print(f"\nConcluído em {time.time() - t0:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de treino CICLOPUS COMBA")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Força o uso de dados sintéticos (ignora data/raw).",
    )
    args = parser.parse_args()
    run(force_synthetic=args.synthetic)
