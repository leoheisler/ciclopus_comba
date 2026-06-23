"""
feature_engineering.py
======================
Engenharia de atributos do projeto.

Foco principal: GERAÇÃO DE TIMESTAMPS SINTÉTICOS.

    ┌──────────────────────────────────────────────────────────────────┐
    │  AVISO: O CICIDS-2017 não possui uma linha temporal contínua       │
    │  adequada para monitoramento. A coluna `Timestamp` criada aqui é   │
    │  100% SINTÉTICA e existe apenas para SIMULAÇÃO OPERACIONAL de SOC   │
    │  (timelines, picos, sazonalidade). Não representa o tempo real     │
    │  dos fluxos capturados.                                            │
    └──────────────────────────────────────────────────────────────────┘

A estratégia de geração:
    * distribui os eventos ao longo de N dias consecutivos;
    * tráfego BENIGN segue um padrão diurno (picos em horário comercial);
    * ataques (DDoS / PortScan) são concentrados em janelas de incidente
      definidas em config, produzindo "picos de atividade" plausíveis;
    * as proporções entre classes são preservadas (apenas o tempo é atribuído);
    * adiciona colunas derivadas (hour, day, date, weekday) para as análises.

Também concentra utilidades de derivação temporal para o dashboard.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config


# ------------------------------------------------------------------
# Distribuições temporais auxiliares
# ------------------------------------------------------------------
def _diurnal_weights(business_hours: tuple[int, int]) -> np.ndarray:
    """
    Peso de probabilidade por hora do dia (0-23) imitando tráfego corporativo:
    baixo de madrugada, pico no horário comercial, queda à noite.
    """
    start, end = business_hours
    hours = np.arange(24)
    # curva sino centrada no meio do expediente
    center = (start + end) / 2
    spread = (end - start) / 2 + 1
    weights = np.exp(-0.5 * ((hours - center) / spread) ** 2)
    weights += 0.05  # piso de tráfego noturno (não zero)
    return weights / weights.sum()


def _random_times_in_window(
    rng: np.random.Generator,
    base_day: pd.Timestamp,
    day_offset: int,
    hour_start: int,
    hour_end: int,
    n: int,
) -> np.ndarray:
    """Gera n instantes uniformes dentro de [hour_start, hour_end) de um dia."""
    day = base_day + pd.Timedelta(days=day_offset)
    start = day + pd.Timedelta(hours=hour_start)
    span_seconds = (hour_end - hour_start) * 3600
    offsets = rng.integers(0, max(span_seconds, 1), size=n)
    return (start + pd.to_timedelta(offsets, unit="s")).values


def _spread_over_days(
    rng: np.random.Generator,
    base_day: pd.Timestamp,
    n_days: int,
    business_hours: tuple[int, int],
    n: int,
) -> np.ndarray:
    """Distribui n eventos ao longo de n_days respeitando o padrão diurno."""
    day_idx = rng.integers(0, n_days, size=n)
    hour_p = _diurnal_weights(business_hours)
    hours = rng.choice(24, size=n, p=hour_p)
    minutes = rng.integers(0, 60, size=n)
    seconds = rng.integers(0, 60, size=n)
    days = base_day + pd.to_timedelta(day_idx, unit="D")
    return (
        days
        + pd.to_timedelta(hours, unit="h")
        + pd.to_timedelta(minutes, unit="m")
        + pd.to_timedelta(seconds, unit="s")
    ).values


# ------------------------------------------------------------------
# Geração de timestamps sintéticos
# ------------------------------------------------------------------
def add_synthetic_timestamp(
    df: pd.DataFrame, settings: dict | None = None
) -> pd.DataFrame:
    """
    Adiciona a coluna `Timestamp` sintética e colunas temporais derivadas.

    O tráfego benigno é espalhado por todos os dias seguindo o padrão diurno.
    Cada classe de ataque é majoritariamente alocada às janelas de incidente
    configuradas (criando picos); uma pequena fração (ruído de fundo) é
    espalhada para realismo.

    Retorna uma cópia do DataFrame com as colunas:
        Timestamp, date, hour, day_label, weekday
    """
    settings = settings or config.SYNTHETIC_TIME
    rng = np.random.default_rng(settings["seed"])
    base_day = pd.Timestamp(settings["start_date"])
    n_days = settings["n_days"]
    business_hours = tuple(settings["business_hours"])
    attack_windows = settings["attack_windows"]

    df = df.reset_index(drop=True).copy()
    timestamps = np.empty(len(df), dtype="datetime64[ns]")

    for label in df[config.LABEL_COL].unique():
        mask = (df[config.LABEL_COL] == label).values
        idx = np.flatnonzero(mask)
        n = idx.size
        if n == 0:
            continue

        if label not in attack_windows:
            # BENIGN (ou classe sem janela): espalha por todos os dias
            timestamps[idx] = _spread_over_days(
                rng, base_day, n_days, business_hours, n
            )
            continue

        # Ataque: ~85% concentrado nas janelas de incidente, ~15% de fundo
        windows = attack_windows[label]
        n_peak = int(n * 0.85)
        n_bg = n - n_peak

        # divide os eventos de pico entre as janelas
        peak_idx = idx[:n_peak]
        if windows and n_peak > 0:
            assignments = rng.integers(0, len(windows), size=n_peak)
            for w, (d_off, h0, h1) in enumerate(windows):
                sub = peak_idx[assignments == w]
                if sub.size:
                    timestamps[sub] = _random_times_in_window(
                        rng, base_day, d_off, h0, h1, sub.size
                    )

        # ruído de fundo espalhado
        bg_idx = idx[n_peak:]
        if n_bg > 0:
            timestamps[bg_idx] = _spread_over_days(
                rng, base_day, n_days, business_hours, n_bg
            )

    df[config.TIMESTAMP_COL] = pd.to_datetime(timestamps)
    df = df.sort_values(config.TIMESTAMP_COL).reset_index(drop=True)
    return _add_time_parts(df)


def _add_time_parts(df: pd.DataFrame) -> pd.DataFrame:
    """Cria colunas derivadas a partir do Timestamp para facilitar análises."""
    ts = df[config.TIMESTAMP_COL]
    df["date"] = ts.dt.date
    df["hour"] = ts.dt.hour
    df["weekday"] = ts.dt.day_name()
    df["day_label"] = ts.dt.strftime("%Y-%m-%d (%a)")
    return df


# ------------------------------------------------------------------
# Agregações temporais para o dashboard
# ------------------------------------------------------------------
def traffic_by_hour(df: pd.DataFrame) -> pd.DataFrame:
    """Volume total de fluxos por hora do dia (0-23)."""
    out = df.groupby("hour").size().reindex(range(24), fill_value=0)
    return out.rename("count").reset_index()


def events_by_hour_class(df: pd.DataFrame) -> pd.DataFrame:
    """Contagem de eventos por hora e por classe."""
    out = (
        df.groupby(["hour", config.LABEL_COL]).size().reset_index(name="count")
    )
    return out


def events_by_day_class(df: pd.DataFrame) -> pd.DataFrame:
    """Contagem de eventos por dia e por classe."""
    out = (
        df.groupby(["day_label", config.LABEL_COL]).size().reset_index(name="count")
    )
    return out


def timeline(df: pd.DataFrame, freq: str = "1h") -> pd.DataFrame:
    """
    Série temporal contínua (resample) por classe: base para a timeline
    de incidentes e evolução de DDoS/PortScan.
    """
    g = (
        df.set_index(config.TIMESTAMP_COL)
        .groupby(config.LABEL_COL)
        .resample(freq)
        .size()
        .reset_index(name="count")
    )
    return g


def peak_periods(df: pd.DataFrame, freq: str = "1h", top: int = 5) -> pd.DataFrame:
    """Identifica os períodos de maior volume de ataques."""
    attacks = df[df[config.LABEL_COL] != "BENIGN"]
    if attacks.empty:
        return pd.DataFrame(columns=[config.TIMESTAMP_COL, "count"])
    series = (
        attacks.set_index(config.TIMESTAMP_COL).resample(freq).size()
    )
    top_periods = series.sort_values(ascending=False).head(top)
    return top_periods.reset_index(name="count")


if __name__ == "__main__":
    from src import data_loader

    raw, _ = data_loader.load_dataset()
    df = add_synthetic_timestamp(raw)
    print(df[[config.TIMESTAMP_COL, config.LABEL_COL, "hour"]].head())
    print("\nPicos de ataque:")
    print(peak_periods(df))
