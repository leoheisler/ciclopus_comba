"""
dashboard.py
============
Network Attack Detection Analysis.

Aplicação web interativa (Streamlit + Plotly) que simula um Centro de
Operações de Segurança (SOC) sobre o dataset CICIDS-2017, cobrindo todo o
pipeline de Data Science: exploração, análise temporal, machine learning,
importância de variáveis, simulador de predição e tela operacional de SOC.

Execução:
    streamlit run app/dashboard.py

Páginas:
    1. Overview                 5. Feature Importance
    2. Exploração dos Dados     6. Simulador de Predição
    3. Monitoramento Temporal   7. Security Operations Center
    4. Machine Learning
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# permite importar o pacote `src` e `config` ao rodar via `streamlit run`
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from src import evaluation, feature_engineering  # noqa: E402
from src.pipeline import get_pipeline  # noqa: E402

# ==================================================================
# Configuração da página + tema
# ==================================================================
st.set_page_config(
    page_title="Network Attack Detection Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = f"""
<style>
    .stApp {{ background-color: {config.COLORS['bg']}; }}
    section[data-testid="stSidebar"] {{ background-color: {config.COLORS['panel']}; }}
    .block-container {{ padding-top: 1.5rem; }}
    h1, h2, h3 {{ color: {config.COLORS['text']}; font-family: 'Segoe UI', sans-serif; }}
    .soc-card {{
        background: linear-gradient(145deg, #1b2230, #11161f);
        border: 1px solid #2a3344; border-radius: 12px;
        padding: 16px 18px; margin-bottom: 8px;
    }}
    .soc-metric-label {{ color: {config.COLORS['muted']}; font-size: 0.8rem;
        text-transform: uppercase; letter-spacing: 0.08em; }}
    .soc-metric-value {{ font-size: 1.9rem; font-weight: 700; }}
    .badge {{ padding: 2px 10px; border-radius: 999px; font-size: 0.75rem;
        font-weight: 600; }}
    .pulse {{ animation: pulse 1.6s infinite; }}
    @keyframes pulse {{ 0% {{opacity:1}} 50% {{opacity:0.35}} 100% {{opacity:1}} }}

    /* Botões: visual arredondado com realce ao passar o mouse */
    .stButton > button {{
        border-radius: 10px;
        border: 1px solid #2a3344;
        background: linear-gradient(145deg, #1b2230, #141a25);
        color: {config.COLORS['text']};
        font-weight: 600;
        padding: 0.45rem 0.6rem;
        transition: all .15s ease;
    }}
    .stButton > button:hover {{
        border-color: {config.COLORS['accent']};
        color: #ffffff;
        background: linear-gradient(145deg, #222c3d, #1a2230);
        box-shadow: 0 0 0 2px rgba(59,130,246,.25);
        transform: translateY(-1px);
    }}
    .stButton > button:active {{ transform: translateY(0); }}
    /* Botão primário (Avançar) com a cor de destaque */
    .stButton > button[kind="primary"] {{
        background: linear-gradient(145deg, {config.COLORS['accent']}, #2563EB);
        border-color: {config.COLORS['accent']}; color: #ffffff;
    }}
    .stButton > button[kind="primary"]:hover {{
        background: linear-gradient(145deg, #4b8ef8, #2f6ef0);
        box-shadow: 0 0 0 3px rgba(59,130,246,.35);
    }}

    /* Painel de controle do SOC */
    .soc-controls {{
        background: linear-gradient(145deg, #1b2230, #11161f);
        border: 1px solid #2a3344; border-radius: 14px;
        padding: 6px 14px 2px 14px; margin-bottom: 10px;
    }}
    .soc-controls-title {{
        color: {config.COLORS['muted']}; font-size: 0.75rem;
        text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 2px;
    }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==================================================================
# Carregamento (cacheado) do pipeline
# ==================================================================
@st.cache_resource(show_spinner="Carregando dados e treinando modelos...")
def load_pipeline(force: bool = False) -> dict:
    return get_pipeline(force_retrain=force)


def style_fig(fig: go.Figure, height: int = 400) -> go.Figure:
    """Aplica o tema visual padrão aos gráficos Plotly.

    Margens generosas + título ancorado no 'container' (acima da área de
    plotagem) e legenda logo abaixo do título evitam que o texto invada o
    gráfico.
    """
    fig.update_layout(
        template=config.PLOTLY_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=dict(l=55, r=30, t=90, b=55),
        title=dict(
            x=0.01, xanchor="left",
            y=0.97, yanchor="top", yref="container",
            font=dict(size=16, color=config.COLORS["text"]),
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.0, x=0,
            bgcolor="rgba(0,0,0,0)",
        ),
        font=dict(color=config.COLORS["text"]),
    )
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)
    return fig


def kpi_card(label: str, value: str, color: str = None, sub: str = "") -> str:
    color = color or config.COLORS["text"]
    sub_html = f"<div class='soc-metric-label'>{sub}</div>" if sub else ""
    return (
        f"<div class='soc-card'>"
        f"<div class='soc-metric-label'>{label}</div>"
        f"<div class='soc-metric-value' style='color:{color}'>{value}</div>"
        f"{sub_html}</div>"
    )


# ==================================================================
# Sidebar: identidade, filtros globais e estado dos dados
# ==================================================================
def sidebar(pipe: dict) -> dict:
    st.sidebar.markdown("## Network Attack Detection Analysis")
    st.sidebar.caption("CICIDS-2017")

    if pipe["is_synthetic"]:
        st.sidebar.warning(
            "Usando **dados sintéticos** (CSVs do CICIDS-2017 não "
            "encontrados em `data/raw`). O pipeline é idêntico para dados reais."
        )

    page = st.sidebar.radio(
        "Navegação",
        [
            "1 · Overview",
            "2 · Exploração dos Dados",
            "3 · Monitoramento Temporal",
            "4 · Feature Importance",
            "5 · Simulador de Predição",
            "6 · Security Operations Center",
        ],
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Filtros globais")

    df = pipe["df"]
    dates = sorted(df[config.TIMESTAMP_COL].dt.date.unique())

    # botão de reset: limpa o estado dos filtros e recarrega com os padrões
    if st.sidebar.button("Limpar filtros", width="stretch"):
        for k in ("flt_classes", "flt_dates", "flt_hours"):
            st.session_state.pop(k, None)
        st.rerun()

    # Classes via "pills": basta clicar para ligar/desligar cada classe.
    classes = st.sidebar.pills(
        "Classes de tráfego (clique para ativar)",
        options=config.TARGET_CLASSES,
        selection_mode="multi",
        default=config.TARGET_CLASSES,
        key="flt_classes",
    )
    if not classes:  # nenhuma selecionada = considerar todas
        classes = config.TARGET_CLASSES

    # Período por slider de datas (mais simples que o calendário de intervalo).
    if len(dates) > 1:
        date_range = st.sidebar.select_slider(
            "Período (dias sintéticos)",
            options=dates,
            value=(dates[0], dates[-1]),
            format_func=lambda d: d.strftime("%d/%m"),
            key="flt_dates",
        )
    else:
        date_range = (dates[0], dates[0])

    hours = st.sidebar.slider(
        "Faixa de horas do dia", 0, 23, (0, 23), key="flt_hours"
    )

    st.sidebar.caption(
        f"Mostrando **{', '.join(classes)}** · "
        f"{date_range[0].strftime('%d/%m')}-{date_range[1].strftime('%d/%m')} · "
        f"{hours[0]}h-{hours[1]}h"
    )

    st.sidebar.markdown("---")
    if st.sidebar.button("Re-treinar modelos", width="stretch"):
        load_pipeline.clear()
        load_pipeline(force=True)
        st.rerun()

    st.sidebar.caption(
        "Tempo é **sintético**: apenas para simulação operacional de SOC."
    )

    return {"page": page, "classes": classes, "date_range": date_range, "hours": hours}


def apply_filters(df: pd.DataFrame, f: dict) -> pd.DataFrame:
    out = df[df[config.LABEL_COL].isin(f["classes"])]
    dr = f["date_range"]
    if isinstance(dr, (list, tuple)) and len(dr) == 2:
        start, end = dr
        out = out[
            (out[config.TIMESTAMP_COL].dt.date >= start)
            & (out[config.TIMESTAMP_COL].dt.date <= end)
        ]
    h0, h1 = f["hours"]
    out = out[(out["hour"] >= h0) & (out["hour"] <= h1)]
    return out


# ==================================================================
# Página 1: Overview
# ==================================================================
def page_overview(df: pd.DataFrame, pipe: dict):
    st.title("Overview Operacional")
    st.caption("Visão executiva do tráfego monitorado e da postura de segurança.")

    total = len(df)
    attacks = int((df[config.LABEL_COL] != "BENIGN").sum())
    benign = total - attacks
    pct_attacks = (attacks / total * 100) if total else 0
    ddos = int((df[config.LABEL_COL] == "DDoS").sum())
    portscan = int((df[config.LABEL_COL] == "PortScan").sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi_card("Total de registros", f"{total:,}".replace(",", ".")),
                unsafe_allow_html=True)
    c2.markdown(kpi_card("Total de ataques", f"{attacks:,}".replace(",", "."),
                         config.COLORS["crit"]), unsafe_allow_html=True)
    c3.markdown(kpi_card("% de ataques", f"{pct_attacks:.1f}%",
                         config.COLORS["warn"]), unsafe_allow_html=True)
    c4.markdown(kpi_card("Tráfego benigno", f"{benign:,}".replace(",", "."),
                         config.COLORS["BENIGN"]), unsafe_allow_html=True)

    c5, c6, c7 = st.columns(3)
    c5.markdown(kpi_card("DDoS detectados", f"{ddos:,}".replace(",", "."),
                         config.COLORS["DDoS"]), unsafe_allow_html=True)
    c6.markdown(kpi_card("PortScan detectados", f"{portscan:,}".replace(",", "."),
                         config.COLORS["PortScan"]), unsafe_allow_html=True)
    c7.markdown(kpi_card("Melhor modelo (F1 macro)",
                         pipe["best_model_name"], config.COLORS["accent"],
                         sub=f"F1={pipe['metrics'][pipe['best_model_name']]['f1_macro']:.3f}"),
                unsafe_allow_html=True)

    st.markdown("### Distribuição do tráfego")
    a, b = st.columns([1, 1.4])

    counts = df[config.LABEL_COL].value_counts().reindex(
        config.TARGET_CLASSES, fill_value=0
    )
    fig_donut = go.Figure(
        go.Pie(
            labels=counts.index, values=counts.values, hole=0.6,
            marker=dict(colors=[config.CLASS_COLOR_MAP[c] for c in counts.index]),
        )
    )
    fig_donut.update_layout(title="Proporção por classe")
    a.plotly_chart(style_fig(fig_donut), width="stretch")

    tl = feature_engineering.timeline(df, freq="3h")
    fig_area = px.area(
        tl, x=config.TIMESTAMP_COL, y="count", color=config.LABEL_COL,
        color_discrete_map=config.CLASS_COLOR_MAP,
        title="Volume de eventos ao longo do tempo (sintético)",
    )
    b.plotly_chart(style_fig(fig_area), width="stretch")


# ==================================================================
# Página 2: Exploração dos Dados
# ==================================================================
def page_eda(df: pd.DataFrame, pipe: dict):
    st.title("Exploração dos Dados (EDA)")

    numeric_cols = [c for c in pipe["feature_names"] if c in df.columns]

    st.markdown("### Distribuição das classes")
    counts = df[config.LABEL_COL].value_counts().reindex(
        config.TARGET_CLASSES, fill_value=0
    ).reset_index()
    counts.columns = [config.LABEL_COL, "count"]
    fig_bar = px.bar(
        counts, x=config.LABEL_COL, y="count", color=config.LABEL_COL,
        color_discrete_map=config.CLASS_COLOR_MAP, text="count",
    )
    st.plotly_chart(style_fig(fig_bar), width="stretch")

    st.markdown("### Histograma e Boxplot por feature")
    col_a, col_b = st.columns(2)
    feature = col_a.selectbox("Feature", numeric_cols, index=0)
    log_x = col_b.checkbox("Escala logarítmica (eixo X)", value=True)

    plot_df = df[[feature, config.LABEL_COL]].copy()
    if log_x:
        # IMPORTANTE: px.histogram(log_x=True) calcula os bins em escala LINEAR
        # e só desenha o eixo em log. Para features de grande amplitude isso faz
        # um único bin ocupar ~70% da largura visível, deixando o gráfico
        # "vazio". Aqui fazemos o binning no PRÓPRIO espaço log (plotamos
        # log10 do valor deslocado), garantindo barras bem distribuídas.
        shift = plot_df[feature].min()  # desloca para permitir log com zeros/negativos
        plot_df["_v"] = np.log10(plot_df[feature] - shift + 1)
        x_label = f"log₁₀({feature})"
    else:
        plot_df["_v"] = plot_df[feature]
        x_label = feature

    h, bx = st.columns(2)
    fig_hist = px.histogram(
        plot_df, x="_v", color=config.LABEL_COL, nbins=60, barmode="overlay",
        color_discrete_map=config.CLASS_COLOR_MAP, opacity=0.65,
        labels={"_v": x_label}, title=f"Histograma · {feature}",
    )
    h.plotly_chart(style_fig(fig_hist), width="stretch")

    fig_box = px.box(
        df, x=config.LABEL_COL, y=feature, color=config.LABEL_COL,
        color_discrete_map=config.CLASS_COLOR_MAP, title=f"Boxplot · {feature}",
    )
    if log_x:
        fig_box.update_yaxes(type="log")
    bx.plotly_chart(style_fig(fig_box), width="stretch")

    st.markdown("### Matriz de correlação")
    top_n = st.slider("Nº de features na correlação", 5, min(20, len(numeric_cols)),
                      min(12, len(numeric_cols)))
    # seleciona as features com maior variância para a matriz
    var_order = df[numeric_cols].var().sort_values(ascending=False).index[:top_n]
    corr = df[var_order].corr()
    fig_corr = px.imshow(
        corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto",
        title="Correlação entre features (Pearson)",
    )
    st.plotly_chart(style_fig(fig_corr, height=520), width="stretch")

    with st.expander("Amostra dos dados e estatísticas"):
        st.dataframe(df.head(200), width="stretch")
        st.dataframe(df[numeric_cols].describe().T, width="stretch")


# ==================================================================
# Página 3: Monitoramento Temporal
# ==================================================================
def page_temporal(df: pd.DataFrame, pipe: dict):
    st.title("Monitoramento Temporal")
    st.info(
        "**Simulação operacional.** Os timestamps são SINTÉTICOS "
        "(o CICIDS-2017 não possui linha temporal contínua). Servem apenas "
        "para demonstrar análises de SOC: timelines, picos e sazonalidade."
    )

    freq = st.select_slider(
        "Granularidade da timeline", options=["30min", "1h", "3h", "6h"], value="1h"
    )

    # Timeline de incidentes (todas as classes)
    tl = feature_engineering.timeline(df, freq=freq)
    fig_tl = px.line(
        tl, x=config.TIMESTAMP_COL, y="count", color=config.LABEL_COL,
        color_discrete_map=config.CLASS_COLOR_MAP,
        title="Timeline de eventos por classe",
    )
    st.plotly_chart(style_fig(fig_tl, height=360), width="stretch")

    c1, c2 = st.columns(2)

    # Volume de tráfego por hora do dia
    by_hour = feature_engineering.traffic_by_hour(df)
    fig_h = px.bar(by_hour, x="hour", y="count",
                   title="Volume de tráfego por hora do dia",
                   color_discrete_sequence=[config.COLORS["accent"]])
    c1.plotly_chart(style_fig(fig_h), width="stretch")

    # Ataques por hora e classe
    ebh = feature_engineering.events_by_hour_class(df)
    ebh = ebh[ebh[config.LABEL_COL] != "BENIGN"]
    fig_ah = px.bar(ebh, x="hour", y="count", color=config.LABEL_COL,
                    color_discrete_map=config.CLASS_COLOR_MAP, barmode="group",
                    title="Ataques por hora do dia")
    c2.plotly_chart(style_fig(fig_ah), width="stretch")

    # Ataques por dia
    ebd = feature_engineering.events_by_day_class(df)
    fig_d = px.bar(ebd, x="day_label", y="count", color=config.LABEL_COL,
                   color_discrete_map=config.CLASS_COLOR_MAP, barmode="stack",
                   title="Eventos por dia (sintético)")
    st.plotly_chart(style_fig(fig_d), width="stretch")

    # Evolução comparada DDoS vs PortScan
    st.markdown("### Evolução temporal: DDoS vs PortScan")
    atk = tl[tl[config.LABEL_COL].isin(["DDoS", "PortScan"])]
    fig_cmp = px.area(atk, x=config.TIMESTAMP_COL, y="count", color=config.LABEL_COL,
                      color_discrete_map=config.CLASS_COLOR_MAP,
                      title="Comparação de evolução de ataques")
    st.plotly_chart(style_fig(fig_cmp), width="stretch")

    # Períodos de pico
    st.markdown("### Períodos de pico de ataques")
    peaks = feature_engineering.peak_periods(df, freq=freq, top=8)
    if not peaks.empty:
        peaks_disp = peaks.copy()
        peaks_disp[config.TIMESTAMP_COL] = peaks_disp[config.TIMESTAMP_COL].astype(str)
        st.dataframe(peaks_disp.rename(
            columns={config.TIMESTAMP_COL: "Início do período", "count": "Ataques"}),
            width="stretch")
    else:
        st.write("Nenhum ataque no período filtrado.")


# ==================================================================
# Página 4: Feature Importance
# ==================================================================
def page_importance(df: pd.DataFrame, pipe: dict):
    st.title("Importância das Variáveis")

    model_name = st.selectbox("Modelo", list(pipe["models"].keys()),
                              index=list(pipe["models"].keys()).index(
                                  pipe["best_model_name"]))
    model = pipe["models"][model_name]
    imp = evaluation.feature_importance(model, pipe["feature_names"])

    if imp.empty:
        st.warning("Modelo não expõe importância de features.")
        return

    top_n = st.slider("Top N features", 5, len(imp), min(15, len(imp)))
    top = imp.head(top_n).iloc[::-1]
    fig = px.bar(top, x="importance", y="feature", orientation="h",
                 color="importance", color_continuous_scale="Viridis",
                 title=f"Top {top_n} features: {model_name}")
    st.plotly_chart(style_fig(fig, height=500), width="stretch")

    st.markdown("#### Ranking completo")
    st.dataframe(imp, width="stretch")


# ==================================================================
# Página 5: Simulador de Predição
# ==================================================================
def page_simulator(df: pd.DataFrame, pipe: dict):
    st.title("Simulador de Predição")
    st.caption("Defina as características de um fluxo e veja a classificação "
               "do melhor modelo em tempo real.")

    model = pipe["models"][pipe["best_model_name"]]
    feature_names = pipe["feature_names"]
    stats = pipe["feature_stats"]
    le = pipe["label_encoder"]

    # seleciona as features mais importantes para sliders interativos
    imp = evaluation.feature_importance(model, feature_names)
    interactive = imp["feature"].head(8).tolist() if not imp.empty else feature_names[:8]

    st.markdown(f"Modelo ativo: **{pipe['best_model_name']}**")

    # Usa o dataset COMPLETO (não o filtrado pela sidebar) para os presets:
    # assim a mediana de uma classe nunca fica vazia/NaN mesmo que a sidebar
    # tenha filtrado aquela classe ou o período.
    full_df = pipe["df"]

    # valores padrão = mediana de cada feature
    input_vals = {f: float(stats.loc[f, "median"]) for f in feature_names}

    cols = st.columns(2)
    presets = {"Mediana global": None}
    for c in config.TARGET_CLASSES:
        presets[f"Perfil típico · {c}"] = c
    preset = cols[0].selectbox("Preset de valores", list(presets.keys()))

    # aplica preset (mediana da classe escolhida), protegendo contra NaN
    if presets[preset] is not None:
        cls_df = full_df[full_df[config.LABEL_COL] == presets[preset]]
        for f in feature_names:
            if f in cls_df.columns:
                med = cls_df[f].median()
                if pd.notna(med):
                    input_vals[f] = float(med)

    cols[1].caption("Ajuste as variáveis mais relevantes abaixo; as demais "
                    "usam a mediana/preset.")

    # Quando o preset muda, reposiciona os sliders ANTES de instanciá-los
    # (via session_state). Sem isso, o Streamlit mantém o valor anterior do
    # widget e os sliders "travam" ao trocar de perfil.
    preset_changed = st.session_state.get("_sim_preset") != preset
    st.session_state["_sim_preset"] = preset

    st.markdown("#### Variáveis interativas (mais importantes)")
    grid = st.columns(2)
    for i, f in enumerate(interactive):
        lo = float(stats.loc[f, "5%"])
        hi = float(stats.loc[f, "95%"])
        if hi <= lo:
            hi = lo + 1.0
        key = f"sim_{f}"
        default = float(np.clip(input_vals[f], lo, hi))
        if preset_changed or key not in st.session_state:
            st.session_state[key] = default
        input_vals[f] = grid[i % 2].slider(f, min_value=lo, max_value=hi, key=key)

    # monta o vetor na ordem correta
    x = pd.DataFrame([[input_vals[f] for f in feature_names]], columns=feature_names)
    if pipe["scaler"] is not None:
        x = pd.DataFrame(pipe["scaler"].transform(x), columns=feature_names)

    pred = model.predict(x)[0]
    pred_label = le.inverse_transform([int(pred)])[0]
    proba = model.predict_proba(x)[0] if hasattr(model, "predict_proba") else None

    st.markdown("### Resultado")
    color = config.CLASS_COLOR_MAP[pred_label]
    st.markdown(
        f"<div class='soc-card' style='border-left:6px solid {color}'>"
        f"<div class='soc-metric-label'>Classificação prevista</div>"
        f"<div class='soc-metric-value' style='color:{color}'>{pred_label}</div>"
        f"</div>", unsafe_allow_html=True)

    if proba is not None:
        proba_df = pd.DataFrame({
            "Classe": le.inverse_transform(np.arange(len(proba))),
            "Probabilidade": proba,
        })
        fig = px.bar(proba_df, x="Classe", y="Probabilidade", color="Classe",
                     color_discrete_map=config.CLASS_COLOR_MAP, text_auto=".2%",
                     title="Probabilidades por classe")
        fig.update_yaxes(range=[0, 1])
        st.plotly_chart(style_fig(fig), width="stretch")


# ==================================================================
# Página 6: Security Operations Center
# ==================================================================
def page_soc(df: pd.DataFrame, pipe: dict):
    st.title("Security Operations Center")
    st.caption("Painel operacional simulado · uma **janela temporal deslizante** "
               "percorre a linha do tempo sintética, como um analista monitorando "
               "o tráfego ao vivo.")

    ts = config.TIMESTAMP_COL
    df = df.sort_values(ts)

    # ---- painel de controle da janela temporal ----
    panel = st.container(border=True)
    panel.markdown(
        "<div class='soc-controls-title'>Controle da janela temporal</div>",
        unsafe_allow_html=True,
    )
    c1, c2, c3 = panel.columns([1.1, 2.4, 1.5], vertical_alignment="bottom")

    window_h = c1.selectbox("Tamanho da janela", [1, 2, 3, 6], index=1,
                            format_func=lambda h: f"{h}h")
    window = pd.Timedelta(hours=window_h)
    step = window / 2  # avanço com sobreposição → sensação de deslizamento

    # posições do cursor = instante FINAL de cada janela, ao longo do período
    t0 = df[ts].min().floor("h")
    t1 = df[ts].max().ceil("h")
    positions, cur = [], t0 + window
    while cur <= t1:
        positions.append(cur)
        cur += step
    if not positions:
        positions = [t1]

    # garante um cursor válido no estado (após troca de janela/filtros)
    if "soc_pos" not in st.session_state or st.session_state["soc_pos"] not in positions:
        st.session_state["soc_pos"] = positions[0]

    # ---- botões de navegação no tempo ----
    idx = positions.index(st.session_state["soc_pos"])
    b = c2.columns(3)
    if b[0].button("Início", width="stretch", help="Voltar ao começo"):
        idx = 0
    if b[1].button("Voltar", width="stretch", help="Janela anterior"):
        idx = max(0, idx - 1)
    if b[2].button("Avançar", width="stretch", type="primary",
                   help="Próxima janela"):
        idx = min(len(positions) - 1, idx + 1)
    st.session_state["soc_pos"] = positions[idx]

    auto = c3.toggle("Auto-play (3s)", value=False,
                     help="Avança a janela automaticamente a cada 3 segundos")

    # slider de scrubbing pela linha do tempo (sincronizado com os botões)
    cursor = panel.select_slider(
        "Posição na linha do tempo (fim da janela)",
        options=positions,
        format_func=lambda d: d.strftime("%d/%m %H:%M"),
        key="soc_pos",
    )

    w_start, w_end = cursor - window, cursor
    wdf = df[(df[ts] > w_start) & (df[ts] <= w_end)]

    # ---- KPIs da janela ----
    n_total = len(wdf)
    n_ddos = int((wdf[config.LABEL_COL] == "DDoS").sum())
    n_ps = int((wdf[config.LABEL_COL] == "PortScan").sum())
    n_atk = n_ddos + n_ps
    risk = min(100, int(n_atk / max(n_total, 1) * 160))  # índice 0-100

    if risk >= 66:
        risk_label, risk_color = "CRÍTICO", config.COLORS["crit"]
    elif risk >= 33:
        risk_label, risk_color = "ELEVADO", config.COLORS["warn"]
    else:
        risk_label, risk_color = "NORMAL", config.COLORS["ok"]

    k = st.columns(4)
    k[0].markdown(kpi_card("Eventos na janela", f"{n_total}"), unsafe_allow_html=True)
    k[1].markdown(kpi_card("DDoS", f"{n_ddos}", config.COLORS["DDoS"]),
                  unsafe_allow_html=True)
    k[2].markdown(kpi_card("PortScan", f"{n_ps}", config.COLORS["PortScan"]),
                  unsafe_allow_html=True)
    k[3].markdown(
        f"<div class='soc-card'><div class='soc-metric-label'>Nível de ameaça</div>"
        f"<div class='soc-metric-value pulse' style='color:{risk_color}'>"
        f"{risk_label}</div><div class='soc-metric-label'>índice {risk}/100</div></div>",
        unsafe_allow_html=True)

    st.caption(
        f"Janela atual: **{w_start.strftime('%d/%m %H:%M')} → "
        f"{w_end.strftime('%d/%m %H:%M')}** · posição {idx + 1}/{len(positions)}"
    )

    # ---- visão geral do período com a janela destacada ----
    tl = feature_engineering.timeline(df, freq="1h")
    fig_over = px.area(tl, x=ts, y="count", color=config.LABEL_COL,
                       color_discrete_map=config.CLASS_COLOR_MAP,
                       title="Linha do tempo completa (janela atual em destaque)")
    fig_over.add_vrect(x0=w_start, x1=w_end, fillcolor=config.COLORS["accent"],
                       opacity=0.25, line_width=0)
    st.plotly_chart(style_fig(fig_over, height=300), width="stretch")

    left, right = st.columns([1.5, 1])

    # gauge de risco
    gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=risk,
        title={"text": "Índice de risco do SOC"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": risk_color},
            "steps": [
                {"range": [0, 33], "color": "#14321f"},
                {"range": [33, 66], "color": "#3a2f12"},
                {"range": [66, 100], "color": "#3a1414"},
            ],
        },
    ))
    right.plotly_chart(style_fig(gauge, height=300), width="stretch")

    # detalhe do fluxo DENTRO da janela
    if n_total:
        rate = (wdf.set_index(ts).groupby(config.LABEL_COL)
                .resample("10min").size().reset_index(name="count"))
        fig_live = px.line(rate, x=ts, y="count", color=config.LABEL_COL,
                           color_discrete_map=config.CLASS_COLOR_MAP, markers=True,
                           title="Fluxo de eventos dentro da janela")
    else:
        fig_live = px.line(title="Fluxo de eventos dentro da janela (vazia)")
    left.plotly_chart(style_fig(fig_live, height=300), width="stretch")

    # ---- alertas ativos na janela ----
    st.markdown("### Alertas ativos na janela")
    alerts = wdf[wdf[config.LABEL_COL] != "BENIGN"].copy()
    if alerts.empty:
        st.success("Nenhum ataque nesta janela. Situação sob controle.")
    else:
        sev_map = {"DDoS": "ALTA", "PortScan": "MEDIA"}
        alerts_disp = pd.DataFrame({
            "Horário": alerts[ts].dt.strftime("%Y-%m-%d %H:%M:%S"),
            "Tipo de ataque": alerts[config.LABEL_COL],
            "Severidade": alerts[config.LABEL_COL].map(sev_map),
            "Porta destino": alerts.get("Destination Port", pd.Series(["-"] * len(alerts))),
            "Status": "Em investigação",
        }).tail(15).iloc[::-1]
        st.dataframe(alerts_disp, width="stretch", hide_index=True)

    # ---- feed de eventos da janela ----
    st.markdown("### Eventos recentes (feed da janela)")
    if n_total:
        feed = wdf.tail(15).iloc[::-1]
        feed_disp = pd.DataFrame({
            "Timestamp": feed[ts].dt.strftime("%H:%M:%S"),
            "Classe": feed[config.LABEL_COL],
            "Pacotes Fwd": feed.get("Total Fwd Packets", pd.Series(["-"] * len(feed))),
            "Flow Packets/s": (feed["Flow Packets/s"].round(1)
                               if "Flow Packets/s" in feed else "-"),
        })
        st.dataframe(feed_disp, width="stretch", hide_index=True)
    else:
        st.info("Sem eventos nesta janela temporal.")

    # ---- auto-play: avança a janela automaticamente ----
    if auto:
        import time
        time.sleep(3)
        nxt = min(len(positions) - 1, idx + 1)
        st.session_state["soc_pos"] = positions[nxt]
        if nxt != idx:  # ainda há janelas à frente
            st.rerun()


# ==================================================================
# Roteamento principal
# ==================================================================
def main():
    pipe = load_pipeline()
    f = sidebar(pipe)
    fdf = apply_filters(pipe["df"], f)

    # páginas 4 (Feature Importance) e 5 (Simulador) não dependem do recorte
    # filtrado; as demais precisam de dados após os filtros.
    if fdf.empty and not f["page"].startswith(("4", "5")):
        st.warning("Nenhum dado para os filtros selecionados. Ajuste a sidebar.")
        return

    page = f["page"]
    if page.startswith("1"):
        page_overview(fdf, pipe)
    elif page.startswith("2"):
        page_eda(fdf, pipe)
    elif page.startswith("3"):
        page_temporal(fdf, pipe)
    elif page.startswith("4"):
        page_importance(fdf, pipe)
    elif page.startswith("5"):
        page_simulator(fdf, pipe)
    elif page.startswith("6"):
        page_soc(fdf, pipe)


if __name__ == "__main__":
    main()
