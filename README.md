# Network Attack Detection Analysis

Aplicação web interativa de **Data Science / Machine Learning** para
**detecção de ataques de rede** sobre o dataset **CICIDS-2017**, simulando um
**Centro de Operações de Segurança (SOC)**.

O projeto cobre **todo o pipeline de Data Science**: carga, limpeza,
engenharia de atributos, treinamento, avaliação e *deploy* analítico, entregue
como um **dashboard Streamlit** com 7 páginas e gráficos Plotly.

> Classes detectadas: **BENIGN**, **DDoS**, **PortScan**.

---

## Funcionalidades

| Página | Conteúdo |
|--------|----------|
| 1 · **Overview** | KPIs: total de registros, ataques, % de ataques, melhor modelo |
| 2 · **Exploração dos Dados** | Distribuição de classes, histogramas, boxplots, correlação |
| 3 · **Monitoramento Temporal** | Timeline, ataques por hora/dia, picos, DDoS × PortScan |
| 4 · **Machine Learning** | Métricas, comparação de modelos, matrizes de confusão |
| 5 · **Feature Importance** | Ranking das variáveis mais relevantes |
| 6 · **Simulador de Predição** | Usuário ajusta features → modelo prevê a classe |
| 7 · **Security Operations Center** | Feed de eventos, índice de risco, alertas, gráficos "ao vivo" |

---

## Arquitetura

```
ciclopus_comba/
├── config.py                 # caminhos, constantes, hiperparâmetros, tema
├── requirements.txt
├── README.md
│
├── data/
│   ├── raw/                  # >>> coloque aqui os CSVs do CICIDS-2017 <<<
│   └── processed/            # dataset tratado (.parquet) - gerado
├── models/                   # modelos + métricas serializados - gerado
│
├── src/                      # núcleo do pipeline de Data Science
│   ├── data_loader.py        # 1. carga (CSV real OU fallback sintético)
│   ├── preprocessing.py      # 2. limpeza, NaN/inf, encoding, scaling
│   ├── feature_engineering.py# 3. TIMESTAMPS SINTÉTICOS + agregações temporais
│   ├── training.py           # 4. treina Decision Tree / Random Forest / XGBoost
│   ├── evaluation.py         # 5. métricas, matriz de confusão, seleção do melhor
│   └── pipeline.py           # orquestrador cacheado (usado pelo dashboard)
│
├── scripts/
│   └── train.py              # roda o pipeline completo via linha de comando
│
└── app/
    └── dashboard.py          # 6. aplicação Streamlit (7 páginas, Plotly)
```

**Fluxo de dados:**
`data_loader` → `feature_engineering` (timestamp) → `preprocessing` →
`training` → `evaluation` → artefatos em `data/processed` e `models/` →
`dashboard` carrega os artefatos e renderiza as páginas.

O `dashboard` reaproveita os artefatos em disco; se não existirem, ele executa
o pipeline completo automaticamente na primeira abertura (com cache).

---

## Sobre os timestamps (LEIA)

> **O CICIDS-2017 não possui uma linha temporal contínua adequada para
> monitoramento.** A coluna `Timestamp` é **100% SINTÉTICA** e existe apenas
> para **simulação operacional de SOC** (timelines, picos, sazonalidade).
> **Não representa o tempo real dos fluxos capturados.**

Como é gerada (`src/feature_engineering.py`):

* tráfego distribuído ao longo de **5 dias** consecutivos;
* tráfego **BENIGN** segue um **padrão diurno** (picos em horário comercial);
* **DDoS** e **PortScan** são concentrados em **janelas de incidente**
  (≈85% dos eventos) gerando **picos de atividade** plausíveis, com ~15% de
  ruído de fundo;
* as **proporções entre classes são preservadas** (apenas o tempo é atribuído);
* parâmetros configuráveis em `config.SYNTHETIC_TIME`.

---

## Modelos

Treinados e comparados: **Decision Tree**, **Random Forest**, **XGBoost**.

Métricas: **Accuracy, Precision, Recall, F1 (macro/weighted), Confusion Matrix**.
O **melhor modelo é selecionado automaticamente** pela métrica
`config.SELECTION_METRIC` (padrão: `f1_macro`).

---

## Como executar

### 1. Pré-requisitos
* Python 3.10+

### 2. Instalar dependências
```bash
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. (Opcional) Obter o dataset real
Baixe o **CICIDS-2017** no Kaggle (versão *MachineLearningCVE*) e copie os CSVs
para `data/raw/`. Arquivos relevantes para este projeto:

* `Monday-WorkingHours.pcap_ISCX.csv` (tráfego BENIGN)
* `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`
* `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv`

> **Sem o dataset?** Sem problema. A aplicação **gera dados sintéticos
> realistas** automaticamente e roda 100% do mesmo jeito (um aviso é exibido).

### 4. (Opcional) Treinar via linha de comando
```bash
python -m scripts.train             # usa dados reais se existirem
python -m scripts.train --synthetic # força dados sintéticos
```

### 5. Rodar o dashboard
```bash
streamlit run app/dashboard.py
```
Acesse **http://localhost:8501**. Na primeira execução, se não houver artefatos
treinados, o app roda o pipeline automaticamente.

---

## Stack

`Python` · `Pandas` · `NumPy` · `Scikit-Learn` · `XGBoost` · `Plotly` ·
`Streamlit` · `joblib`

---

## Notas acadêmicas

Este projeto foi estruturado como **projeto final de Data Science**,
demonstrando o pipeline completo de forma modular e reprodutível, com entrega
em uma **aplicação interativa** (e não apenas um notebook). As análises
temporais devem ser interpretadas como **simulação operacional**, dado o caráter
sintético dos timestamps.
