# ==========================================
# IMPORTS
# ==========================================
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

st.set_page_config(layout="wide")

st.title("📊 Market Probability Scanner")

# ==========================================
# INPUTS
# ==========================================
uploaded_file = st.file_uploader("Upload TXT com tickers", type="txt")

period_days = st.number_input("Período de previsão (dias)", value=10)

tp = st.number_input("Take Profit (%)", value=5.0) / 100
sl = st.number_input("Stop Loss (%)", value=3.0) / 100

n_simulations = st.number_input("Simulações Monte Carlo", value=500)

# ==========================================
# FUNÇÕES
# ==========================================

def load_tickers(file):
    """Carrega tickers aceitando vírgula ou quebra de linha"""
    content = file.read().decode("utf-8")
    tickers = content.replace("\n", ",").split(",")
    tickers = [t.strip().upper() for t in tickers if t.strip() != ""]
    return tickers


def format_ticker(ticker):
    """Garante sufixo .SA"""
    return ticker if ticker.endswith(".SA") else ticker + ".SA"


def download_data(ticker):
    """Baixa dados históricos"""
    ticker = format_ticker(ticker)

    df = yf.download(ticker, period="2y", interval="1d", progress=False)

    if df.empty or len(df) < 100:
        raise ValueError("Dados insuficientes")

    df["Return"] = df["Close"].pct_change()

    return df.dropna()


def calculate_kpis(df):
    """Calcula indicadores básicos"""

    mean_return = df["Return"].mean()
    volatility = df["Return"].std()
    sharpe = mean_return / volatility if volatility != 0 else 0

    momentum = df["Close"].pct_change(30).iloc[-1]
    trend = df["Close"].iloc[-1] / df["Close"].iloc[-50]

    return {
        "mean_return": mean_return,
        "volatility": volatility,
        "sharpe": sharpe,
        "momentum": momentum,
        "trend": trend
    }


def monte_carlo_prob(df, period, tp, sl, sims):
    """Simulação de Monte Carlo"""

    last_price = df["Close"].iloc[-1]

    mu = df["Return"].mean()
    sigma = df["Return"].std()

    tp_hits = 0
    sl_hits = 0
    final_returns = []

    for _ in range(sims):

        price = last_price

        for _ in range(period):

            r = np.random.normal(mu, sigma)
            price *= (1 + r)

            ret = (price - last_price) / last_price

            if ret >= tp:
                tp_hits += 1
                final_returns.append(ret)
                break

            if ret <= -sl:
                sl_hits += 1
                final_returns.append(ret)
                break

        else:
            ret = (price - last_price) / last_price
            final_returns.append(ret)

    prob_tp = tp_hits / sims
    prob_sl = sl_hits / sims
    exp_return = np.mean(final_returns)

    return prob_tp, prob_sl, exp_return, final_returns


def score_asset(prob_tp, exp_return, sharpe, momentum):
    """Score ponderado"""

    return (
        prob_tp * 0.4
        + exp_return * 2
        + sharpe * 0.2
        + momentum * 0.2
    )

# ==========================================
# PIPELINE PRINCIPAL
# ==========================================

if uploaded_file:
    if st.button("🔍 Analisar"):

        tickers = load_tickers(uploaded_file)

        st.info(f"{len(tickers)} tickers carregados")

        results = []
        raw_data = {}
        simulations = {}

        progress = st.progress(0)

        for i, ticker in enumerate(tickers):

            try:
                df = download_data(ticker)

                raw_data[ticker] = df

                kpis = calculate_kpis(df)

                prob_tp, prob_sl, exp_return, final_returns = monte_carlo_prob(
                    df, period_days, tp, sl, n_simulations
                )

                simulations[ticker] = final_returns

                score = score_asset(
                    prob_tp,
                    exp_return,
                    kpis["sharpe"],
                    kpis["momentum"]
                )

                results.append({
                    "Ticker": ticker,
                    "Score": score,
                    "Prob_TP": prob_tp,
                    "Prob_SL": prob_sl,
                    "Expected_Return": exp_return,
                    **kpis
                })

            except Exception as e:
                st.warning(f"{ticker}: {e}")

            progress.progress((i + 1) / len(tickers))

        # ==========================================
        # RESULTADOS
        # ==========================================

        results_df = pd.DataFrame(results)

        if results_df.empty:
            st.error("❌ Nenhum ticker válido foi processado.")
            st.stop()

        results_df = results_df.sort_values("Score", ascending=False)

        st.subheader("🏆 Ranking Geral")
        st.dataframe(results_df)

        top5 = results_df.head(5)

        st.subheader("🔥 Top 5 Oportunidades")
        st.dataframe(top5)

        # ==========================================
        # VISUALIZAÇÃO
        # ==========================================

        for ticker in top5["Ticker"]:

            st.header(f"📌 {ticker}")

            df = raw_data[ticker]
            kpis = results_df[results_df["Ticker"] == ticker].iloc[0]

            st.write("KPIs")
            st.write(kpis)

            # Regressão linear
            x = np.arange(len(df)).reshape(-1, 1)
            y = df["Close"].values

            model = LinearRegression()
            model.fit(x, y)
            reg = model.predict(x)

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=df.index,
                y=df["Close"],
                name="Preço"
            ))

            fig.add_trace(go.Scatter(
                x=df.index,
                y=reg,
                name="Tendência (Regressão)"
            ))

            st.plotly_chart(fig, use_container_width=True)

            # Histograma Monte Carlo
            st.subheader("Distribuição de Retornos (Monte Carlo)")

            hist = go.Figure()

            hist.add_trace(go.Histogram(
                x=simulations[ticker],
                nbinsx=50
            ))

            st.plotly_chart(hist, use_container_width=True)