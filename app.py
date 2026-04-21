import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go

from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from arch import arch_model

# =========================
# CONFIG
# =========================

st.set_page_config(layout="wide")
st.title("📊 Quant Trading Pipeline (Refatorado)")

# =========================
# INPUT
# =========================

tickers_input = st.text_input("Tickers (ex: PETR4.SA, VALE3.SA)")
rodar = st.button("Executar Pipeline")

# =========================
# FUNÇÕES
# =========================

def baixar_dados(tickers):
    dados = {}

    for t in tickers:
        df = yf.download(t, period="2y", progress=False)

        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df["Ticker"] = t
            dados[t] = df

    return dados


def features(df):
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["volatility_10"] = df["log_return"].rolling(10).std()
    df["momentum_5"] = df["Close"] - df["Close"].shift(5)
    df["lag_1"] = df["log_return"].shift(1)

    vol = df["Volume"]
    if isinstance(vol, pd.DataFrame):
        vol = vol.iloc[:, 0]

    df["volume_return"] = vol * df["log_return"]

    return df


def escolher_distribuicao(data):
    s = stats.skew(data)
    k = stats.kurtosis(data, fisher=False)

    if k < 3.5:
        return "normal"
    elif abs(s) < 0.2:
        return "t"
    else:
        return "skewt"


def ajustar_garch(data, dist):
    model = arch_model(data * 100, vol="Garch", p=1, q=1, dist=dist)
    return model.fit(disp="off")


# 🔥 CORRIGIDO (Monte Carlo com dt)
def monte_carlo(mu, sigma, n=100, T=50, dt=1/252):
    paths = []

    for _ in range(n):
        S = 1
        serie = []

        for _ in range(T):
            z = np.random.normal()
            S *= np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
            serie.append(S)

        paths.append(serie)

    return np.array(paths)


def estrategia_ml(df):
    df["target"] = (df["log_return"].shift(-1) > 0).astype(int)

    features_cols = ["log_return", "lag_1", "momentum_5", "volatility_10"]

    df_model = df[features_cols + ["target"]].dropna()

    split = int(len(df_model) * 0.8)

    train = df_model.iloc[:split]
    test = df_model.iloc[split:]

    model = RandomForestClassifier(n_estimators=100)
    model.fit(train[features_cols], train["target"])

    proba = model.predict_proba(test[features_cols])[:, 1]

    return test.reset_index(drop=True), proba


# 🔥 CORRIGIDO (alinhamento correto)
def backtest(test, proba, threshold=0.55):
    pos = np.zeros(len(proba))

    pos[proba > threshold] = 1
    pos[proba < (1 - threshold)] = -1

    returns = test["log_return"].values

    # alinhar (usar retorno seguinte)
    returns = returns[1:]
    pos = pos[:-1]

    strat = pos * returns
    strat = pd.Series(strat).fillna(0)

    sharpe = strat.mean() / strat.std() * np.sqrt(252)

    return strat, sharpe


# =========================
# EXECUÇÃO
# =========================

if rodar:

    tickers = [t.strip().upper() for t in tickers_input.split(",")]

    dados = baixar_dados(tickers)

    for nome, df in dados.items():

        st.header(f"📌 {nome}")

        df = features(df)
        data = df["log_return"].dropna()

        # -----------------
        # Distribuição
        # -----------------

        dist = escolher_distribuicao(data)
        st.write("Distribuição:", dist)

        # -----------------
        # GARCH
        # -----------------

        try:
            garch = ajustar_garch(data, dist)
            sigma = garch.conditional_volatility / 100
            df["cond_vol"] = sigma
        except:
            st.warning("GARCH falhou")
            continue

        # -----------------
        # ML
        # -----------------

        test, proba = estrategia_ml(df)

        if len(proba) == 0:
            st.warning("Poucos dados para ML")
            continue

        strat, sharpe = backtest(test, proba)

        st.write("Sharpe:", round(sharpe, 3))

        # -----------------
        # Monte Carlo
        # -----------------

        mu = data.mean()
        sigma_val = data.std()

        paths = monte_carlo(mu, sigma_val)

        x = np.arange(paths.shape[1])

        fig = go.Figure()
        for i in range(min(10, len(paths))):
            fig.add_trace(go.Scatter(x=x, y=paths[i], mode="lines"))

        fig.update_layout(
            title="Monte Carlo",
            xaxis_title="Dias",
            yaxis_title="Preço"
        )

        st.plotly_chart(fig)

        # -----------------
        # Curva estratégia
        # -----------------

        cum = (1 + strat).cumprod()
        cum = cum / cum.iloc[0]

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(y=cum, mode="lines", name="Estratégia"))

        fig2.update_layout(title="Equity Curve")

        st.plotly_chart(fig2)
