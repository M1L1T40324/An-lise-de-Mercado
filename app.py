import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go

from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from arch import arch_model

# =========================
# CONFIG
# =========================

st.set_page_config(layout="wide")
st.title("📊 Quant Trading Pipeline Completo")

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
        df = yf.download(t, progress=False)

        if not df.empty:
            # 🔥 CORREÇÃO AQUI
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df["Ticker"] = t
            dados[t] = df

    return dados

def features(df):

    # garantir colunas simples
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["volatility_10"] = df["log_return"].rolling(10).std()
    df["momentum_5"] = df["Close"] - df["Close"].shift(5)
    df["lag_1"] = df["log_return"].shift(1)

    # 🔥 garantir Series
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
    if dist == "normal":
        d = "normal"
    elif dist == "t":
        d = "t"
    else:
        d = "skewt"

    model = arch_model(data*100, vol="Garch", p=1, q=1, dist=d)
    res = model.fit(disp="off")

    return res

def monte_carlo(mu, sigma, n=100, T=50):
    paths = []

    for _ in range(n):
        S = 1
        serie = []

        for _ in range(T):
            z = np.random.normal()
            S *= np.exp((mu - 0.5*sigma**2) + sigma*z)
            serie.append(S)

        paths.append(serie)

    return np.array(paths)

def estrategia_ml(df):
    df["target"] = (df["log_return"].shift(-1) > 0).astype(int)

    features_cols = ["log_return","lag_1","momentum_5","volatility_10"]

    df_model = df[features_cols + ["target"]].dropna()

    split = int(len(df_model)*0.8)

    train = df_model.iloc[:split]
    test = df_model.iloc[split:]

    model = RandomForestClassifier(n_estimators=100)
    model.fit(train[features_cols], train["target"])

    proba = model.predict_proba(test[features_cols])[:,1]

    return test, proba

def backtest(test, proba, threshold=0.55):
    pos = np.zeros(len(proba))
    pos[proba > threshold] = 1
    pos[proba < (1-threshold)] = -1

    returns = test["log_return"].shift(-1)
    strat = pos * returns
    strat = strat.dropna()

    sharpe = strat.mean()/strat.std()*np.sqrt(252)

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
        st.write("Distribuição escolhida:", dist)

        # -----------------
        # GARCH
        # -----------------

        garch = ajustar_garch(data, dist)
        sigma = garch.conditional_volatility / 100

        df["cond_vol"] = sigma

        # -----------------
        # ML
        # -----------------

        test, proba = estrategia_ml(df)

        strat, sharpe = backtest(test, proba)

        st.write("Sharpe Ratio:", sharpe)

        # -----------------
        # Monte Carlo
        # -----------------

        mu = data.mean()
        sigma_val = data.std()

        paths = monte_carlo(mu, sigma_val)

        fig = go.Figure()
        for i in range(10):
            fig.add_trace(go.Scatter(y=paths[i], mode='lines'))

        st.plotly_chart(fig)

        # -----------------
        # Probabilidades de retorno
        # -----------------

        targets = [0.01, 0.02, 0.05]  # 1%, 2%, 5%

        probs = {}

        for t in targets:
            probs[f"{int(t*100)}%"] = np.mean(data > t)

        st.write("Probabilidades históricas de atingir retorno diário:")
        st.write(probs)

        # -----------------
        # Curva da estratégia
        # -----------------

        cum = (1+strat).cumprod()

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(y=cum, mode='lines', name="Estratégia"))

        st.plotly_chart(fig2)
