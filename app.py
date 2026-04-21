import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from arch import arch_model

# =========================
# CONFIG
# =========================

st.set_page_config(layout="wide")
st.title("📊 Quant Trading Pipeline PRO + Scanner de Oportunidades")

# =========================
# PARÂMETROS
# =========================

TP = 0.03      # 3% take profit
SL = -0.02     # -2% stop loss
N_DIAS = 10    # horizonte
N_SIM = 300    # simulações monte carlo

# =========================
# INPUT
# =========================

tickers_input = st.text_input("Tickers (ex: PETR4.SA, VALE3.SA)")
rodar = st.button("Executar")

# =========================
# FUNÇÕES
# =========================

def baixar_dados(tickers):
    dados = {}
    for t in tickers:
        df = yf.download(t, progress=False)
        if not df.empty:
            df["Ticker"] = t
            dados[t] = df
    return dados

def features(df):
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["volatility_10"] = df["log_return"].rolling(10).std()
    df["momentum_5"] = df["Close"] - df["Close"].shift(5)
    df["lag_1"] = df["log_return"].shift(1)
    df["lag_2"] = df["log_return"].shift(2)
    return df

def escolher_dist(data):
    s = stats.skew(data)
    k = stats.kurtosis(data, fisher=False)
    if k < 3.5:
        return "normal"
    elif abs(s) < 0.2:
        return "t"
    else:
        return "skewt"

def ajustar_garch(data, dist):
    model = arch_model(data*100, vol="Garch", p=1, q=1, dist=dist)
    res = model.fit(disp="off")
    return res

def walk_forward_ml(df, features_cols):

    df["target"] = (df["log_return"].shift(-1) > 0).astype(int)
    df = df.dropna()

    preds = []

    for i in range(100, len(df)-1):
        train = df.iloc[:i]
        test = df.iloc[i:i+1]

        model = RandomForestClassifier(n_estimators=100)
        model.fit(train[features_cols], train["target"])

        p = model.predict_proba(test[features_cols])[:,1][0]
        preds.append(p)

    return np.array(preds), df.iloc[100:len(df)-1]

def calcular_score(proba, vol):
    vol = max(vol, 1e-6)
    return proba / vol

def monte_carlo_garch_tp_sl(res):

    params = res.params
    omega = params['omega']
    alpha = params['alpha[1]']
    beta = params['beta[1]']

    tp_hit = 0
    sl_hit = 0

    for _ in range(N_SIM):
        sigma2 = omega / (1 - alpha - beta)
        S = 1

        for _ in range(N_DIAS):
            z = np.random.normal()
            sigma2 = omega + alpha*(z**2) + beta*sigma2
            sigma = np.sqrt(sigma2)

            S *= np.exp(sigma*z)

            if S - 1 >= TP:
                tp_hit += 1
                break

            if S - 1 <= SL:
                sl_hit += 1
                break

    return tp_hit / N_SIM, sl_hit / N_SIM

def plot_regressao(df, nome):

    y = df["Close"].values.reshape(-1,1)
    X = np.arange(len(y)).reshape(-1,1)

    model = LinearRegression()
    model.fit(X, y)
    trend = model.predict(X)

    plt.figure(figsize=(10,4))
    plt.plot(y, label="Preço")
    plt.plot(trend, label="Regressão", linestyle="--")
    plt.title(nome)
    plt.legend()
    st.pyplot(plt)

def backtest(proba, df):

    returns = df["log_return"].values
    positions = 2*proba - 1

    strat = positions * returns
    strat = pd.Series(strat)

    sharpe = strat.mean()/strat.std()*np.sqrt(252)
    cum = (1+strat).cumprod()

    return sharpe, cum

# =========================
# EXECUÇÃO
# =========================

if rodar:

    tickers = [t.strip().upper() for t in tickers_input.split(",")]

    dados = baixar_dados(tickers)

    resultados = []

    for nome, df in dados.items():

        df = features(df)

        data = df["log_return"].dropna()

        dist = escolher_dist(data)
        garch = ajustar_garch(data, dist)

        sigma = garch.conditional_volatility / 100
        df["cond_vol"] = sigma

        features_cols = ["log_return","lag_1","lag_2","momentum_5","volatility_10","cond_vol"]

        proba, df_ml = walk_forward_ml(df, features_cols)

        if len(proba) == 0:
            continue

        score = calcular_score(proba[-1], df_ml["cond_vol"].values[-1])

        resultados.append({
            "ticker": nome,
            "score": score,
            "df": df,
            "garch": garch,
            "proba": proba,
            "df_ml": df_ml
        })

    # =========================
    # TOP 5
    # =========================

    top5 = sorted(resultados, key=lambda x: x["score"], reverse=True)[:5]

    st.header("🏆 Top 5 ativos por score")

    for r in top5:
        st.write(f"{r['ticker']} | Score: {r['score']:.4f}")

    # =========================
    # DETALHES DOS TOP 5
    # =========================

    for r in top5:

        st.subheader(f"📊 {r['ticker']}")

        df = r["df"].dropna()

        # Plot regressão
        plot_regressao(df, r["ticker"])

        # Probabilidades TP/SL
        p_tp, p_sl = monte_carlo_garch_tp_sl(r["garch"])

        st.write(f"Probabilidade TP ({TP*100:.1f}%): {p_tp:.2%}")
        st.write(f"Probabilidade SL ({SL*100:.1f}%): {p_sl:.2%}")

        # Backtest
        sharpe, cum = backtest(r["proba"], r["df_ml"])

        st.write("Sharpe:", round(sharpe, 2))

        fig = go.Figure()
        fig.add_trace(go.Scatter(y=cum, name="Estratégia"))
        st.plotly_chart(fig)
