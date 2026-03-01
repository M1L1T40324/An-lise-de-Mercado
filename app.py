# ============================
# IMPORTS
# ============================

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

np.random.seed(42)

# ============================
# DATA LAYER
# ============================

def load_data(ticker, period="5y"):
    data = yf.download(ticker, period=period, auto_adjust=True)
    return data["Close"]

def estimate_params(close, window=252):
    log_ret = np.log(close / close.shift(1)).dropna()
    mu = log_ret.rolling(window).mean().dropna()
    sigma = log_ret.rolling(window).std().dropna()
    return mu.iloc[-1], sigma.iloc[-1]

# ============================
# PROBABILIDADE ANALÍTICA
# ============================

def hitting_probability(mu, sigma, tp, sl):
    if sigma <= 0:
        return 0.0, 0.0

    if abs(mu) < 1e-8:
        p_tp = sl / (tp + sl)
        return p_tp, 1 - p_tp

    a = 2 * mu / (sigma**2)

    numerator = 1 - np.exp(-a * sl)
    denominator = np.exp(a * tp) - np.exp(-a * sl)

    if denominator == 0:
        return 0.0, 0.0

    p_tp = numerator / denominator
    p_tp = np.clip(p_tp, 0, 1)

    return p_tp, 1 - p_tp

# ============================
# SIMULAÇÃO MONTE CARLO
# ============================

def simulate_paths(mu, sigma, horizon, n_sim=5000):
    eps = np.random.randn(n_sim, horizon)
    returns = mu + sigma * eps
    paths = returns.cumsum(axis=1)
    return paths

def simulate_trade_distribution(mu, sigma, tp, sl, horizon, n_sim=5000):

    paths = simulate_paths(mu, sigma, horizon, n_sim)
    pnl = []

    for path in paths:
        hit_tp = np.where(path >= tp)[0]
        hit_sl = np.where(path <= -sl)[0]

        tp_time = hit_tp[0] if len(hit_tp) else np.inf
        sl_time = hit_sl[0] if len(hit_sl) else np.inf

        if tp_time < sl_time:
            pnl.append(tp)
        elif sl_time < tp_time:
            pnl.append(-sl)
        else:
            pnl.append(path[-1])

    return np.array(pnl)

# ============================
# MÉTRICAS DE RISCO
# ============================

def risk_metrics(pnl):

    EV = pnl.mean()
    std = pnl.std()
    sharpe = EV / std if std > 0 else 0

    cum = np.cumsum(pnl)
    drawdown = cum - np.maximum.accumulate(cum)
    max_dd = drawdown.min()

    cvar = pnl[pnl <= np.quantile(pnl, 0.05)].mean()

    return {
        "EV": EV,
        "Sharpe": sharpe,
        "MaxDD": max_dd,
        "CVaR_5%": cvar
    }

# ============================
# KELLY CONTÍNUO REAL
# ============================

def kelly_continuous(pnl):

    def objective(f):
        if f <= 0:
            return -np.inf
        return np.mean(np.log(1 + f * pnl))

    f_grid = np.linspace(0.0, 0.5, 200)
    values = [objective(f) for f in f_grid]

    idx = np.argmax(values)
    return f_grid[idx]

# ============================
# OTIMIZAÇÃO TP/SL
# ============================

def optimize_tp_sl(mu, sigma, horizon):

    tp_range = np.linspace(0.01, 0.30, 15)
    sl_range = np.linspace(0.01, 0.30, 15)

    results = []

    for tp in tp_range:
        for sl in sl_range:

            pnl = simulate_trade_distribution(mu, sigma, tp, sl, horizon)

            metrics = risk_metrics(pnl)
            kelly = kelly_continuous(pnl)

            results.append({
                "TP": tp,
                "SL": sl,
                "EV": metrics["EV"],
                "Sharpe": metrics["Sharpe"],
                "MaxDD": metrics["MaxDD"],
                "CVaR": metrics["CVaR_5%"],
                "Kelly": min(kelly, 0.25)
            })

    df = pd.DataFrame(results)
    best = df.loc[df["EV"].idxmax()]

    return best, df

# ============================
# STREAMLIT UI
# ============================

st.title("⚔️ Motor Quantitativo de Swing Trade")

ticker = st.text_input("Ticker", "PETR4.SA")
horizon = st.slider("Horizonte (dias)", 5, 60, 15)

if st.button("Analisar"):

    close = load_data(ticker)
    S0 = close.iloc[-1]

    mu, sigma = estimate_params(close)

    retorno_esperado = mu * horizon
    preco_esperado = S0 * np.exp(retorno_esperado)

    prob_pos = 1 - (np.exp(-2 * mu * retorno_esperado / (sigma**2)) 
                    if sigma > 0 else 0)

    best, df_all = optimize_tp_sl(mu, sigma, horizon)

    tp = best["TP"]
    sl = best["SL"]

    p_tp, p_sl = hitting_probability(mu, sigma, tp, sl)

    pnl = simulate_trade_distribution(mu, sigma, tp, sl, horizon)

    metrics = risk_metrics(pnl)
    kelly = kelly_continuous(pnl)

    st.subheader(f"{ticker}")
    st.write(f"Preço Atual: {S0:.2f}")

    st.write(f"Retorno Esperado ({horizon} dias): {retorno_esperado*100:.2f}%")
    st.write(f"Preço Esperado: {preco_esperado:.2f}")

    st.write(f"Probabilidade retorno positivo: {(pnl > 0).mean()*100:.2f}%")

    st.subheader("Probabilidades Condicionais")
    st.write(f"TP antes do SL: {p_tp*100:.2f}%")
    st.write(f"SL antes do TP: {p_sl*100:.2f}%")

    st.subheader("TP/SL Ideais (Max EV)")
    st.write(f"TP Ideal: {tp*100:.2f}%")
    st.write(f"SL Ideal: {sl*100:.2f}%")
    st.write(f"Valor Esperado Máximo: {best['EV']:.4f}")

    st.subheader("Métricas de Risco")
    st.write(f"Sharpe: {metrics['Sharpe']:.4f}")
    st.write(f"Max Drawdown: {metrics['MaxDD']:.4f}")
    st.write(f"CVaR 5%: {metrics['CVaR_5%']:.4f}")

    st.subheader("Alocação Ótima (Kelly)")
    st.write(f"Fração Ótima: {kelly:.2f}")

    st.subheader("Distribuição Simulada")
    fig, ax = plt.subplots()
    ax.hist(pnl, bins=50)
    st.pyplot(fig)
