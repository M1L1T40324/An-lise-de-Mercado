import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def plot_price_regression(data, ticker):

    df = data.copy()

    df = df.dropna()
    df["t"] = np.arange(len(df))

    # regressão linear
    coef = np.polyfit(df["t"], df["Close"], 1)
    trend = np.poly1d(coef)

    df["trend"] = trend(df["t"])

    fig = go.Figure()

    # preço
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Close"],
            mode="lines",
            name="Preço",
            line=dict(width=2)
        )
    )

    # regressão
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["trend"],
            mode="lines",
            name="Regressão Linear",
            line=dict(width=3, dash="dash")
        )
    )

    fig.update_layout(
        title=f"{ticker} — Cotação com Tendência",
        xaxis_title="Data",
        yaxis_title="Preço",
        template="plotly_dark",
        hovermode="x unified"
    )

    return fig

np.random.seed(42)

# ============================
# DATA LAYER (ROBUSTO)
# ============================

def load_data(ticker, period="5y"):

    try:
        data = yf.download(
            ticker,
            period=period,
            auto_adjust=True,
            progress=False
        )

        if data.empty:
            return None

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        if "Close" not in data.columns:
            return None

        close = data["Close"].dropna()

        if len(close) < 300:
            return None

        return close.astype(float)

    except:
        return None


# ============================
# PARAMETER ESTIMATION
# ============================

def estimate_params(close, window=252):

    log_ret = np.log(close / close.shift(1)).dropna()

    if len(log_ret) < window:
        raise ValueError("Dados insuficientes.")

    mu_series = log_ret.rolling(window).mean()
    sigma_series = log_ret.rolling(window).std()

    mu = float(mu_series.iloc[-1])
    sigma = float(sigma_series.iloc[-1])

    return mu, sigma


# ============================
# HITTING PROBABILITY
# ============================

def hitting_probability(mu, sigma, tp, sl):

    if sigma <= 0:
        return 0.0, 0.0

    if abs(mu) < 1e-8:
        p_tp = sl / (tp + sl)
        return p_tp, 1 - p_tp

    a = 2 * mu / (sigma ** 2)

    num = 1 - np.exp(-a * sl)
    den = np.exp(a * tp) - np.exp(-a * sl)

    if den == 0:
        return 0.5, 0.5

    p_tp = num / den
    p_tp = float(np.clip(p_tp, 0, 1))

    return p_tp, 1 - p_tp


# ============================
# MONTE CARLO ENGINE
# ============================

def simulate_paths(mu, sigma, horizon, n_sim=5000):

    eps = np.random.randn(n_sim, horizon)
    returns = mu + sigma * eps
    paths = np.cumsum(returns, axis=1)

    return paths


def simulate_trade_distribution(mu, sigma, tp, sl, horizon, n_sim=5000):

    paths = simulate_paths(mu, sigma, horizon, n_sim)

    pnl = []

    for path in paths:

        tp_hit = np.where(path >= tp)[0]
        sl_hit = np.where(path <= -sl)[0]

        tp_time = tp_hit[0] if len(tp_hit) else np.inf
        sl_time = sl_hit[0] if len(sl_hit) else np.inf

        if tp_time < sl_time:
            pnl.append(tp)

        elif sl_time < tp_time:
            pnl.append(-sl)

        else:
            pnl.append(path[-1])

    return np.array(pnl)


# ============================
# RISK METRICS
# ============================

def risk_metrics(pnl):

    EV = pnl.mean()

    std = pnl.std()
    sharpe = EV / std if std > 0 else 0

    cum = np.cumsum(pnl)

    drawdown = cum - np.maximum.accumulate(cum)
    max_dd = drawdown.min()

    cvar = pnl[pnl <= np.quantile(pnl, 0.05)].mean()

    return EV, sharpe, max_dd, cvar


# ============================
# KELLY CONTINUOUS
# ============================

def kelly_continuous(pnl):

    f_grid = np.linspace(0, 0.5, 200)

    best_f = 0
    best_val = -np.inf

    for f in f_grid:

        val = np.mean(np.log(1 + f * pnl))

        if val > best_val:
            best_val = val
            best_f = f

    return best_f


# ============================
# TP SL OPTIMIZATION
# ============================

def optimize_tp_sl(mu, sigma, horizon):

    tp_range = np.linspace(0.01, 0.20, 12)
    sl_range = np.linspace(0.01, 0.20, 12)

    rows = []

    for tp in tp_range:
        for sl in sl_range:

            pnl = simulate_trade_distribution(
                mu, sigma, tp, sl, horizon
            )

            EV, sharpe, max_dd, cvar = risk_metrics(pnl)
            kelly = kelly_continuous(pnl)

            rows.append({
                "TP": tp,
                "SL": sl,
                "EV": EV,
                "Sharpe": sharpe,
                "MaxDD": max_dd,
                "CVaR": cvar,
                "Kelly": min(kelly, 0.25)
            })

    df = pd.DataFrame(rows)
    fig = plot_price_regression(data, ticker)
    st.plotly_chart(fig, use_container_width=True)

    best = df.sort_values("EV", ascending=False).iloc[0]

    return best, df


# ============================
# STREAMLIT UI
# ============================

st.title("⚔️ Motor Quantitativo de Swing Trade")

ticker = st.text_input("Ticker", "PETR4.SA")
horizon = st.slider("Horizonte (dias)", 5, 60, 15)

if st.button("Analisar"):

    close = load_data(ticker)

    if close is None:
        st.error("Erro ao carregar dados do ticker.")
        st.stop()

    try:
        mu, sigma = estimate_params(close)
    except:
        st.error("Dados insuficientes para estimar parâmetros.")
        st.stop()

    S0 = float(close.iloc[-1])

    retorno_esperado = mu * horizon
    preco_esperado = S0 * np.exp(retorno_esperado)

    best, df_all = optimize_tp_sl(mu, sigma, horizon)

    tp = float(best["TP"])
    sl = float(best["SL"])

    p_tp, p_sl = hitting_probability(mu, sigma, tp, sl)

    pnl = simulate_trade_distribution(mu, sigma, tp, sl, horizon)

    EV, sharpe, max_dd, cvar = risk_metrics(pnl)
    kelly = kelly_continuous(pnl)

    prob_pos = float((pnl > 0).mean())

    st.subheader(ticker)

    st.write(f"Preço Atual: {S0:.2f}")
    st.write(f"Retorno Esperado ({horizon} dias): {retorno_esperado*100:.2f}%")
    st.write(f"Preço Esperado: {preco_esperado:.2f}")

    st.write(f"Probabilidade retorno positivo: {prob_pos*100:.2f}%")

    st.subheader("Probabilidades Condicionais")

    st.write(f"TP antes do SL: {p_tp*100:.2f}%")
    st.write(f"SL antes do TP: {p_sl*100:.2f}%")

    st.subheader("TP/SL Ideais")

    st.write(f"TP Ideal: {tp*100:.2f}%")
    st.write(f"SL Ideal: {sl*100:.2f}%")
    st.write(f"Valor Esperado: {best['EV']:.4f}")

    st.subheader("Métricas de Risco")

    st.write(f"Sharpe: {sharpe:.4f}")
    st.write(f"Max Drawdown: {max_dd:.4f}")
    st.write(f"CVaR 5%: {cvar:.4f}")

    st.subheader("Kelly Ótimo")

    st.write(f"Fração: {kelly:.2f}")

    st.subheader("Distribuição de PnL")

    fig, ax = plt.subplots()
    ax.hist(pnl, bins=40)
    st.pyplot(fig)
