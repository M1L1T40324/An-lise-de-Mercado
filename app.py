import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.stats import t

# ============================
# DATA LAYER
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

        close = data["Close"].dropna()

        if len(close) < 300:
            return None

        return close.astype(float)

    except:
        return None


# ============================
# RETURNS
# ============================

def compute_log_returns(close):

    log_ret = np.log(close / close.shift(1)).dropna()

    return log_ret


# ============================
# DRIFT (SHRINKAGE)
# ============================

def estimate_drift(log_ret):

    mu_hist = log_ret.mean()

    shrink = 0.25

    mu = shrink * mu_hist

    return float(mu)


# ============================
# VOLATILITY EWMA
# ============================

def estimate_volatility_ewma(log_ret, lam=0.94):

    var = log_ret.var()

    for r in log_ret:
        var = lam * var + (1 - lam) * (r**2)

    return float(np.sqrt(var))


# ============================
# STUDENT T RETURNS
# ============================

def simulate_returns(mu, sigma, horizon, n_sim, df=5):

    shocks = t.rvs(df, size=(n_sim, horizon))

    scale = sigma / np.sqrt(df/(df-2))

    returns = mu + scale * shocks

    return returns


# ============================
# PRICE PATHS
# ============================

def simulate_paths(mu, sigma, horizon, n_sim=5000):

    returns = simulate_returns(mu, sigma, horizon, n_sim)

    log_price = np.cumsum(returns, axis=1)

    return log_price


# ============================
# TRADE DISTRIBUTION
# ============================

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

    cvar = pnl[pnl <= np.quantile(pnl,0.05)].mean()

    skew = pd.Series(pnl).skew()

    return EV, sharpe, max_dd, cvar, skew


# ============================
# GROWTH RATE
# ============================

def growth_rate(pnl):

    pnl = np.clip(pnl, -0.99, None)

    return np.mean(np.log(1+pnl))


# ============================
# OBJECTIVE FUNCTION
# ============================

def objective(pnl):

    EV, sharpe, max_dd, cvar, skew = risk_metrics(pnl)

    growth = growth_rate(pnl)

    score = growth - 0.5*abs(cvar) - 0.2*abs(max_dd)

    score += 0.3 * skew

    return score


# ============================
# KELLY
# ============================

def kelly_continuous(pnl):

    f_grid = np.linspace(0,0.5,200)

    best_f = 0
    best_val = -np.inf

    pnl = np.clip(pnl,-0.99,None)

    for f in f_grid:

        val = np.mean(np.log(1+f*pnl))

        if val > best_val:
            best_val = val
            best_f = f

    return best_f


# ============================
# TP SL OPTIMIZATION
# ============================

def optimize_tp_sl(mu, sigma, horizon):

    tp_range = np.linspace(0.01,0.25,15)
    sl_range = np.linspace(0.01,0.25,15)

    rows = []

    best_score = -np.inf
    best_row = None

    for tp in tp_range:
        for sl in sl_range:

            pnl = simulate_trade_distribution(
                mu,sigma,tp,sl,horizon
            )

            EV, sharpe, max_dd, cvar, skew = risk_metrics(pnl)

            score = objective(pnl)

            kelly = kelly_continuous(pnl)

            row = {
                "TP":tp,
                "SL":sl,
                "EV":EV,
                "Sharpe":sharpe,
                "MaxDD":max_dd,
                "CVaR":cvar,
                "Skew":skew,
                "Kelly":min(kelly,0.25),
                "Score":score
            }

            rows.append(row)

            if score > best_score:
                best_score = score
                best_row = row

    df = pd.DataFrame(rows)

    return best_row, df


# ============================
# REGRESSION CHART
# ============================

def plot_regression(close):

    df = pd.DataFrame({"price":close})

    df["t"] = np.arange(len(df))

    coef = np.polyfit(df["t"],df["price"],1)

    trend = np.poly1d(coef)

    df["trend"] = trend(df["t"])

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["price"],
            mode="lines",
            name="Preço"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["trend"],
            mode="lines",
            name="Regressão",
            line=dict(dash="dash")
        )
    )

    return fig

def read_tickers(file):

    content = file.read().decode("utf-8")

    tickers = [
        line.strip()
        for line in content.splitlines()
        if line.strip()
    ]

    return tickers

def analyze_ticker(ticker, horizon):

    close = load_data(ticker)

    if close is None:
        return None

    log_ret = compute_log_returns(close)

    mu = estimate_drift(log_ret)

    sigma = estimate_volatility_ewma(log_ret)

    best, df_all = optimize_tp_sl(mu, sigma, horizon)

    tp = best["TP"]
    sl = best["SL"]

    pnl = simulate_trade_distribution(
        mu, sigma, tp, sl, horizon
    )

    EV, sharpe, max_dd, cvar, skew = risk_metrics(pnl)

    prob_pos = (pnl > 0).mean()

    score = best["Score"]

    return {
        "Ticker": ticker,
        "Score": score,
        "EV": EV,
        "Sharpe": sharpe,
        "ProbWin": prob_pos,
        "TP": tp,
        "SL": sl,
        "Kelly": best["Kelly"]
    }


# ============================
# STREAMLIT UI
# ============================

st.title("Motor Quantitativo de Swing Trade")

uploaded_file = st.file_uploader(
    "Upload arquivo .txt com tickers",
    type=["txt"]
)

horizon = st.slider("Horizonte (dias)",5,60,15)

if uploaded_file and st.button("Analisar Ativos"):

    tickers = read_tickers(uploaded_file)

    results = []

    progress = st.progress(0)

    for i, ticker in enumerate(tickers):

        res = analyze_ticker(ticker, horizon)

        if res:
            results.append(res)

        progress.progress((i+1)/len(tickers))

    df = pd.DataFrame(results)

    if df.empty:
        st.error("Nenhum ativo válido.")
        st.stop()

    top5 = df.sort_values("Score", ascending=False).head(5)

    st.subheader("Top 5 Ativos para Swing")
    st.dataframe(top5)

    
