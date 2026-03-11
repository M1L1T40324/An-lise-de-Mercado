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

@st.cache_data
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

        return data

    except:
        return None


# ============================
# RETURNS
# ============================

def compute_log_returns(close):

    return np.log(close / close.shift(1)).dropna()


# ============================
# DRIFT (SHRINKAGE)
# ============================

def estimate_drift(log_ret):

    mu_hist = log_ret.mean()

    shrink = 0.25

    return float(shrink * mu_hist)


# ============================
# VOLATILITY EWMA
# ============================

def estimate_volatility_ewma(log_ret, lam=0.94):

    var = log_ret.var()

    for r in log_ret:
        var = lam * var + (1 - lam) * (r**2)

    return float(np.sqrt(var))


# ============================
# ATR VOLATILITY
# ============================

def compute_atr(data, window=14):

    high = data["High"]
    low = data["Low"]
    close = data["Close"]

    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(window).mean()

    return float(atr.iloc[-1])


# ============================
# STUDENT T RETURNS
# ============================

def simulate_returns(mu, sigma, horizon, n_sim=4000, df=5):

    shocks = t.rvs(df, size=(n_sim, horizon))

    scale = sigma / np.sqrt(df/(df-2))

    returns = mu + scale * shocks

    return returns


# ============================
# PATHS
# ============================

def simulate_paths(mu, sigma, horizon, n_sim=4000):

    returns = simulate_returns(mu, sigma, horizon, n_sim)

    return np.cumsum(returns, axis=1)


# ============================
# TRADE DISTRIBUTION
# ============================

def simulate_trade_distribution(mu, sigma, tp, sl, horizon):

    paths = simulate_paths(mu, sigma, horizon)

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

    sharpe = EV/std if std > 0 else 0

    cum = np.cumsum(pnl)

    dd = cum - np.maximum.accumulate(cum)

    max_dd = dd.min()

    cvar = pnl[pnl <= np.quantile(pnl,0.05)].mean()

    skew = pd.Series(pnl).skew()

    return EV, sharpe, max_dd, cvar, skew


# ============================
# OBJECTIVE CONSERVATIVE
# ============================

def objective_conservative(pnl):

    EV, sharpe, max_dd, cvar, skew = risk_metrics(pnl)

    prob_win = (pnl > 0).mean()

    score = (
        2.0 * prob_win +
        0.6 * sharpe +
        0.5 * EV -
        1.5 * abs(max_dd) -
        1.5 * abs(cvar)
    )

    return score


# ============================
# OBJECTIVE AGGRESSIVE
# ============================

def objective_aggressive(pnl):

    EV, sharpe, max_dd, cvar, skew = risk_metrics(pnl)

    growth = np.mean(np.log(1 + np.clip(pnl,-0.99,None)))

    score = (
        1.4 * growth +
        1.0 * EV +
        0.4 * sharpe +
        0.4 * skew -
        0.8 * abs(max_dd)
    )

    return score


# ============================
# KELLY
# ============================

def kelly_continuous(pnl):

    pnl = np.clip(pnl,-0.99,None)

    f_grid = np.linspace(0,0.5,150)

    best_f = 0
    best_val = -np.inf

    for f in f_grid:

        val = np.mean(np.log(1+f*pnl))

        if val > best_val:

            best_val = val
            best_f = f

    return best_f


# ============================
# TP SL OPTIMIZATION
# ============================

def optimize_tp_sl(mu, sigma, horizon, strategy, atr_pct):

    rows = []
    best_row = None
    best_score = -np.inf

    if strategy == "Conservadora":

        tp_range = np.linspace(0.01,0.02,8)
        sl = max(atr_pct*1.2,0.01)

        objective = objective_conservative

    else:

        tp_range = np.linspace(0.03,0.06,10)

        sl_base = max(atr_pct*1.5,0.015)
        sl_range = np.linspace(sl_base,0.04,8)

        objective = objective_aggressive


    if strategy == "Conservadora":

        for tp in tp_range:

            pnl = simulate_trade_distribution(
                mu,sigma,tp,sl,horizon
            )

            EV, sharpe, max_dd, cvar, skew = risk_metrics(pnl)

            prob_win = (pnl > 0).mean()

            score = objective(pnl)

            kelly = kelly_continuous(pnl)

            row = {
                "TP":tp,
                "SL":sl,
                "EV":EV,
                "Sharpe":sharpe,
                "ProbWin":prob_win,
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


    else:

        for tp in tp_range:

            for sl in sl_range:

                pnl = simulate_trade_distribution(
                    mu,sigma,tp,sl,horizon
                )

                EV, sharpe, max_dd, cvar, skew = risk_metrics(pnl)

                prob_win = (pnl > 0).mean()

                # FILTROS PROFISSIONAIS
                if (
                    EV < 0.006 or
                    prob_win < 0.45 or
                    sharpe < 0.2
                ):
                    continue

                score = objective(pnl)

                kelly = kelly_continuous(pnl)

                row = {
                    "TP":tp,
                    "SL":sl,
                    "EV":EV,
                    "Sharpe":sharpe,
                    "ProbWin":prob_win,
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


# ============================
# TICKER LOADER
# ============================

def read_tickers(file):

    content = file.read().decode("utf-8")

    content = content.replace("\n",",")

    tickers = [
        t.strip()
        for t in content.split(",")
        if t.strip()
    ]

    return tickers


# ============================
# ANALYZE TICKER
# ============================

def analyze_ticker(ticker, horizon, strategy):

    data = load_data(ticker)

    if data is None:
        return None

    close = data["Close"]

    log_ret = compute_log_returns(close)

    mu = estimate_drift(log_ret)

    sigma = estimate_volatility_ewma(log_ret)

    # VOLATILIDADE ANUAL
    vol_annual = sigma * np.sqrt(252)

    if vol_annual < 0.20:
        return None

    # TENDÊNCIA
    ma100 = close.rolling(100).mean().iloc[-1]

    if close.iloc[-1] < ma100:
        return None

    # LIQUIDEZ
    if data["Volume"].mean() < 200000:
        return None

    atr = compute_atr(data)

    atr_pct = atr / close.iloc[-1]

    best, df_all = optimize_tp_sl(
        mu,
        sigma,
        horizon,
        strategy,
        atr_pct
    )

    if best is None:
        return None

    tp = best["TP"]
    sl = best["SL"]

    pnl = simulate_trade_distribution(
        mu,
        sigma,
        tp,
        sl,
        horizon
    )

    EV, sharpe, max_dd, cvar, skew = risk_metrics(pnl)

    prob_win = (pnl > 0).mean()

    return {
        "Ticker":ticker,
        "Score":best["Score"],
        "ProbWin":prob_win,
        "EV":EV,
        "Sharpe":sharpe,
        "TP":tp,
        "SL":sl,
        "Kelly":best["Kelly"],
        "VolAnual":vol_annual
    }


# ============================
# STREAMLIT UI
# ============================

st.title("Motor Quantitativo de Swing Trade")

strategy = st.selectbox(
    "Estratégia",
    ["Conservadora","Agressiva"]
)

uploaded_file = st.file_uploader(
    "Upload arquivo .txt com tickers",
    type=["txt"]
)

horizon = st.slider("Horizonte (dias)",5,60,15)

if uploaded_file and st.button("Analisar Ativos"):

    tickers = read_tickers(uploaded_file)

    results = []

    progress = st.progress(0)

    for i,ticker in enumerate(tickers):

        res = analyze_ticker(
            ticker,
            horizon,
            strategy
        )

        if res:
            results.append(res)

        progress.progress((i+1)/len(tickers))

    df = pd.DataFrame(results)

    if df.empty:
        st.error("Nenhum ativo válido.")
        st.stop()

    top5 = df.sort_values(
        "Score",
        ascending=False
    ).head(5)

    st.subheader("Top 5 Ativos para Swing")

    st.dataframe(top5)
