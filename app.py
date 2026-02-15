import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.ar_model import AutoReg
import streamlit as st
import warnings

warnings.filterwarnings("ignore")

# ============================================
# CONFIGURAÇÕES
# ============================================

TICKERS = ["PETR4.SA","VALE3.SA","ITUB4.SA","BBDC4.SA","BBAS3.SA"]
HORIZON = 21
SIMS = 4000
TP = 0.10
SL = -0.05

MAX_PORTFOLIO_EXPOSURE = 0.60
MIN_VOLUME = 5_000_000

# ============================================
# FUNÇÕES
# ============================================

def get_data(ticker):
    try:
        data = yf.download(ticker, period="2y", interval="1d", progress=False)
        if data.empty:
            return None
        return data
    except:
        return None

def estimate_params(data):
    returns = np.log(data["Close"] / data["Close"].shift(1)).dropna()

    if len(returns) < 50:
        return None, None, None

    try:
        model = AutoReg(returns, lags=1).fit()
        mu_raw = float(model.params.iloc[0])
    except:
        mu_raw = float(returns.mean())

    mu = float(0.6 * mu_raw)
    mu = float(np.clip(mu, -0.02, 0.02))

    sigma = float(returns.std())

    return mu, sigma, returns


def simulate_paths(mu, sigma):

    mu = float(mu)
    sigma = float(sigma)

    dt = 1

    z = np.random.standard_t(df=5, size=(SIMS, HORIZON))
    z = z / np.sqrt(5/(5-2))

    drift = mu * dt
    diffusion = sigma * np.sqrt(dt) * z

    paths = np.cumsum(drift + diffusion, axis=1)

    return paths


def prob_tp_sl(paths, tp, sl):
    tp_hits = np.any(paths >= tp, axis=1)
    sl_hits = np.any(paths <= sl, axis=1)

    p_tp = np.mean(tp_hits)

    # haircut estrutural
    p_tp *= 0.90
    p_sl = 1 - p_tp

    return p_tp, p_sl

def kelly_fraction(tp, sl, p):
    b = tp / abs(sl)
    q = 1 - p
    k = (b*p - q)/b
    return max(k, 0)

# ============================================
# EXECUÇÃO
# ============================================

results = []
returns_dict = {}

for ticker in TICKERS:

    data = get_data(ticker)
    if data is None:
        st.write(f"{ticker} sem dados.")
        continue

    # Volume opcional (não trava tudo)
    try:
        vol = data["Volume"].rolling(20).mean().iloc[-1]
        if pd.notna(vol) and vol < MIN_VOLUME:
            st.write(f"{ticker} removido por baixo volume.")
            continue
    except:
        pass

    mu, sigma, returns = estimate_params(data)
    if mu is None:
        st.write(f"{ticker} sem dados suficientes.")
        continue

    paths = simulate_paths(mu, sigma)
    p_tp, p_sl = prob_tp_sl(paths, TP, SL)

    EV = p_tp*TP - p_sl*abs(SL)

    var = p_tp*(TP**2) + p_sl*(SL**2) - EV**2
    std_strategy = np.sqrt(abs(var))

    if std_strategy == 0:
        continue

    score = EV / std_strategy
    kelly = 0.5 * kelly_fraction(TP, SL, p_tp)

    results.append({
        "Ticker": ticker,
        "EV": EV,
        "Score": score,
        "Kelly": kelly
    })

    returns_dict[ticker] = returns

# ============================================
# VALIDAÇÃO ANTES DE ORDENAR
# ============================================

if len(results) == 0:
    st.error("Nenhum ativo passou nos filtros. Ajuste parâmetros.")
    st.stop()

df = pd.DataFrame(results)

if "Score" not in df.columns:
    st.error("Erro interno: coluna Score não encontrada.")
    st.stop()

df = df.sort_values("Score", ascending=False)

# ============================================
# CORRELAÇÃO VIA DRAWDOWN
# ============================================

returns_df = pd.DataFrame(returns_dict)

if returns_df.shape[1] > 1:

    dd = returns_df.cumsum() - returns_df.cumsum().cummax()
    corr_matrix = dd.corr()

    adjusted_rows = []

    for _, row in df.iterrows():
        ticker = row["Ticker"]

        if ticker not in corr_matrix.columns:
            continue

        avg_corr = corr_matrix[ticker].drop(ticker).mean()
        corr_penalty = 1 - avg_corr

        adj_score = row["Score"] * corr_penalty
        adj_kelly = row["Kelly"] * corr_penalty

        adjusted_rows.append({
            "Ticker": ticker,
            "AdjScore": adj_score,
            "Kelly": adj_kelly
        })

    final_df = pd.DataFrame(adjusted_rows)
    final_df = final_df.sort_values("AdjScore", ascending=False)

else:
    final_df = df.copy()
    final_df["AdjScore"] = final_df["Score"]

# ============================================
# NORMALIZAÇÃO DE EXPOSIÇÃO
# ============================================

kelly_sum = final_df["Kelly"].sum()

if kelly_sum > MAX_PORTFOLIO_EXPOSURE:
    scale = MAX_PORTFOLIO_EXPOSURE / kelly_sum
    final_df["Kelly"] *= scale

final_df["Weight_%"] = final_df["Kelly"] * 100

# ============================================
# OUTPUT
# ============================================

st.subheader("Carteira Agressiva Controlada")
st.dataframe(final_df)
