import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.ar_model import AutoReg
import warnings

warnings.filterwarnings("ignore")

# ============================================
# CONFIGURAÇÕES
# ============================================

TICKERS = ["PETR4.SA","VALE3.SA","ITUB4.SA","BBDC4.SA","BBAS3.SA"]
HORIZON = 21
SIMS = 5000
TP = 0.10
SL = -0.05

MAX_PORTFOLIO_EXPOSURE = 0.60
MIN_VOLUME = 5_000_000

# ============================================
# FUNÇÕES
# ============================================

def get_data(ticker):
    data = yf.download(ticker, period="2y", interval="1d", progress=False)
    return data

def estimate_params(data):
    returns = np.log(data["Close"] / data["Close"].shift(1)).dropna()

    # AR(1)
    model = AutoReg(returns, lags=1).fit()
    mu_raw = model.params[0]

    # Shrinkage agressivo-controlado (60% original)
    mu = 0.6 * mu_raw
    mu = np.clip(mu, -0.02, 0.02)

    sigma = returns.std()

    return mu, sigma, returns

def simulate_paths(mu, sigma):
    dt = 1
    paths = []

    for _ in range(SIMS):
        # Student-t com df=5 (cauda pesada)
        z = np.random.standard_t(df=5, size=HORIZON)
        z = z / np.sqrt(5/(5-2))

        path = np.cumsum(mu*dt + sigma*np.sqrt(dt)*z)
        paths.append(path)

    return np.array(paths)

def prob_tp_sl(paths, tp, sl):
    tp_hits = 0
    sl_hits = 0

    for path in paths:
        if np.max(path) >= tp:
            tp_hits += 1
        elif np.min(path) <= sl:
            sl_hits += 1

    p_tp = tp_hits / len(paths)

    # Haircut estrutural (10%)
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
    try:
        data = get_data(ticker)

        if data["Volume"].rolling(20).mean().iloc[-1] < MIN_VOLUME:
            continue

        mu, sigma, returns = estimate_params(data)

        paths = simulate_paths(mu, sigma)
        p_tp, p_sl = prob_tp_sl(paths, TP, SL)

        EV = p_tp*TP - p_sl*abs(SL)

        var = p_tp*(TP**2) + p_sl*(SL**2) - EV**2
        std_strategy = np.sqrt(abs(var))

        score = EV / std_strategy if std_strategy > 0 else 0

        kelly = 0.5 * kelly_fraction(TP, SL, p_tp)  # Half Kelly

        results.append({
            "Ticker": ticker,
            "EV": EV,
            "Score": score,
            "Kelly": kelly
        })

        returns_dict[ticker] = returns

    except:
        continue

df = pd.DataFrame(results)
df = df.sort_values("Score", ascending=False)

# ============================================
# PENALIZAÇÃO POR CORRELAÇÃO (Drawdown)
# ============================================

returns_df = pd.DataFrame(returns_dict)
dd = returns_df.cumsum() - returns_df.cumsum().cummax()
corr_matrix = dd.corr()

adjusted_weights = []

for i, row in df.iterrows():
    ticker = row["Ticker"]

    avg_corr = corr_matrix[ticker].drop(ticker).mean()
    corr_penalty = 1 - avg_corr

    adj_score = row["Score"] * corr_penalty
    adj_kelly = row["Kelly"] * corr_penalty

    adjusted_weights.append((ticker, adj_score, adj_kelly))

final_df = pd.DataFrame(adjusted_weights,
                        columns=["Ticker","AdjScore","Kelly"])

final_df = final_df.sort_values("AdjScore", ascending=False)

# ============================================
# NORMALIZAÇÃO DA CARTEIRA
# ============================================

kelly_sum = final_df["Kelly"].sum()

if kelly_sum > MAX_PORTFOLIO_EXPOSURE:
    scale = MAX_PORTFOLIO_EXPOSURE / kelly_sum
    final_df["Kelly"] *= scale

final_df["Weight_%"] = final_df["Kelly"] * 100

print(final_df)
