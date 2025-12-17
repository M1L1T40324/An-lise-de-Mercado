# Streamlit app: GBM features + XGBoost TP/SL scanner

import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
try:
    from arch import arch_model
    ARCH_OK = True
except:
    ARCH_OK = False
from statsmodels.tsa.ar_model import AutoReg
from scipy.stats import norm

def prob_tp_sl_deterministic(mu, sigma, tp, sl, horizon):
    """
    Probabilidade determinística de atingir TP ou SL
    assumindo retornos normais agregados no horizonte.
    """
    mu_h = mu * horizon
    sigma_h = sigma * np.sqrt(horizon)

    if sigma_h <= 0:
        return 0.0, 0.0

    z_tp = (tp - mu_h) / sigma_h
    z_sl = (-sl - mu_h) / sigma_h

    p_tp = 1 - norm.cdf(z_tp)
    p_sl = norm.cdf(z_sl)

    total = p_tp + p_sl
    if total == 0:
        return 0.0, 0.0

    return p_tp / total, p_sl / total


def ewma_volatility(returns, lambda_=0.94):
    var = returns.ewm(alpha=1 - lambda_).var()
    return np.sqrt(var)
    
def ar_garch_features_safe(close):
    close = close.astype(float)
    log_ret = np.log(close / close.shift(1)).dropna()

    # AR(1)
    ar = AutoReg(log_ret, lags=1, old_names=False).fit()
    mu_hat = ar.fittedvalues

    if ARCH_OK:
        garch = arch_model(
            log_ret * 100,
            mean="Zero",
            vol="GARCH",
            p=1, q=1,
            dist="normal"
        ).fit(disp="off")
        sigma_hat = garch.conditional_volatility / 100
    else:
        sigma_hat = ewma_volatility(log_ret)

    df = pd.DataFrame(index=log_ret.index)
    df["mu_ar"] = mu_hat
    df["sigma"] = sigma_hat

    return df.dropna()

def prob_tp_sl(mu, sigma, tp, sl, horizon, n_sim=3000):
    tp_hit = 0
    sl_hit = 0

    for _ in range(n_sim):
        path = np.cumsum(
            mu + sigma * np.random.randn(horizon)
        )

        if np.any(path >= tp):
            tp_time = np.argmax(path >= tp)
        else:
            tp_time = np.inf

        if np.any(path <= -sl):
            sl_time = np.argmax(path <= -sl)
        else:
            sl_time = np.inf

        if tp_time < sl_time:
            tp_hit += 1
        elif sl_time < tp_time:
            sl_hit += 1

    total = tp_hit + sl_hit
    if total == 0:
        return 0.0, 0.0

    return tp_hit / total, sl_hit / total


def levy_tail_penalty(tp, sigma, alpha=3.0, jump_intensity=0.02):
    """
    Penaliza TP agressivo considerando cauda pesada
    """
    tail_prob = (tp / sigma) ** (-alpha)
    crash_risk = jump_intensity * tail_prob

    penalty = max(0, 1 - crash_risk)
    return penalty

def compute_ev(tp, sl, p_tp, p_sl):
    return p_tp * tp - p_sl * sl

def kelly_fraction(tp, sl, p):
    b = tp / sl
    return max((p * (b + 1) - 1) / b, 0)


# =========================
# 1. GBM FEATURE ENGINEERING
# =========================

def gbm_features(close):
    close = close.astype(float)
    log_ret = np.log(close / close.shift(1))

    mu = log_ret.rolling(252).mean() * 252
    sigma = log_ret.rolling(252).std() * np.sqrt(252)

    df = pd.DataFrame(index=close.index)
    df["mu_gbm"] = mu
    df["sigma_gbm"] = sigma
    df["vol_5d"] = log_ret.rolling(5).std()
    df["vol_10d"] = log_ret.rolling(10).std()

    return df.dropna()


# =========================
# 2. TP / SL SIMULATION
# =========================

def label_tp_sl(df, tp, sl, horizon):
    y = []

    for i in range(len(df) - horizon):
        entry = float(df["Close"].iloc[i])
        future = df.iloc[i + 1 : i + horizon + 1]

        tp_price = entry * (1 + tp)
        sl_price = entry * (1 - sl)

        hit_tp = (future["High"] >= tp_price).any()
        hit_sl = (future["Low"] <= sl_price).any()

        if hit_tp and not hit_sl:
            y.append(1)
        elif hit_sl and not hit_tp:
            y.append(0)
        else:
            y.append(np.nan)

    return pd.Series(y, index=df.index[:len(y)])

# =========================
# 3. TRAIN XGBOOST MODEL
# =========================

def train_model(X, y, min_samples=150):
    data = pd.concat([X, y.rename("target")], axis=1).dropna()

    if len(data) < min_samples:
        raise ValueError(f"Amostras insuficientes: {len(data)}")

    split = int(len(data) * 0.7)

    X_train, y_train = data.iloc[:split][X.columns], data.iloc[:split]["target"]
    X_test, y_test = data.iloc[split:][X.columns], data.iloc[split:]["target"]

    model = XGBClassifier(
        n_estimators=250,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="auc",
        random_state=42
    )

    model.fit(X_train, y_train)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    return model, auc


# =========================
# 4. TP/SL GRID SEARCH (COM RISCO)
# =========================

def simulate_strategy(mu, sigma, tp, sl, horizon, n_sim=10000):
    pnl = []

    for _ in range(n_sim):
        path = np.cumsum(mu + sigma * np.random.randn(horizon))

        if np.any(path >= tp):
            pnl.append(tp)
        elif np.any(path <= -sl):
            pnl.append(-sl)
        else:
            pnl.append(path[-1])

    return np.array(pnl)


def risk_metrics(pnl, alpha=0.95):
    cum = np.cumsum(pnl)
    dd = cum - np.maximum.accumulate(cum)

    return {
        "Max_Drawdown": dd.min(),
        "CVaR": pnl[pnl <= np.quantile(pnl, 1 - alpha)].mean(),
        "Prob_Ruina": np.mean(pnl < dd.min())
    }

def evaluate_tp_sl_ar_garch(df, feats, tp_list, sl_list, horizon):
    results = []

    mu = feats["mu_ar"].iloc[-1]
    sigma = feats["sigma"].iloc[-1]

    for tp in tp_list:
        for sl in sl_list:

            # --- PROBABILIDADE DETERMINÍSTICA ---
            p_tp, p_sl = prob_tp_sl_deterministic(
                mu,
                sigma,
                tp,
                sl,
                horizon
            )

            if p_tp <= 0 or p_sl <= 0:
                continue

            EV = compute_ev(tp, sl, p_tp, p_sl)

            # Penalização por SL irrealista
            sl_penalty = min(
                1.0,
                sl / (2 * sigma * np.sqrt(horizon))
            )
            EV *= sl_penalty

            # Penalização de cauda (Lévy)
            penalty = levy_tail_penalty(
                tp,
                sigma * np.sqrt(horizon),
                alpha=2.5,
                jump_intensity=0.05
            )

            EV_adj = EV * penalty

            # Kelly fracionado e ajustado por horizonte
            kelly_raw = kelly_fraction(tp, sl, p_tp)
            kelly = min(kelly_raw / np.sqrt(horizon), 0.15)

            results.append({
                "TP": tp,
                "SL": sl,
                "Prob_TP": p_tp,
                "Prob_SL": p_sl,
                "EV": EV,
                "EV_adj": EV_adj,
                "Kelly_frac": kelly,
                "Levy_penalty": penalty
            })

    return pd.DataFrame(results)



# =========================
# 5. STREAMLIT UI
# =========================

st.title("GBM + XGBoost TP/SL Scanner (Risk-Aware)")

symbol = st.text_input("Ticker", "PETR4.SA")
horizon = st.slider("Horizonte (dias)", 5, 20, 10)

if st.button("Rodar modelo"):
    data = yf.download(symbol, period="5y", auto_adjust=True)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    feats = ar_garch_features_safe(data["Close"])
    df = pd.concat([data, feats], axis=1).dropna()

    tp_list = np.linspace(0.02, 0.10, 6)
    sl_list = np.linspace(0.01, 0.06, 6)

    res = evaluate_tp_sl_ar_garch(df, feats, tp_list, sl_list, horizon)

    if res.empty:
        st.warning("Nenhuma combinação viável.")
        st.stop()

    best = res.sort_values("EV_adj", ascending=False).iloc[0]

    b = best.TP / best.SL
    p = best.Prob_TP
    kelly_raw = (p * (b + 1) - 1) / b
    kelly = np.clip(kelly_raw, 0, 0.25)

    st.dataframe(res)
    st.success(
        f"Melhor combo → TP {best.TP:.2%}, SL {best.SL:.2%}, "
        f"EV ajustado {best.EV_adj:.2%}, Kelly {best.Kelly_frac:.2%}"
    )

st.sidebar.markdown("### 📥 Entrada de tickers")

uploaded_file = st.file_uploader(
    "Upload CSV ou TXT com tickers",
    type=["csv", "txt"]
)

raw_tickers = ""

if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        df_t = pd.read_csv(uploaded_file, header=None)
        raw_tickers = ",".join(df_t.iloc[:, 0].astype(str))
    else:
        raw_tickers = uploaded_file.read().decode("utf-8")

st.sidebar.subheader("📦 Scan multi-ticker (portfólio ótimo)")

if st.sidebar.button("Rodar scan e montar portfólio"):

    raw_tickers = st.text_area(
        "Tickers (vírgula ou quebra de linha)",
        raw_tickers,
        height=200
    )

    tickers = [
        t.strip().upper()
        for t in raw_tickers.replace("\n", ",").split(",")
        if t.strip() != ""
    ]

    # =============================
    # FASE 1 — SCAN DETERMINÍSTICO
    # =============================
    portfolio_rows = []

    progress = st.progress(0)
    status = st.empty()

    with st.spinner("Fase 1: Scan determinístico..."):
        for i, sym in enumerate(tickers):
            try:
                status.text(f"[Fase 1] {sym} ({i+1}/{len(tickers)})")

                data = yf.download(
                    sym, period="5y", auto_adjust=True, progress=False
                )

                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)

                if not {"Open", "High", "Low", "Close"}.issubset(data.columns):
                    continue

                feats = ar_garch_features_safe(data["Close"])
                df = pd.concat([data, feats], axis=1).dropna()

                if len(df) < 300:
                    continue

                tp_list = np.linspace(0.02, 0.10, 6)
                sl_list = np.linspace(0.01, 0.06, 6)

                # 👉 USO DETERMINÍSTICO
                res = evaluate_tp_sl_ar_garch(
                    df, feats, tp_list, sl_list, horizon
                )

                if res.empty:
                    continue

                best = res.sort_values(
                    "EV_adj", ascending=False
                ).iloc[0]

                portfolio_rows.append({
                    "Ticker": sym,
                    "TP": best.TP,
                    "SL": best.SL,
                    "mu": feats["mu_ar"].iloc[-1],
                    "sigma": feats["sigma"].iloc[-1],
                    "EV_det": best.EV_adj,
                    "Kelly_det": best.Kelly_frac
                })

            except Exception:
                continue

            progress.progress((i + 1) / len(tickers))

    portfolio_df = pd.DataFrame(portfolio_rows)

    if portfolio_df.empty:
        st.warning("Nenhum ticker válido encontrado.")
        st.stop()

    # Rank determinístico
    portfolio_df = portfolio_df.sort_values(
        "EV_det", ascending=False
    )

    # Seleciona apenas os melhores (ex: top 10)
    top_n = min(10, len(portfolio_df))
    candidates_df = portfolio_df.head(top_n).copy()

    # =============================
    # FASE 2 — SIMULAÇÃO (APENAS TOPS)
    # =============================
    sim_rows = []

    with st.spinner("Fase 2: Simulações Monte Carlo..."):
        for _, row in candidates_df.iterrows():
            mu = row["mu"]
            sigma = row["sigma"]
            tp = row["TP"]
            sl = row["SL"]

            # 👉 AGORA SIM: simulação
            p_tp, p_sl = prob_tp_sl(
                mu, sigma, tp, sl, horizon, n_sim=5000
            )

            if p_tp <= 0 or p_sl <= 0:
                continue

            EV_sim = compute_ev(tp, sl, p_tp, p_sl)
            kelly_sim = kelly_fraction(tp, sl, p_tp)

            sim_rows.append({
                "Ticker": row["Ticker"],
                "TP": tp,
                "SL": sl,
                "Prob_TP": p_tp,
                "EV_ajustado": EV_sim,
                "Kelly_%": min(kelly_sim / np.sqrt(horizon), 0.15) * 100
            })

    sim_df = pd.DataFrame(sim_rows)

    if sim_df.empty:
        st.warning("Nenhuma estratégia válida após simulação.")
        st.stop()

    # =============================
    # MONTA PORTFÓLIO FINAL
    # =============================
    sim_df = sim_df.sort_values("EV_ajustado", ascending=False)

    selected = []
    kelly_sum = 0.0

    for _, row in sim_df.iterrows():
        if kelly_sum + row["Kelly_%"] <= 100.0:
            selected.append(row)
            kelly_sum += row["Kelly_%"]
        else:
            break

    final_df = pd.DataFrame(selected)

    # =============================
    # EV DA CARTEIRA
    # =============================
    final_df["w"] = final_df["Kelly_%"] / 100

    EV_carteira = (final_df["w"] * final_df["EV_ajustado"]).sum()
    EV_dia = EV_carteira / horizon
    EV_anual = (1 + EV_carteira) ** (252 / horizon) - 1

    st.success(f"Portfólio montado | Kelly total: {kelly_sum:.2f}%")

    st.metric(
        f"EV esperado da carteira (H = {horizon} dias)",
        f"{EV_carteira:.2%}"
    )

    st.metric(
        "EV médio diário (aprox.)",
        f"{EV_dia:.3%}"
    )

    st.metric(
        "EV anualizado (composto)",
        f"{EV_anual:.2%}"
    )

    st.dataframe(final_df.reset_index(drop=True))








