# Streamlit app: GBM features + XGBoost TP/SL scanner

import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from arch import arch_model
from statsmodels.tsa.ar_model import AutoReg
from scipy.stats import norm

def ar_garch_features(close):
    close = close.astype(float)
    log_ret = np.log(close / close.shift(1)).dropna()

    # ===== AR(1) =====
    ar_model = AutoReg(log_ret, lags=1, old_names=False).fit()
    mu_hat = ar_model.fittedvalues

    # ===== GARCH(1,1) =====
    garch = arch_model(
        log_ret * 100,
        mean="Zero",
        vol="GARCH",
        p=1, q=1,
        dist="normal"
    ).fit(disp="off")

    sigma_hat = garch.conditional_volatility / 100

    df = pd.DataFrame(index=log_ret.index)
    df["mu_ar"] = mu_hat
    df["sigma_garch"] = sigma_hat
    df["z_score"] = log_ret / sigma_hat

    return df.dropna()

def prob_tp_sl(mu, sigma, tp, sl, horizon):
    mu_h = mu * horizon
    sigma_h = sigma * np.sqrt(horizon)

    p_tp = 1 - norm.cdf((tp - mu_h) / sigma_h)
    p_sl = norm.cdf((-sl - mu_h) / sigma_h)

    # normalização condicional
    total = p_tp + p_sl
    if total == 0:
        return 0.0, 0.0

    return p_tp / total, p_sl / total

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

def evaluate_tp_sl_ar_garch(df, feats, tp_list, sl_list, horizon):
    results = []

    mu = feats["mu_ar"].iloc[-1]
    sigma = feats["sigma_garch"].iloc[-1]

    for tp in tp_list:
        for sl in sl_list:

            p_tp, p_sl = prob_tp_sl(mu, sigma, tp, sl, horizon)

            if p_tp <= 0 or p_sl <= 0:
                continue

            EV = compute_ev(tp, sl, p_tp, p_sl)

            # Lévy penalty
            penalty = levy_tail_penalty(tp, sigma)

            EV_adj = EV * penalty

            kelly = kelly_fraction(tp, sl, p_tp)
            kelly = min(kelly, 0.3)  # Kelly fracionado

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

    feats = ar_garch_features(data["Close"])
    df = pd.concat([data, feats], axis=1).dropna()

    tp_list = np.linspace(0.02, 0.10, 6)
    sl_list = np.linspace(0.01, 0.06, 6)

    res = evaluate_tp_sl(df, feats, tp_list, sl_list, horizon)

    if res.empty:
        st.warning("Nenhuma combinação viável.")
        st.stop()

    best = res.sort_values("EV_adj", ascending=False).iloc[0]

    st.dataframe(res)
    st.success(
        f"Melhor combo → TP {best.TP:.2%}, SL {best.SL:.2%}, "
        f"EV ajustado {best.EV_adj:.2%}, Kelly {best.Kelly_frac:.2%}"
    )

st.divider()
st.subheader("📦 Scan multi-ticker (portfólio ótimo)")

if st.button("Rodar scan e montar portfólio"):
    tickers = st.text_area(
        "Tickers (separados por vírgula)",
        "PETR4.SA,VALE3.SA,ITUB4.SA,BBDC4.SA,BBAS3.SA,WEGE3.SA"
    ).split(",")

    portfolio_rows = []

    with st.spinner("Rodando modelos..."):
        for sym in tickers:
            sym = sym.strip()

            try:
                data = yf.download(sym, period="5y")

                # Fix Streamlit Cloud / yfinance
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)

                required_cols = {"Open", "High", "Low", "Close"}
                if not required_cols.issubset(data.columns):
                    continue

                feats = gbm_features(data["Close"])
                df = pd.concat([data, feats], axis=1).dropna()

                if len(df) < 300:
                    continue

                res = evaluate_tp_sl_ar_garch(df, feats, tp_list, sl_list, horizon)

                if res.empty:
                    continue

                best = res.sort_values("EV", ascending=False).iloc[0]

                # Kelly fracionado
                b = best.TP / best.SL
                p = best.Prob_TP
                kelly = max((p * (b + 1) - 1) / b, 0)

                portfolio_rows.append({
                    "Ticker": sym,
                    "TP": best.TP,
                    "SL": best.SL,
                    "Prob_TP": best.Prob_TP,
                    "EV_ajustado": best.EV,
                    "Kelly_%": kelly * 100
                })

            except:
                continue

    portfolio_df = pd.DataFrame(portfolio_rows)

    if portfolio_df.empty:
        st.warning("Nenhum ticker válido encontrado.")
        st.stop()

    # Ordena por EV ajustado
    portfolio_df = portfolio_df.sort_values("EV_ajustado", ascending=False)

    # Seleciona até Kelly somar 100%
    selected = []
    kelly_sum = 0.0

    for _, row in portfolio_df.iterrows():
        if kelly_sum + row["Kelly_%"] <= 100:
            selected.append(row)
            kelly_sum += row["Kelly_%"]
        else:
            break

    final_df = pd.DataFrame(selected)

    st.success(f"Portfólio montado | Kelly total: {kelly_sum:.2f}%")
    st.dataframe(final_df.reset_index(drop=True))




