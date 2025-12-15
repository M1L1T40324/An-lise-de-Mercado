# Streamlit app: GBM features + XGBoost TP/SL scanner
# Autor: exemplo educacional

import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

# =========================
# 1. GBM FEATURE ENGINEERING
# =========================

def gbm_features(close):
    close = close.astype(float)

    log_ret = np.log(close / close.shift(1))

    mu = log_ret.rolling(252).mean() * 252
    sigma = log_ret.rolling(252).std() * np.sqrt(252)

    vol_5d = log_ret.rolling(5).std()
    vol_10d = log_ret.rolling(10).std()

    df = pd.DataFrame(index=close.index)

    df["mu_gbm"] = mu
    df["sigma_gbm"] = sigma
    df["vol_5d"] = vol_5d
    df["vol_10d"] = vol_10d

    return df.dropna()


# =========================
# 2. TP / SL SIMULATION
# =========================

def label_tp_sl(df, tp, sl, horizon):
    y = []

    for i in range(len(df) - horizon):
        entry = df["Close"].iloc[i]
        future = df.iloc[i+1:i+horizon+1]

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

def train_model(X, y, min_samples=100):
    data = pd.concat([X, y.rename("target")], axis=1).dropna()

    if len(data) < min_samples:
        raise ValueError(
            f"Amostras insuficientes para treino: {len(data)} (< {min_samples})"
        )

    split = int(len(data) * 0.7)

    X_train = data.iloc[:split][X.columns]
    y_train = data.iloc[:split]["target"]

    X_test = data.iloc[split:][X.columns]
    y_test = data.iloc[split:]["target"]

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="auc",
        random_state=42
    )

    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)

    return model, auc

# =========================
# 4. TP/SL GRID SEARCH
# =========================

def evaluate_tp_sl(df, model, features, tp_list, sl_list):
    results = []

    X_last = df[features].iloc[-1:]
    prob_tp = model.predict_proba(X_last)[0, 1]

    for tp in tp_list:
        for sl in sl_list:
            EV = prob_tp * tp - (1 - prob_tp) * sl
            results.append({
                "TP": tp,
                "SL": sl,
                "Prob_TP": prob_tp,
                "EV": EV
            })

    return pd.DataFrame(results)

# =========================
# 5. STREAMLIT UI
# =========================

st.title("GBM + XGBoost TP/SL Scanner")

symbol = st.text_input("Ticker", "PETR4.SA")
horizon = st.slider("Horizonte (dias)", 5, 20, 10)

if st.button("Rodar modelo para 1 ativo"):
    data = yf.download(symbol, period="5y", auto_adjust=True)

    feats = gbm_features(data["Close"])
    df = pd.concat([data, feats], axis=1).dropna()

    tp_list = np.linspace(0.05, 0.15, 5)
    sl_list = np.linspace(0.01, 0.10, 5)

    y = label_tp_sl(df, tp_list[0], sl_list[0], horizon)

    X = df[feats.columns]
    
    if len(df) < 300:
    st.warning("Histórico insuficiente após feature engineering.")
    st.stop()

    try:
        model, auc = train_model(X, y)
    except ValueError as e:
        st.warning(str(e))
        st.stop()

    res = evaluate_tp_sl(df, model, feats.columns, tp_list, sl_list)
    best = res.sort_values("EV", ascending=False).iloc[0]

    st.write("AUC do modelo:", round(auc, 3))
    st.dataframe(res)

    st.success(
        f"Melhor combinação → TP: {best.TP:.2%}, "
        f"SL: {best.SL:.2%}, EV: {best.EV:.2%}"
    )

# =========================
# 6. MULTI-TICKER SCAN
# =========================

if st.button("Scan múltiplos tickers"):
    tickers = st.text_area(
        "Tickers (separados por vírgula)",
        "PETR4.SA,VALE3.SA,ITUB4.SA"
    ).split(",")

    scan_results = []

    for sym in tickers:
        try:
            data = yf.download(sym.strip(), period="5y", auto_adjust=True)
            feats = gbm_features(data["Close"])
            df = pd.concat([data, feats], axis=1).dropna()

            y = label_tp_sl(df, 0.05, 0.02, horizon)
            X = df[feats.columns]

            model, _ = train_model(X, y)
            res = evaluate_tp_sl(df, model, feats.columns, tp_list, sl_list)
            best = res.sort_values("EV", ascending=False).iloc[0]

            scan_results.append({
                "Ticker": sym.strip(),
                "TP": best.TP,
                "SL": best.SL,
                "EV": best.EV
            })
        except:
            continue

    scan_df = pd.DataFrame(scan_results)

    st.subheader("Top 4 Tickers")
    st.dataframe(scan_df.sort_values("EV", ascending=False).head(4))


