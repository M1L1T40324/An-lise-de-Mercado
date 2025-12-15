!pip install streamlit yfinance numpy pandas scikit-learn



# Streamlit app: GBM features + XGBoost TP/SL scanner
# Autor: exemplo educacional

import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# =========================
# 1. GBM FEATURE ENGINEERING
# =========================

def gbm_features(close, window=20):
    log_ret = np.log(close / close.shift(1))

    mu = log_ret.rolling(window).mean() * 252
    sigma = log_ret.rolling(window).std() * np.sqrt(252)

    return pd.DataFrame({
        "mu_gbm": mu,
        "sigma_gbm": sigma,
        "ret_1d": log_ret,
        "vol_5d": log_ret.rolling(5).std(),
        "vol_10d": log_ret.rolling(10).std(),
    })

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

    return pd.Series(y)

# =========================
# 3. TRAIN XGBOOST MODEL
# =========================

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, shuffle=False
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )

    model.fit(X_train, y_train)
    prob = model.predict_proba(X_test)[:,1]

    auc = roc_auc_score(y_test, prob)
    return model, auc

# =========================
# 4. TP/SL GRID SEARCH
# =========================

def evaluate_tp_sl(df, model, features, tp_list, sl_list, horizon):
    results = []

    X_last = df[features].iloc[-1:]
    prob_tp = model.predict_proba(X_last)[0,1]

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
    data = yf.download(symbol, period="5y")

    feats = gbm_features(data["Close"])
    df = pd.concat([data, feats], axis=1).dropna()

    tp_list = np.linspace(0.05, 0.15, 5)
    sl_list = np.linspace(0.01, 0.10, 5)

    y = label_tp_sl(df, tp_list[0], sl_list[0], horizon)
    df = df.iloc[:len(y)]
    y = y.dropna()
    X = df.loc[y.index, feats.columns]

    model, auc = train_model(X, y)

    res = evaluate_tp_sl(df, model, feats.columns, tp_list, sl_list, horizon)
    best = res.sort_values("EV", ascending=False).iloc[0]

    st.write("AUC do modelo:", auc)
    st.dataframe(res)

    st.success(f"Melhor combinação → TP: {best.TP:.2%}, SL: {best.SL:.2%}, EV: {best.EV:.2%}")

# =========================
# 6. MULTI-TICKER SCAN
# =========================

if st.button("Scan múltiplos tickers"):
    tickers = st.text_area("Tickers (separados por vírgula)", "PETR4.SA,VALE3.SA,ITUB4.SA").split(",")
    scan_results = []

    for sym in tickers:
        try:
            data = yf.download(sym.strip(), period="5y")
            feats = gbm_features(data["Close"])
            df = pd.concat([data, feats], axis=1).dropna()

            y = label_tp_sl(df, 0.05, 0.02, horizon)
            df = df.iloc[:len(y)]
            y = y.dropna()
            X = df.loc[y.index, feats.columns]

            model, _ = train_model(X, y)
            res = evaluate_tp_sl(df, model, feats.columns, tp_list, sl_list, horizon)
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
    top4 = scan_df.sort_values("EV", ascending=False).head(4)

    st.subheader("Top 4 Tickers")
    st.dataframe(top4)
