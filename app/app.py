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

def evaluate_tp_sl(df, features_df, tp_list, sl_list, horizon):
    results = []

    sigma_ref = features_df["sigma_gbm"].iloc[-1]
    max_tp_realista = 2 * sigma_ref      # filtro físico
    kelly_frac = 0.3                     # Kelly fracionado

    for tp in tp_list:
        if tp > max_tp_realista:
            continue

        for sl in sl_list:
            y = label_tp_sl(df[["Open", "High", "Low", "Close"]], tp, sl, horizon).dropna()
            if len(y) < 80:
                continue

            X = features_df.loc[y.index]

            try:
                model, auc = train_model(X, y)
            except:
                continue

            prob_tp = model.predict_proba(X.iloc[-1:])[0, 1]

            # EV clássico
            EV = prob_tp * tp - (1 - prob_tp) * sl

            # Kelly
            b = tp / sl
            q = 1 - prob_tp
            kelly = (b * prob_tp - q) / b if b > 0 else 0
            kelly = max(0, kelly * kelly_frac)

            # EV ajustado ao risco
            EV_adj = EV * kelly

            results.append({
                "TP": tp,
                "SL": sl,
                "Prob_TP": prob_tp,
                "EV": EV,
                "EV_adj": EV_adj,
                "Kelly_frac": kelly,
                "AUC": auc
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

    feats = gbm_features(data["Close"])
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
