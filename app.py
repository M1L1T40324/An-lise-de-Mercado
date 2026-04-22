import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestRegressor
from arch import arch_model

# =========================
# CONFIG
# =========================

st.set_page_config(layout="wide")
st.title("📊 Quant Trading Pipeline (Corrigido e Profissional)")

# =========================
# PARÂMETROS
# =========================

TP = 0.2
SL = -0.1
N_DIAS = 20
N_SIM = 10000
COST = 0.001

# =========================
# INPUT
# =========================

tickers_input = st.text_input("Tickers (ex: PETR4.SA, VALE3.SA)")
rodar = st.button("Executar")

# =========================
# FUNÇÕES
# =========================

def baixar_dados(tickers):
    dados = {}
    for t in tickers:
        df = yf.download(t, period="2y", progress=False)

        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            dados[t] = df

    return dados


def features(df):
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["vol"] = df["log_return"].rolling(10).std()
    df["momentum"] = df["Close"] - df["Close"].shift(5)
    df["lag1"] = df["log_return"].shift(1)
    df["lag2"] = df["log_return"].shift(2)
    return df


def ajustar_garch(data):
    model = arch_model(data*100, vol="Garch", p=1, q=1)
    res = model.fit(disp="off")
    return res


def monte_carlo_garch(res, mu_ml=0):

    params = res.params
    omega = params['omega']
    alpha = params['alpha[1]']
    beta = params['beta[1]']

    tp_hit = 0
    sl_hit = 0
    none_hit = 0

    dt = 1/252

    for _ in range(N_SIM):

        sigma2 = omega / (1 - alpha - beta)
        S = 1
        hit = False

        for _ in range(N_DIAS):

            z = np.random.normal()

            sigma2 = omega + alpha*(z**2) + beta*sigma2
            sigma = np.sqrt(sigma2) / 100  # 🔥 CORREÇÃO

            S *= np.exp((mu_ml - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*z)

            if S - 1 >= TP:
                tp_hit += 1
                hit = True
                break

            if S - 1 <= SL:
                sl_hit += 1
                hit = True
                break

        if not hit:
            none_hit += 1

    total = N_SIM

    return tp_hit/total, sl_hit/total, none_hit/total
def modelo_ml(df):

    df["target"] = df["log_return"].shift(-1)

    cols = ["log_return","lag1","lag2","momentum","vol"]

    df = df[cols + ["target"]].dropna()

    split = int(len(df)*0.8)

    train = df.iloc[:split]
    test = df.iloc[split:]

    model = RandomForestRegressor(n_estimators=100)
    model.fit(train[cols], train["target"])

    pred = model.predict(test[cols])

    return test.reset_index(drop=True), pred


def backtest(test, pred):

    returns = test["target"].values

    # posição proporcional ao retorno esperado
    pos = pred / (np.std(pred) + 1e-6)

    strat = pos * returns

    # custo
    trades = np.abs(np.diff(pos))
    strat[1:] -= trades * COST

    strat = pd.Series(strat).fillna(0)

    sharpe = strat.mean()/strat.std()*np.sqrt(252)

    cum = (1+strat).cumprod()
    cum = cum / cum.iloc[0]

    return strat, sharpe, cum


# =========================
# EXECUÇÃO
# =========================

if rodar:

    tickers = [t.strip().upper() for t in tickers_input.split(",")]
    dados = baixar_dados(tickers)

    for nome, df in dados.items():

        st.header(f"📌 {nome}")

        df = features(df)
        data = df["log_return"].dropna()

        if len(data) < 200:
            st.warning("Poucos dados")
            continue

        # -----------------
        # GARCH
        # -----------------

        try:
            garch = ajustar_garch(data)
        except:
            st.warning("GARCH falhou")
            continue

        # -----------------
        # ML
        # -----------------

        test, pred = modelo_ml(df)

        if len(pred) == 0:
            st.warning("ML falhou")
            continue

        # -----------------
        # BACKTEST
        # -----------------

        strat, sharpe, cum = backtest(test, pred)

        st.write("Sharpe:", round(sharpe, 3))

        fig = go.Figure()
        fig.add_trace(go.Scatter(y=cum, name="Estratégia"))
        fig.update_layout(title="Equity Curve")
        st.plotly_chart(fig)

        # -----------------
        # TP vs SL
        # -----------------

        p_tp, p_sl, p_none = monte_carlo_garch(garch, mu_ml=pred[-1])
        st.subheader("🎯 Probabilidade de Trade")

        st.write(f"TP ({TP*100:.1f}%): {p_tp:.2%}")
        st.write(f"SL ({SL*100:.1f}%): {p_sl:.2%}")

        expectativa = p_tp*TP + p_sl*SL
        st.write("Expectativa:", round(expectativa,4))
        st.subheader("🎯 Probabilidades do Trade")
        st.write(f"TP: {p_tp:.2%}")
        st.write(f"SL: {p_sl:.2%}")
        st.write(f"Nenhum: {p_none:.2%}")

        # -----------------
        # DECISÃO
        # -----------------

        if expectativa > 0:
            st.success("Trade com edge positivo")
        else:
            st.error("Sem vantagem estatística")