import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

st.set_page_config(layout="wide")
st.title("📊 Dashboard Emocional de Ativos")

# =========================
# INPUTS
# =========================

tickers_input = st.text_input("Ativos (separados por vírgula)", "")
period = st.selectbox("Período", ["1y","2y","5y"])
tp_percent = st.number_input("Take Profit (%)", value=5.0)/100
sl_percent = st.number_input("Stop Loss (%)", value=5.0)/100
forecast_days = st.number_input("Período Projeção (dias)", value=30)
corr_limit = 0.7

if tickers_input:

    tickers = [t.strip().upper() for t in tickers_input.split(",")]
    returns_dict = {}
    results = []

    for ticker in tickers:

        data = yf.download(ticker, period=period)
        if data.empty:
            continue

        prices = data['Close']
        log_returns = np.log(prices/prices.shift(1)).dropna()
        returns_dict[ticker] = log_returns

        mu = float(log_returns.mean()*252)
        sigma = float(log_returns.std()*np.sqrt(252))
        
        S0 = float(prices.iloc[-1])
        T = forecast_days/252
        
        expected_price = float(S0*np.exp(mu*T))
        
        tp_price = S0*(1+tp_percent)
        sl_price = S0*(1-sl_percent)
        
        d_tp = (np.log(tp_price/S0)-(mu-0.5*sigma**2)*T)/(sigma*np.sqrt(T))
        prob_tp = float(1-norm.cdf(d_tp))
        
        d_sl = (np.log(sl_price/S0)-(mu-0.5*sigma**2)*T)/(sigma*np.sqrt(T))
        prob_sl = float(norm.cdf(d_sl))
        
        emotional_index = float((expected_price/S0-1)*prob_tp - sigma*prob_sl)

        if emotional_index > 0.05:
            emotion = "🟢 Otimista"
        elif emotional_index < -0.05:
            emotion = "🔴 Estressado"
        else:
            emotion = "🟡 Neutro"

        # =====================
        # BACKTEST
        # =====================

        wins = 0
        losses = 0
        for i in range(len(prices)-forecast_days):
            entry = prices.iloc[i]
            future = prices.iloc[i:i+forecast_days]

            if (future >= entry*(1+tp_percent)).any():
                wins += 1
            elif (future <= entry*(1-sl_percent)).any():
                losses += 1

        total = wins + losses
        winrate = wins/total if total>0 else 0
        payoff = (tp_percent*winrate - sl_percent*(1-winrate))

        # =====================
        # OUTPUT
        # =====================

        st.subheader(f"{ticker} - {emotion}")
        st.write(f"Winrate Backtest: {winrate:.2%}")
        st.write(f"Expectativa Matemática: {payoff:.4f}")
        st.write(f"Prob TP: {prob_tp:.2%} | Prob SL: {prob_sl:.2%}")

        results.append({
            "Ticker":ticker,
            "Score": emotional_index,
            "Retorno": expected_price/S0-1,
            "ProbTP": prob_tp
        })

    # =====================
    # OTIMIZAÇÃO COM CORRELAÇÃO
    # =====================

    if len(results)>10:

        st.header("Simulação Carteira Otimizada")

        df_ret = pd.DataFrame(returns_dict)
        corr_matrix = df_ret.corr()

        df_scores = pd.DataFrame(results).sort_values("Score",ascending=False)

        selected = []
        for ticker in df_scores["Ticker"]:
            if not selected:
                selected.append(ticker)
            else:
                if all(abs(corr_matrix[ticker][s]) < corr_limit for s in selected):
                    selected.append(ticker)
            if len(selected)==5:
                break

        st.write("Ativos Selecionados (baixa correlação + alto score):")
        st.write(selected)
