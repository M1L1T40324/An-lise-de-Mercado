import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.stats import norm

st.set_page_config(layout="wide")
st.title("📊 Dashboard Emocional de Ativos")


# =========================
# FUNÇÕES AUXILIARES
# =========================

def calculate_gbm_metrics(prices, forecast_days):
    log_returns = np.log(prices / prices.shift(1)).dropna()

    mu = float(log_returns.mean() * 252)
    sigma = float(log_returns.std() * np.sqrt(252))

    if sigma <= 0 or np.isnan(sigma):
        return None

    S0 = float(prices.iloc[-1])
    T = forecast_days / 252

    expected_price = float(S0 * np.exp(mu * T))
    expected_return = expected_price / S0 - 1

    z_pos = (-(mu - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    prob_positive = 1 - norm.cdf(z_pos)

    z_95 = 1.96
    lower_price = S0 * np.exp((mu - 0.5 * sigma**2) * T - z_95 * sigma * np.sqrt(T))
    upper_price = S0 * np.exp((mu - 0.5 * sigma**2) * T + z_95 * sigma * np.sqrt(T))

    return {
        "S0": S0,
        "mu": mu,
        "sigma": sigma,
        "expected_price": expected_price,
        "expected_return": expected_return,
        "prob_positive": prob_positive,
        "lower_price": lower_price,
        "upper_price": upper_price
    }


def optimize_tp_sl(S0, mu, sigma, forecast_days):

    if sigma <= 0:
        return None, None, None

    sigma_period = sigma * np.sqrt(forecast_days / 252)
    max_move = min(0.25, 2 * sigma_period)  # trava máxima em 25%

    tp_range = np.linspace(0.01, max_move, 25)
    sl_range = np.linspace(0.01, max_move, 25)

    best_ev = -np.inf
    best_tp = None
    best_sl = None

    alpha = (2 * mu) / (sigma**2)

    for tp in tp_range:
        for sl in sl_range:

            tp_price = S0 * (1 + tp)
            sl_price = S0 * (1 - sl)

            try:
                prob_tp = (
                    1 - (sl_price / S0) ** alpha
                ) / (
                    (tp_price / S0) ** alpha - (sl_price / S0) ** alpha
                )

                prob_sl = 1 - prob_tp
                ev = tp * prob_tp - sl * prob_sl

                if ev > best_ev:
                    best_ev = ev
                    best_tp = tp
                    best_sl = sl

            except:
                continue

    return best_tp, best_sl, best_ev


def backtest_tp_sl(prices, forecast_days, tp_percent, sl_percent):
    wins = 0
    losses = 0

    for i in range(len(prices) - forecast_days):
        entry = prices.iloc[i]
        future = prices.iloc[i:i + forecast_days]

        if (future >= entry * (1 + tp_percent)).any():
            wins += 1
        elif (future <= entry * (1 - sl_percent)).any():
            losses += 1

    total = wins + losses
    winrate = wins / total if total > 0 else 0
    payoff = tp_percent * winrate - sl_percent * (1 - winrate)

    return winrate, payoff


# =========================
# INPUTS
# =========================

tickers_input = st.text_input("Ativos (separados por vírgula)", "")
period = st.selectbox("Período", ["1y", "2y", "5y"])
tp_percent = st.number_input("Take Profit (%)", value=5.0) / 100
sl_percent = st.number_input("Stop Loss (%)", value=5.0) / 100
forecast_days = st.number_input("Período Projeção (dias)", value=30)
corr_limit = 0.7


# =========================
# PROCESSAMENTO
# =========================

if tickers_input:
    

    tickers = [t.strip().upper() for t in tickers_input.split(",")]
    returns_dict = {}
    results = []

    for ticker in tickers:

        data = yf.download(ticker, period=period)
        if data.empty:
            continue

        prices = data['Close']
        if isinstance(prices, pd.DataFrame):
            prices = prices.iloc[:, 0]

        prices = prices.dropna().astype(float)

        metrics = calculate_gbm_metrics(prices, forecast_days)
        if metrics is None:
            continue

        S0 = metrics["S0"]
        mu = metrics["mu"]
        sigma = metrics["sigma"]

        best_tp, best_sl, best_ev = optimize_tp_sl(S0, mu, sigma, forecast_days)

        tp_price = S0 * (1 + tp_percent)
        sl_price = S0 * (1 - sl_percent)

        alpha = (2 * mu) / (sigma**2)
        try:
            prob_tp = (
                1 - (sl_price / S0) ** alpha
            ) / (
                (tp_price / S0) ** alpha - (sl_price / S0) ** alpha
            )
        except:
            prob_tp = 0.5

        prob_sl = 1 - prob_tp
        ev_trade = tp_percent * prob_tp - sl_percent * prob_sl
        risk_5_losses = (prob_sl) ** 5

        winrate, payoff = backtest_tp_sl(
            prices, forecast_days, tp_percent, sl_percent
        )

        log_returns = np.log(prices / prices.shift(1)).dropna()
        returns_dict[ticker] = log_returns

        # =====================
        # OUTPUT
        # =====================

        st.subheader(ticker)
        st.write(f"Preço Atual: {S0:.2f}")
        st.write(f"Retorno Esperado ({forecast_days} dias): {metrics['expected_return']:.2%}")
        st.write(f"Preço Esperado: {metrics['expected_price']:.2f}")
        st.write(f"Probabilidade retorno positivo: {metrics['prob_positive']:.2%}")
        st.write("Probabilidades Condicionais:")
        st.write(f"TP antes do SL: {prob_tp:.2%}")
        st.write(f"SL antes do TP: {prob_sl:.2%}")
        st.write(f"Valor Esperado do Trade: {ev_trade:.4f}")
        st.write("Intervalo de Confiança 95%:")
        st.write(f"{metrics['lower_price']:.2f} — {metrics['upper_price']:.2f}")
        st.write(f"Risco de 5 perdas consecutivas: {risk_5_losses:.4%}")

        st.subheader("TP/SL Ideais (Max EV)")
        st.write(f"TP Ideal: {best_tp:.2%}" if best_tp else "TP Ideal: N/A")
        st.write(f"SL Ideal: {best_sl:.2%}" if best_sl else "SL Ideal: N/A")
        st.write(f"Valor Esperado Máximo: {best_ev:.4f}" if best_ev else "EV Máximo: N/A")

        st.write(f"Winrate Backtest: {winrate:.2%}")
        st.write(f"Expectativa Matemática: {payoff:.4f}")

        results.append({
            "Ticker": ticker,
            "Score": payoff
        })


    # =====================
    # OTIMIZAÇÃO COM CORRELAÇÃO
    # =====================

    if len(results) > 3:

        st.header("Simulação Carteira Otimizada")

        df_ret = pd.DataFrame(returns_dict)
        corr_matrix = df_ret.corr()

        df_scores = pd.DataFrame(results).sort_values("Score", ascending=False)

        selected = []

        for ticker in df_scores["Ticker"]:
            if not selected:
                selected.append(ticker)
            else:
                if all(abs(corr_matrix[ticker][s]) < corr_limit for s in selected):
                    selected.append(ticker)
            if len(selected) == 5:
                break

        st.write("Ativos Selecionados (baixa correlação + alta expectativa):")
        st.write(selected)
