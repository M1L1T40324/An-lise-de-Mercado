# ================================
# IMPORTS
# ================================
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ================================
# FUNÇÕES AUXILIARES
# ================================

# Função para baixar dados
def get_data(ticker, period="1y"):
    """
    Baixa dados históricos do ativo usando yfinance.
    Para ativos da B3, adiciona '.SA'
    """
    ticker = ticker.upper() + ".SA"
    data = yf.download(ticker, period=period)
    return data

# Retornos
def calculate_returns(df):
    df['Return'] = df['Adj Close'].pct_change()
    return df

# Volatilidade anualizada
def calculate_volatility(df):
    return df['Return'].std() * np.sqrt(252)

# RSI
def calculate_rsi(df, period=14):
    delta = df['Adj Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Drawdown
def calculate_drawdown(df):
    cum = (1 + df['Return']).cumprod()
    peak = cum.cummax()
    drawdown = (cum - peak) / peak
    return drawdown.min()

# Sharpe Ratio simplificado
def calculate_sharpe(df):
    return (df['Return'].mean() / df['Return'].std()) * np.sqrt(252)

# ================================
# INTERFACE STREAMLIT
# ================================

st.set_page_config(page_title="Swing Trade Analyzer", layout="wide")

st.title("📈 Analisador de Swing Trade - B3")

ticker = st.text_input("Digite o ticker (ex: PETR4, VALE3):", "PETR4")
period = st.selectbox("Período:", ["6mo", "1y", "2y", "5y"])

if st.button("Analisar"):

    df = get_data(ticker, period)

    if df.empty:
        st.error("Erro ao buscar dados.")
    else:
        # ================================
        # CÁLCULOS
        # ================================
        df = calculate_returns(df)

        df['MA9'] = df['Adj Close'].rolling(9).mean()
        df['MA21'] = df['Adj Close'].rolling(21).mean()
        df['RSI'] = calculate_rsi(df)

        volatility = calculate_volatility(df)
        sharpe = calculate_sharpe(df)
        drawdown = calculate_drawdown(df)

        current_price = df['Adj Close'].iloc[-1]

        # ================================
        # KPIs
        # ================================
        st.subheader("📊 KPIs do Ativo")

        col1, col2, col3 = st.columns(3)

        col1.metric("Preço Atual", f"R$ {current_price:.2f}")
        col1.metric("Volatilidade (anual)", f"{volatility:.2%}")

        col2.metric("Sharpe Ratio", f"{sharpe:.2f}")
        col2.metric("Drawdown Máx", f"{drawdown:.2%}")

        col3.metric("RSI Atual", f"{df['RSI'].iloc[-1]:.2f}")
        col3.metric("Retorno Médio Diário", f"{df['Return'].mean():.4%}")

        # ================================
        # GRÁFICO
        # ================================
        st.subheader("📉 Gráfico de Preço")

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(df.index, df['Adj Close'], label='Preço')
        ax.plot(df.index, df['MA9'], label='MM9')
        ax.plot(df.index, df['MA21'], label='MM21')

        ax.set_title(f"{ticker.upper()} - Preço e Médias Móveis")
        ax.legend()
        ax.grid()

        st.pyplot(fig)

        # ================================
        # INTERPRETAÇÃO SIMPLES
        # ================================
        st.subheader("🧠 Leitura Rápida")

        if df['MA9'].iloc[-1] > df['MA21'].iloc[-1]:
            st.success("Tendência de curto prazo: Alta 📈")
        else:
            st.warning("Tendência de curto prazo: Baixa 📉")

        if df['RSI'].iloc[-1] > 70:
            st.warning("RSI indica sobrecompra ⚠️")
        elif df['RSI'].iloc[-1] < 30:
            st.success("RSI indica sobrevenda 🔥")
        else:
            st.info("RSI neutro")