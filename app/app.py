# Versão refatorada com seleção de swing trade e simulação

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


st.set_page_config(page_title="☝🤓 AI Market Analysis", layout="wide")
st.title("📊 AI Market Analysis – Seleção e Simulação de Swing Trade")

# --- Entrada de Dados ---
tickers = st.text_input("Digite os tickers separados por vírgula:", "PETR4.SA, VALE3.SA, ITUB4.SA")
tickers = [t.strip().upper() for t in tickers.split(",") if t.strip()]

periodo = st.selectbox("Selecione o período:", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"])
intervalo = st.selectbox("Selecione o intervalo:", ["1d", "1wk", "1mo"])

data = yf.download(tickers, period=periodo, interval=intervalo, group_by='ticker', auto_adjust=True)

# --- Função para corrigir colunas quando multi-ticker ---
def corrigir_colunas(df, ticker):
    if isinstance(df.columns, pd.MultiIndex):
        df = df[ticker].copy()
    return df

# --- Função Seleção de Swing Trade ---
def selecionar_swingtrade(tickers):
    candidatos = []
    for t in tickers:
        try:
            df = yf.download(t, period="60d", interval="1d")
            df.dropna(inplace=True)

            df["MM20"] = df["Close"].rolling(20).mean()
            df["MM50"] = df["Close"].rolling(50).mean()
            df["RSI"] = ta.momentum.RSIIndicator(df["Close"]).rsi()
            df["Vol_Med"] = df["Volume"].rolling(20).mean()

            atual = df.iloc[-1]

            cond_tendencia = atual["MM20"] > atual["MM50"]
            cond_pullback = abs(atual["Close"] - atual["MM20"]) / atual["MM20"] <= 0.02
            cond_rsi = 45 <= atual["RSI"] <= 60
            cond_volume = atual["Volume"] >= atual["Vol_Med"]

            if cond_tendencia and cond_pullback and cond_rsi and cond_volume:
                candidatos.append((t, atual["Close"], atual["RSI"]))
        except:
            pass

    candidatos = sorted(candidatos, key=lambda x: x[2])
    return candidatos

st.header("🎯 Seleção Automática de Swing Trade")
if st.button("Selecionar oportunidades agora"):
    resultado = selecionar_swingtrade(tickers)
    if resultado:
        st.success("Tickers com setup favorável detectados:")
        st.dataframe(pd.DataFrame(resultado, columns=["Ticker", "Preço Atual", "RSI"]))
    else:
        st.warning("Nenhum setup ideal encontrado hoje.")

# --- Função Simulação Swing Trade ---
def simular_swingtrade(ticker, quantidade, dias_hold):
    df = yf.download(ticker, period="120d", interval="1d")
    df.dropna(inplace=True)

    preco_entrada = df["Close"].iloc[-1]
    indice_saida = len(df) - 1 - dias_hold
    if indice_saida < 0:
        return None

    preco_saida = df["Close"].iloc[indice_saida]

    lucro_bruto = (preco_saida - preco_entrada) * quantidade
    valor_venda = preco_saida * quantidade
    taxa_b3 = valor_venda * 0.0003
    corretagem = 0

    lucro_liquido = lucro_bruto - taxa_b3 - corretagem
    retorno_pct = (preco_saida / preco_entrada - 1) * 100

    return {
        "Ticker": ticker,
        "Preço Entrada": round(preco_entrada, 2),
        "Preço Saída (Simulado)": round(preco_saida, 2),
        "Qtd": quantidade,
        "Lucro Bruto (R$)": round(lucro_bruto, 2),
        "Lucro Líquido (R$)": round(lucro_liquido, 2),
        "Retorno (%)": round(retorno_pct, 2)
    }

st.header("💸 Simulação de Operação (Swing Trade)")
sim_ticker = st.selectbox("Escolha um Ticker para simular", tickers)
quantidade = st.number_input("Quantidade de Ações", min_value=1, value=100)
dias_hold = st.slider("Dias até a Venda (Hold)", min_value=1, max_value=20, value=5)

if st.button("Simular Operação"):
    resultado = simular_swingtrade(sim_ticker, quantidade, dias_hold)
    if resultado:
        st.subheader("Resultado da Simulação")
        resultado = simular_swing(ticker, quantidade, dias_hold)
        col1, col2, col3 = st.columns(3)
        col1.metric("Preço de Entrada", f"R$ {resultado['Preço Entrada']}")
        col2.metric("Preço de Saída (Simulado)", f"R$ {resultado['Preço Saída (Simulado)']}")
        col3.metric("Quantidade", resultado["Qtd"])
        col4, col5 = st.columns(2)
        lucro = resultado["Lucro Líquido (R$)"]
        retorno_pct = resultado["Retorno (%)"]
        col4.metric(
            "Lucro Líquido",
            f"R$ {lucro:,.2f}",
            f"{retorno_pct:.2f}%"
        )
        col5.metric(
            "Lucro Bruto",
            f"R$ {resultado['Lucro Bruto (R$)']:,.2f}"
        )

    else:
        st.warning("Histórico insuficiente para simular esse prazo.")


