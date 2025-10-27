import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from datetime import date, timedelta

st.set_page_config(page_title="Análise de Mercado com IA", layout="wide")

st.title("📈 Sistema Inteligente de Análise de Ações")
st.markdown("Explore indicadores, gráficos e previsões com modelos de machine learning.")

# Entrada do usuário
ticker = st.text_input("Digite o ticker da ação (ex: PETR4.SA, AAPL):", "AAPL")
periodo = st.selectbox("Período:", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y"])

# Baixar dados
df = yf.download(ticker, period=periodo)
df.reset_index(inplace=True)

# Verificação
if df.empty:
    st.error("Não foi possível carregar os dados. Verifique o ticker.")
    st.stop()

# ======= Indicadores Financeiros =======
df["Retorno_Diário"] = df["Close"].pct_change()
retorno_medio = df["Retorno_Diário"].mean()
retorno_anual = (1 + retorno_medio) ** 252 - 1
vol_anual = df["Retorno_Diário"].std() * np.sqrt(252)
sharpe = (retorno_anual - 0.05) / vol_anual
if df["Close"].dropna().empty:
    z_score = 0.0
else:
    z_score = float((df["Close"].iloc[-1] - df["Close"].dropna().mean()) / df["Close"].dropna().std())


# Mostrar métricas
col1, col2, col3, col4 = st.columns(4)
col1.metric("📅 Retorno Médio Diário", f"{retorno_medio*100:.2f}%")
col2.metric("📈 Retorno Anualizado", f"{retorno_anual*100:.2f}%")
col3.metric("💥 Volatilidade Anualizada", f"{vol_anual*100:.2f}%")
col4.metric("⚖️ Índice de Sharpe", f"{sharpe:.2f}")

st.metric("📊 Z-Score Atual", f"{z_score:.2f}")

# ======= Candlestick =======
st.subheader("Gráfico de Candlestick com Média Móvel")
fig_candle = go.Figure(data=[go.Candlestick(
    x=df['Date'], open=df['Open'], high=df['High'],
    low=df['Low'], close=df['Close'], name='Candlestick'
)])
fig_candle.add_trace(go.Scatter(
    x=df['Date'], y=df['Close'].rolling(window=20).mean(),
    mode='lines', name='Média Móvel (20 dias)', line=dict(color='orange')
))
fig_candle.update_layout(
    template="plotly_dark", height=500,
    xaxis_title="Data", yaxis_title="Preço"
)
st.plotly_chart(fig_candle, use_container_width=True)

# ======= Volume =======
if periodo in ["1mo", "3mo", "6mo", "1y"]:
    st.subheader("Volume de Negociações")
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Bar(x=df["Date"], y=df["Volume"], name="Volume", marker_color='lightblue'))
    fig_vol.update_layout(template="plotly_dark", height=300)
    st.plotly_chart(fig_vol, use_container_width=True)

# ======= Machine Learning =======
st.subheader("🧠 Previsão de Preços com Machine Learning")

# Preparar dados
df["Dias"] = np.arange(len(df))
X = df[["Dias"]]
y = df["Close"]

# Modelos
model_lr = LinearRegression()
model_rf = RandomForestRegressor(n_estimators=200, random_state=42)

model_lr.fit(X, y)
model_rf.fit(X, y)

# Previsões
dias_futuros = np.arange(len(df), len(df) + 30).reshape(-1, 1)
pred_lr = model_lr.predict(dias_futuros)
pred_rf = model_rf.predict(dias_futuros)

# Avaliação
y_pred_lr = model_lr.predict(X)
r2_lr = r2_score(y, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y, y_pred_lr))

y_pred_rf = model_rf.predict(X)
r2_rf = r2_score(y, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y, y_pred_rf))

col1, col2 = st.columns(2)
col1.metric("R² Linear Regression", f"{r2_lr:.3f}")
col1.metric("RMSE Linear", f"{rmse_lr:.3f}")
col2.metric("R² Random Forest", f"{r2_rf:.3f}")
col2.metric("RMSE Random Forest", f"{rmse_rf:.3f}")

# ======= Gráfico Previsões =======
st.subheader("🔮 Preço Real vs Previsão (30 dias)")
fig_pred = go.Figure()
fig_pred.add_trace(go.Scatter(x=df["Date"], y=df["Close"], mode="lines", name="Preço Real"))
fig_pred.add_trace(go.Scatter(
    x=pd.date_range(df["Date"].iloc[-1], periods=31, freq="D")[1:], 
    y=pred_lr, mode="lines", name="Regressão Linear", line=dict(color="orange")
))
fig_pred.add_trace(go.Scatter(
    x=pd.date_range(df["Date"].iloc[-1], periods=31, freq="D")[1:], 
    y=pred_rf, mode="lines", name="Random Forest", line=dict(color="green")
))
fig_pred.update_layout(template="plotly_dark", height=500)
st.plotly_chart(fig_pred, use_container_width=True)

st.caption("⚠️ Este modelo é experimental. Não constitui recomendação de investimento.")

