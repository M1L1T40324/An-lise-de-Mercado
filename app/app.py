import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go

# Configuração inicial
st.set_page_config(page_title="Previsão de Mercado", layout="wide")
st.title("📊 Previsão de Mercado com Modelagem Não Linear")

# Entradas do usuário
ticker = st.text_input("Digite o código do ativo (ex: PETZ3.SA, AAPL, BTC-USD):", "PETZ3.SA", key="ticker_input")
ndays = st.number_input("Quantos dias à frente deseja prever?", min_value=1, max_value=60, value=7, key="dias_previsao")

# Função para carregar dados
def load_data(ticker):
    df = yf.download(ticker, period="2y")
    if df.empty:
        raise ValueError(f"Nenhum dado encontrado para {ticker}")
    df["SMA20"] = df["Close"].rolling(window=20).mean()
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["Volatility"] = df["Close"].rolling(window=20).std()
    df.dropna(inplace=True)
    return df

# Função principal
def process_stock(ticker, ndays):
    try:
        df = load_data(ticker)
        st.write(f"✅ Dados carregados para **{ticker}** ({len(df)} registros).")

        # Features e target
        features = ["SMA20", "EMA20", "Volatility"]
        X = df[features]
        y = df["Close"]

        # Escalonamento e transformação polinomial
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        poly = PolynomialFeatures(degree=3)
        X_poly = poly.fit_transform(X_scaled)

        # Treino
        model = LinearRegression()
        model.fit(X_poly, y)

        # Predição histórica
        df["Predicted"] = model.predict(X_poly)

        # Avaliação
        mse = mean_squared_error(y, df["Predicted"])
        st.write(f"📏 **Erro Quadrático Médio (MSE):** {mse:.6f}")

        # Previsão futura com ruído estável
        future_predictions = []
        future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=ndays)
        last_features = df[features].iloc[-1].values.reshape(1, -1)

        current_features = last_features.copy()
        base_volatility = df["Volatility"].iloc[-1]

        for _ in range(ndays):
            scaled = scaler.transform(current_features)
            poly_features = poly.transform(scaled)
            next_price = model.predict(poly_features)[0]

            # Adiciona ruído leve com base na volatilidade histórica
            noise = np.random.normal(0, base_volatility * 0.02)
            next_price_noisy = next_price + noise

            future_predictions.append(next_price_noisy)

            # Atualiza features mantendo a coerência dimensional
            sma = (current_features[0, 0] * 19 + next_price_noisy) / 20
            ema = 0.1 * next_price_noisy + 0.9 * current_features[0, 1]
            vol = np.std(future_predictions[-20:]) if len(future_predictions) >= 20 else base_volatility
            current_features = np.array([[sma, ema, vol]])

        # Monta DataFrame das previsões futuras
        future_df = pd.DataFrame({"Data": future_dates, "Previsão": future_predictions})
        st.subheader("📆 Previsões Futuras")
        st.dataframe(future_df.tail())

        # GRÁFICO
        fig = go.Figure()

        # Preço real
        fig.add_trace(go.Scatter(
            x=df.index, y=df["Close"],
            mode="lines", name="Preço Real",
            line=dict(color="royalblue", width=2)
        ))

        # Predição histórica
        fig.add_trace(go.Scatter(
            x=df.index, y=df["Predicted"],
            mode="lines", name="Predição Histórica",
            line=dict(color="orange", width=2)
        ))

        # Previsão futura
        fig.add_trace(go.Scatter(
            x=future_dates, y=future_predictions,
            mode="lines+markers", name="Previsão Futura",
            line=dict(color="limegreen", dash="dot", width=3)
        ))

        fig.update_layout(
            title=f"Previsão de {ticker}",
            xaxis_title="Data",
            yaxis_title="Preço (R$)",
            template="plotly_dark",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"⚠️ Erro ao processar {ticker}: {e}")

# Executa o app
if ticker:
    process_stock(ticker, ndays)
