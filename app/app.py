import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import smtplib
from email.mime.text import MIMEText
import threading
import time

EMAIL_REMETENTE = "brandpeterr@gmail.com"
SENHA_EMAIL = "ishk nagl dgxu mmar"  # precisa ser senha de app do Gmail

def enviar_email(destinatario, assunto, mensagem):
    msg = MIMEText(mensagem)
    msg["Subject"] = assunto
    msg["From"] = EMAIL_REMETENTE
    msg["To"] = destinatario

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(EMAIL_REMETENTE, SENHA_EMAIL)
        server.send_message(msg)


st.sidebar.header("🔍 Monitoramento Automático")

ticker_input = st.sidebar.text_input("Ticker (ex: VALE3.SA)")
alvo_alta = st.sidebar.number_input("Alerta de ALTA acima de:", min_value=0.0, step=0.1)
alvo_baixa = st.sidebar.number_input("Alerta de QUEDA abaixo de:", min_value=0.0, step=0.1)
email_dest = st.sidebar.text_input("Email para alerta:")

if st.sidebar.button("Adicionar ao Monitoramento"):
    novo = pd.DataFrame([[ticker_input, alvo_alta, alvo_baixa, email_dest]],
                        columns=["ticker","alvo_alta","alvo_baixa","email"])

    try:
        df = pd.read_csv("watchlist.csv")
        df = pd.concat([df, novo], ignore_index=True)
    except:
        df = novo
    
    df.to_csv("watchlist.csv", index=False)
    st.sidebar.success(f"{ticker_input} adicionado ao monitoramento!")


st.set_page_config(page_title="☝🤓 AI Market Analysis", layout="wide")
st.title("📊 Análise de Mercado com Regressão, Indicadores Estatísticos e Retornos")

# Entrada de dados
tickers = st.text_input("Digite os tickers separados por vírgula:", "BEEF3.SA")
tickers = [t.strip().upper() for t in tickers.split(",") if t.strip()]

periodo = st.selectbox("Selecione o período:", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"])
intervalo = st.selectbox("Selecione o intervalo:", ["1d", "1wk", "1mo"])

# Baixar dados
data = yf.download(tickers, period=periodo, interval=intervalo, group_by='ticker', auto_adjust=True)

def corrigir_colunas(df, ticker):
    """Corrige colunas quando há MultiIndex (vários tickers)."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df[ticker].copy()
    return df

for ticker in tickers:
    st.subheader(f"📈 {ticker}")

    try:
        df = corrigir_colunas(data, ticker).dropna()

        # --- Cálculos Estatísticos ---
        df["Return"] = df["Close"].pct_change()
        mean_daily = df["Return"].mean()
        annual_return = (1 + mean_daily) ** 252 - 1
        annual_vol = df["Return"].std() * np.sqrt(252)
        sharpe = (annual_return - 0.1) / annual_vol if annual_vol != 0 else np.nan

        # --- Regressão Linear ---
        X = np.arange(len(df)).reshape(-1, 1)
        y = df["Close"].values
        model = LinearRegression().fit(X, y)
        df["Regressão"] = model.predict(X)

        # --- Distância e Z-Score ---
        df["Distância"] = df["Close"] - df["Regressão"]
        df["Z_Score"] = (df["Close"] - df["Close"].mean()) / df["Close"].std()

        # --- Métricas ---
        col1, col2 = st.columns(2)
        col1.metric("💰 Preço atual", f"R$ {df['Close'].iloc[-1]:.2f}")
        col2.metric("📉 Retorno médio diário", f"{mean_daily:.4%}")
        col2.metric("📈 Retorno anualizado", f"{annual_return:.2%}")

        col3, col4 = st.columns(2)
        col1.metric("📊 Volatilidade anualizada", f"{annual_vol:.2%}")
        col2.metric("⚖️ Índice de Sharpe", f"{sharpe:.2f}")
        col2.metric("🧭 Z-Score atual", f"{df['Z_Score'].iloc[-1]:.2f}")
        required_features = ['SMA20', 'EMA20', 'Volatility']
        existing_features = [f for f in required_features if f in df.columns]

        df['Return'] = df['Close'].pct_change()
        df['SMA5'] = df['Close'].rolling(5).mean()
        df['SMA10'] = df['Close'].rolling(10).mean()
        df['EMA5'] = df['Close'].ewm(span=5, adjust=False).mean()
        df['EMA10'] = df['Close'].ewm(span=10, adjust=False).mean()
        df['Volatility5'] = df['Return'].rolling(5).std()
        df['Volatility10'] = df['Return'].rolling(10).std()
        df['Close_minus_SMA5'] = df['Close'] - df['SMA5']
        df['Close_minus_SMA10'] = df['Close'] - df['SMA10']
        df.fillna(method='bfill', inplace=True)
        features = ['Return', 'SMA5', 'SMA10', 'EMA5', 'EMA10',
            'Volatility5', 'Volatility10',
            'Close_minus_SMA5', 'Close_minus_SMA10']
        X = df[features]
        y = df['Close']
        X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred)**0.5
        mse = mean_squared_error(y_test, y_pred)
        df.loc[X_test.index, 'Pred_Close'] = y_pred
        col1.metric("📉 Erro Real do Modelo ", f"(RMSE): R$ {rmse:.2f}")
        col1.metric("📉 Erro Médio do Modelo ", f"(MSE): R$ {mse:.2f}")
        X = np.arange(len(df)).reshape(-1, 1)
        y = df['Close'].values.reshape(-1, 1)
        model = LinearRegression()
        model.fit(X, y)
        n_days = st.number_input("Quantos dias você quer viajar no futuro? 🤖🔮",
                                 min_value=1,
                                 max_value=365,
                                 value=5,
                                 step=1,
                                 key=f"n_days_{ticker}"
                                )
        last_index = len(df)
        future_indices = np.arange(last_index, last_index + n_days).reshape(-1, 1)
        future_pred = model.predict(future_indices)
        future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=n_days)
        future_df = pd.DataFrame({'Close': future_pred.flatten()},
                                 index=future_dates)
        combined_df = pd.concat([df[['Close']], future_df])
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['Close'],
            mode='lines',
            name='Histórico'
        ))
        fig.add_trace(go.Scatter(
            x=future_df.index,
            y=future_df['Close'],
            mode='lines+markers',
            name='Previsão Futura',
            line=dict(dash='dash', color='orange')
        ))
        fig.update_layout(

            title="Histórico + Previsão Futura",
            xaxis_title="Data",
            yaxis_title="Preço",
            plot_bgcolor='rgb(20,20,20)',
            paper_bgcolor='rgb(20,20,20)',
            font=dict(color='white')
        )
        st.plotly_chart(fig, use_container_width=True)


        # --- Gráfico 1: Candle + Linha de Regressão ---
        fig1 = go.Figure()

        fig1.add_trace(go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Candlestick"
        ))

        fig1.add_trace(go.Scatter(
            x=df.index, y=df["Regressão"],
            mode="lines", name="Linha de Regressão",
            line=dict(color="orange", width=2)
        ))

        fig1.update_layout(
            title="Candlestick com Linha de Regressão",
            xaxis_title="Data", yaxis_title="Preço (R$)",
            template="plotly_dark",
            hovermode="x unified",
            xaxis_rangeslider_visible=False
        )
        st.plotly_chart(fig1, use_container_width=True)

        # --- Gráfico 2: Variação da Distância ---
        df["Distância_var"] = df["Distância"].diff()
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=df.index, y=df["Distância_var"],
            mode="lines", name="Variação da Distância"
        ))
        fig2.update_layout(
            title="📏 Variação da Distância entre o Preço e a Linha de Regressão",
            xaxis_title="Data", yaxis_title="Variação (R$)",
            template="plotly_dark", hovermode="x unified"
        )
        st.plotly_chart(fig2, use_container_width=True)

        # --- Gráfico 3: Volume (apenas até 1 ano) ---
        if periodo in ["1mo", "3mo", "6mo", "1y"]:
            fig3 = go.Figure()
            fig3.add_trace(go.Bar(
                x=df.index, y=df["Volume"],
                name="Volume", marker_color="blue"
            ))
            fig3.update_layout(
                title="📦 Volume de Negociações",
                xaxis_title="Data", yaxis_title="Volume",
                template="plotly_dark", hovermode="x unified"
            )
            st.plotly_chart(fig3, use_container_width=True)
            def sentinela():
                while True:
                    try:
                        df = pd.read_csv("watchlist.csv")
                        for _, row in df.iterrows():
                            ticker = row["ticker"]
                            preco = yf.Ticker(ticker).history(period="1m")["Close"][-1]
                            if preco >= row["alvo_alta"]:
                                enviar_email(row["email"], f"📈 ALTA: {ticker}", f"Preço atual: {preco}")
                                if preco <= row["alvo_baixa"]:
                                    enviar_email(row["email"], f"📉 BAIXA: {ticker}", f"Preço atual: {preco}")
                    except Exception as e:
                        print("Erro no monitoramento:", e)
                        time.sleep(300)  # 5 minutos
                        # Inicia o sentinela apenas uma vez
            if "sentinela_rodando" not in st.session_state:
                threading.Thread(target=sentinela, daemon=True).start()
                st.session_state["sentinela_rodando"] = True
    except Exception as e:
        st.error(f"Erro ao processar {ticker}: {e}")




