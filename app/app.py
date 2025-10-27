import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title="☝🤓 Market Analysis", layout="wide")
st.title("📊 Análise de Mercado com Regressão, Indicadores Estatísticos e Retornos")

# Entrada de dados
tickers = st.text_input("Digite os tickers separados por vírgula:", "PETR4.SA, VALE3.SA, ITUB4.SA")
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
        n_days = st.number_input(
            label="Número de dias para previsão",
            min_value=1,     # mínimo 1 dia
            max_value=365,   # máximo 1 ano
            value=5,         # valor padrão
            step=1
        )
        future_preds = []
        df_future = df.copy()
        for i in range(n_days):
            last_row = df_future.iloc[-1]
            next_features = pd.DataFrame({
                'Return': [0],
                'SMA5': [df_future['Close'].rolling(5).mean().iloc[-1]],
                'SMA10': [df_future['Close'].rolling(10).mean().iloc[-1]],
                'EMA5': [df_future['Close'].ewm(span=5, adjust=False).mean().iloc[-1]],
                'EMA10': [df_future['Close'].ewm(span=10, adjust=False).mean().iloc[-1]],
                'Volatility5': [df_future['Return'].rolling(5).std().iloc[-1]],
                'Volatility10': [df_future['Return'].rolling(10).std().iloc[-1]],'Close_minus_SMA5': [last_row['Close'] - df_future['Close'].rolling(5).mean().iloc[-1]],
                'Close_minus_SMA10': [last_row['Close'] - df_future['Close'].rolling(10).mean().iloc[-1]],
            })
            pred = model.predict(next_features)[0]
            future_preds.append(pred)
    
            # Atualiza DataFrame para o próximo passo
            new_row = pd.DataFrame({'Close': [pred]})
            df_future = pd.concat([df_future, new_row], ignore_index=True)
         
         
        st.dataframe(df_future.tail(n_days))
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

    except Exception as e:
        st.error(f"Erro ao processar {ticker}: {e}")





























