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

st.set_page_config(page_title="☝🤓 AI Market Analysis", layout="wide")
st.title("📊 Dashboard de ativos para tomada de decisão, com regressão e outras probabilidades 🎲")

# Entrada de dados
tickers = st.text_input("Digite os tickers separados por vírgula:", "BEEF3.SA")
tickers = [t.strip().upper() for t in tickers.split(",") if t.strip()]

periodo = st.selectbox("Selecione o período:", ["2y", "5y", "10y", "max"])
intervalo = st.selectbox("Selecione o intervalo:", ["1d", "1wk", "1mo"])

st.sidebar.title("Ajustes ⚙")

TP_PERCENT = st.sidebar.text_input("Porcentagem de TP alvo", "7")
SL_PERCENT = st.sidebar.text_input("Porcentagem de SL máximo", "5")

TP_PERCENT = 1 + float(TP_PERCENT) / 100.0
SL_PERCENT = 1 - float(SL_PERCENT) / 100.0

st.sidebar.title("Horizonte da Simulação ⏳")

min_days = st.sidebar.number_input(
    "Mínimo de dias do swing",
    min_value=1,
    value=10
)

max_days = st.sidebar.number_input(
    "Máximo de dias do swing",
    min_value=min_days,
    value=15
)

# Baixar dados
data = yf.download(tickers, period=periodo, interval=intervalo, group_by='ticker', auto_adjust=True)

def calcular_prob_retorno(df, tp, sl, min_days=10, max_days=15, n_sims=30000):
    returns = df["Return"].dropna()
    mu = returns.mean()
    sigma = returns.std()

    prob_tp = 0
    prob_sl = 0
    prob_neutro = 0

    preco_inicial = df["Close"].iloc[-1]

    for _ in range(n_sims):
        price = preco_inicial
        hit = None

        # sorteia o número de dias do swing
        horizon_days = np.random.randint(min_days, max_days + 1)

        for _ in range(horizon_days):
            ret = np.random.normal(mu, sigma)
            price *= (1 + ret)

            if price >= tp:
                hit = "TP"
                break
            if price <= sl:
                hit = "SL"
                break

        if hit == "TP":
            prob_tp += 1
        elif hit == "SL":
            prob_sl += 1
        else:
            prob_neutro += 1

    prob_tp /= n_sims
    prob_sl /= n_sims
    prob_neutro /= n_sims

    retorno_esperado = (
        prob_tp * (tp / preco_inicial - 1) +
        prob_sl * (sl / preco_inicial - 1)
    )

    return prob_tp, prob_sl, prob_neutro, retorno_esperado



def prob_mc_tp_before_sl(df, tp_price, sl_price, min_days=10, max_days=15, n_sims=50000):
    returns = df["Return"].dropna()
    mean = returns.mean()
    std = returns.std()

    wins = 0
    start_price = df["Close"].iloc[-1]

    for _ in range(n_sims):
        price = start_price
        horizon = np.random.randint(min_days, max_days + 1)

        for _ in range(horizon):
            ret = np.random.normal(mean, std)
            price *= (1 + ret)

            if price >= tp_price:
                wins += 1
                break
            if price <= sl_price:
                break

    return wins / n_sims

def corrigir_colunas(df, ticker):
    """Corrige colunas quando há MultiIndex (vários tickers)."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df[ticker].copy()
    return df

def buscar_t_melhores(tickers, data, horizon_days=5):
    resultados = []

    for ticker in tickers:
        try:
            df = corrigir_colunas(data, ticker).dropna()
            df["Return"] = df["Close"].pct_change()

            current = df["Close"].iloc[-1]
            tp = current * TP_PERCENT
            sl = current * SL_PERCENT


            prob_tp, prob_sl, _, ret_exp = calcular_prob_retorno(df, tp, sl, min_days, max_days)


            resultados.append((ticker, ret_exp, prob_tp, prob_sl))

        except:
            continue

    df_rank = pd.DataFrame(resultados,
                           columns=["Ticker", "Retorno Esperado", "Prob_TP", "Prob_SL"])
    df_rank.sort_values(by="Retorno Esperado", ascending=False, inplace=True)

    return df_rank.head(3)




for ticker in tickers:
    st.subheader(f"📈 {ticker}")

    try:
        df = corrigir_colunas(data, ticker).dropna()

        # --- Cálculos Estatísticos ---
        df["Return"] = df["Close"].pct_change()

        # --- Volatilidade ajustada ao horizonte de simulação ---
        horizon_mean = (min_days + max_days) / 2
        vol_horizon = df["Return"].std() * np.sqrt(horizon_mean)

        mean_daily = df["Return"].mean()
        annual_return = (1 + mean_daily) ** 252 - 1
        annual_vol = df["Return"].std() * np.sqrt(252)
        sharpe = (annual_return - 0.1) / annual_vol if annual_vol != 0 else np.nan
        std_daily = df["Return"].std()
        mean_daily = df["Return"].mean()
        horizon_mean = (min_days + max_days) / 2
        horizon_return = (1 + mean_daily) ** horizon_mean - 1

        annual_vol = df["Return"].std() * np.sqrt(252)
        Weekly_vol = df["Return"].std() * np.sqrt(5)
        cum = (1 + df["Return"]).cumprod()
        peak = cum.cummax()
        # --- Max Drawdown ---
        drawdown = (cum - peak) / peak
        max_drawdown = drawdown.min()
        df["Drawdown"] = drawdown

        var_95 = np.percentile(df["Return"], 5)
        cvar_95 = df["Return"][df["Return"] <= var_95].mean()
        var_99 = np.percentile(df["Return"], 1)
        cvar_99 = df["Return"][df["Return"] <= var_99].mean()
        df["var_95"] = float(var_95)
        df["cvar_95"] = float(cvar_95)
        df["var_99"] = float(var_99)
        df["cvar_99"] = float(cvar_99)
        delta = df["Close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        rs = gain.ewm(span=14).mean() / loss.ewm(span=14).mean()
        df["RSI"] = 100 - (100 / (1 + rs))
        ema12 = df["Close"].ewm(span=12).mean()
        ema26 = df["Close"].ewm(span=26).mean()
        df["MACD"] = ema12 - ema26
        df["Signal"] = df["MACD"].ewm(span=9).mean()
        df["BB_Mid"] = df["Close"].rolling(20).mean()
        df["BB_Std"] = df["Close"].rolling(20).std()
        df["BB_Upper"] = df["BB_Mid"] + 2 * df["BB_Std"]
        df["BB_Lower"] = df["BB_Mid"] - 2 * df["BB_Std"]
        df["SMA20"] = df["Close"].rolling(20).mean()
        df["EMA20"] = df["Close"].ewm(span=20).mean()

         # Defina TP e SL (exemplo 5% TP / -3% SL)
        tp_price = df["Close"].iloc[-1] * TP_PERCENT
        sl_price = df["Close"].iloc[-1] * SL_PERCENT

        current_price = df["Close"].iloc[-1]
        target_price = tp_price
        stop_price = sl_price

        prob_tp, prob_sl, prob_neutro, retorno_esp = calcular_prob_retorno(
            df, tp_price, sl_price, min_days, max_days
            )



        sharpe = (annual_return - 0.1) / annual_vol if annual_vol != 0 else np.nan
        # Simulação Monte Carlo
        target = 0.05   
        N = 50000      
        drift_horizon = mean_daily * horizon_mean
        sim = np.random.normal(horizon_return, vol_horizon, N)

        # Probabilidade
        prob = np.mean(sim >= target)

        # --- Regressão Linear ---
        X = np.arange(len(df)).reshape(-1, 1)
        y = df["Close"].values
        model = LinearRegression().fit(X, y)
        df["Regressão"] = model.predict(X)

        # --- Distância e Z-Score ---
        df["Distância"] = df["Close"] - df["Regressão"]
        df["Z_Score"] = (df["Close"] - df["Close"].mean()) / df["Close"].std()


        # --- Métricas ---
        col1, col2, col3 = st.columns(3)
        col1.metric("💰 Preço atual", f"R$ {df['Close'].iloc[-1]:.2f}")
        col2.metric("📈 Retorno médio no horizonte", f"{horizon_return:.2%}")
        col2.metric("📈 Retorno anualizado", f"{annual_return:.2%}")
        col3.metric("⚔️ Prob. TP antes de SL", f"{prob_tp:.2%}")
        col3.metric("💥 Prob. SL antes de TP", f"{prob_sl:.2%}")
        col3.metric("🎯 Retorno Esperado", f"{retorno_esp:.2%}")
        col3.metric("💀 Max Drawdown", f"{max_drawdown:.2%}")
    
        col3, col4 = st.columns(2)
        col1.metric("📊 Volatilidade no Horizonte", f"{vol_horizon:.2%}")
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
        #Gráfico DrawDown
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=df.index,
            y=df["Drawdown"],
            mode="lines",
            name="Drawdown"
            ))
        fig_dd.update_layout(
            title="💀 Histórico do Drawdown",
            xaxis_title="Data",
            yaxis_title="Drawdown",
            template="plotly_dark",
            hovermode="x unified"
            )
        st.plotly_chart(fig_dd, use_container_width=True)

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








