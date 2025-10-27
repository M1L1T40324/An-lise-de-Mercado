import streamlit as st
import certifi, ssl, os
os.environ["SSL_CERT_FILE"] = certifi.where()
ssl._create_default_https_context = ssl._create_unverified_context
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title="☝🤓 Market analysis", layout="wide")
st.title("📊 Downloader de Dados Financeiros + Dashboard com Regressão e Desvios")

# Input
tickers_input = st.text_input("Digite o(s) ticker(s) (ex: PETR4.SA, AAPL, BTC-USD):")
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

if st.button("Baixar dados"):
    if tickers:
        st.write(f"Baixando dados de **{', '.join(tickers)}**...")
        df = yf.download(tickers, group_by='ticker', auto_adjust=True, period="max")
        df = df.dropna(how="all")

        if df.empty:
            st.error("⚠️ Nenhum dado encontrado. Verifique os tickers digitados.")
        else:
            st.success(f"✅ Dados baixados! Total de tickers: {len(tickers)}")
            final_df = pd.DataFrame()

            for ticker in tickers:
                try:
                    ticker_df = df[ticker].copy()
                except:
                    ticker_df = df.copy()

                # Cálculos básicos
                ticker_df[f"{ticker}_Return"] = ticker_df["Close"].pct_change()
                ticker_df[f"{ticker}_SMA20"] = ticker_df["Close"].rolling(window=20).mean()
                ticker_df[f"{ticker}_EMA20"] = ticker_df["Close"].ewm(span=20, adjust=False).mean()
                ticker_df[f"{ticker}_Volatility"] = ticker_df[f"{ticker}_Return"].rolling(window=20).std()

                # Métricas globais
                ret_series = ticker_df[f"{ticker}_Return"].dropna()
                mean_daily = ret_series.mean()
                annual_return = (1 + mean_daily)**252 - 1
                annual_vol = ret_series.std() * np.sqrt(252)
                sharpe = (annual_return - 0.1) / annual_vol if annual_vol != 0 else np.nan

                # Regressão Linear
                X = np.arange(len(ticker_df)).reshape(-1, 1)
                y = ticker_df["Close"].values.reshape(-1, 1)
                model = LinearRegression()
                model.fit(X, y)
                ticker_df["Close_Pred"] = model.predict(X)

                # Métricas de previsão
                r2 = model.score(X, y)
                rmse = np.sqrt(mean_squared_error(y, ticker_df["Close_Pred"]))
                mape = np.mean(np.abs((y - ticker_df["Close_Pred"]) / y)) * 100
                ticker_df["Deviation"] = ticker_df["Close"] - ticker_df["Close_Pred"]
                z_score = ticker_df["Deviation"].mean() / ticker_df["Deviation"].std()

                # Resultados principais
                st.subheader(f"📈 Dashboard - {ticker}")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Retorno médio diário", f"{mean_daily:.4f}")
                    st.metric("Volatilidade anualizada", f"{annual_vol:.4f}")
                    st.metric("Z-Score médio", f"{z_score:.4f}")
                with col2:
                    st.metric("Retorno anualizado", f"{annual_return:.4f}")
                    st.metric("Índice de Sharpe", f"{sharpe:.4f}")
                    st.metric("R² da regressão", f"{r2:.4f}")
                with col3:
                    st.metric("RMSE (erro médio quadrático)", f"{rmse:.4f}")
                    st.metric("MAPE (%)", f"{mape:.2f}%")

                # Gráfico 1 - Candlestick
                fig_candle = go.Figure(data=[go.Candlestick(
                    x=ticker_df.index,
                    open=ticker_df["Open"],
                    high=ticker_df["High"],
                    low=ticker_df["Low"],
                    close=ticker_df["Close"],
                    increasing_line_color="green",
                    decreasing_line_color="red"
                )])
                fig_candle.update_layout(
                    title=f"{ticker} - Candlestick",
                    xaxis_rangeslider_visible=False,
                    plot_bgcolor="rgb(20,20,20)",
                    paper_bgcolor="rgb(20,20,20)",
                    font=dict(color="white")
                )
                st.plotly_chart(fig_candle, use_container_width=True)

                # Gráfico 2 - Preço + Regressão Linear
                fig_reg = go.Figure()
                fig_reg.add_trace(go.Scatter(
                    x=ticker_df.index,
                    y=ticker_df["Close"],
                    mode="lines",
                    name="Close Real"
                ))
                fig_reg.add_trace(go.Scatter(
                    x=ticker_df.index,
                    y=ticker_df["Close_Pred"],
                    mode="lines",
                    name="Regressão Linear",
                    line=dict(dash="dash", color="orange")
                ))
                fig_reg.update_layout(
                    title=f"{ticker} - Preço + Regressão Linear",
                    plot_bgcolor="rgb(20,20,20)",
                    paper_bgcolor="rgb(20,20,20)",
                    font=dict(color="white")
                )
                st.plotly_chart(fig_reg, use_container_width=True)

                # Gráfico 3 - Volatilidade
                fig_vol = go.Figure()
                fig_vol.add_trace(go.Scatter(
                    x=ticker_df.index,
                    y=ticker_df[f"{ticker}_Volatility"],
                    mode="lines",
                    name="Volatilidade (20 dias)"
                ))
                fig_vol.update_layout(
                    title=f"{ticker} - Volatilidade",
                    plot_bgcolor="rgb(20,20,20)",
                    paper_bgcolor="rgb(20,20,20)",
                    font=dict(color="white")
                )
                st.plotly_chart(fig_vol, use_container_width=True)

                # Gráfico 4 - Variação da distância do preço à regressão
                ticker_df["Deviation_Change"] = ticker_df["Deviation"].diff()
                fig_dev = go.Figure()
                fig_dev.add_trace(go.Scatter(
                    x=ticker_df.index,
                    y=ticker_df["Deviation_Change"],
                    mode="lines",
                    name="Variação do Desvio"
                ))
                fig_dev.update_layout(
                    title=f"{ticker} - Variação da Distância do Preço até a Regressão",
                    plot_bgcolor="rgb(20,20,20)",
                    paper_bgcolor="rgb(20,20,20)",
                    font=dict(color="white")
                )
                st.plotly_chart(fig_dev, use_container_width=True)

                # Junta ao DataFrame final
                cols_to_add = [col for col in ticker_df.columns if ticker in col or col in ["Close_Pred", "Deviation", "Deviation_Change"]]
                if final_df.empty:
                    final_df = ticker_df[cols_to_add].copy()
                else:
                    final_df = final_df.join(ticker_df[cols_to_add], how="outer")

            # Mostrar DataFrame final e botão para download
            st.subheader("📄 Dados finais com cálculos por ticker")
            st.dataframe(final_df.tail())
            nome_arquivo = "tickers_dados_compostos.csv"
            final_df.to_csv(nome_arquivo)
            st.download_button(
                label="📥 Baixar CSV",
                data=final_df.to_csv().encode("utf-8"),
                file_name=nome_arquivo,
                mime="text/csv"
            )
    else:
        st.warning("Digite pelo menos um ticker.")
