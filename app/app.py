import streamlit as st
import certifi, ssl, os
os.environ["SSL_CERT_FILE"] = certifi.where()
ssl._create_default_https_context = ssl._create_unverified_context
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="☝🤓 Market analysis", layout="wide")
st.title("📊 Downloader de Dados Financeiros + Dashboard com Regressão e Estatísticas")

# Input tickers
tickers_input = st.text_input("Digite o(s) ticker(s) separados por vírgula (ex: PETR4.SA, AAPL, BTC-USD):")
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

# Período máximo (todos os dados)
PERIOD = "max"

if st.button("Baixar dados"):
    if not tickers:
        st.warning("Digite pelo menos um ticker.")
    else:
        st.write(f"Baixando dados de **{', '.join(tickers)}**...")
        df = yf.download(tickers, group_by='ticker', auto_adjust=True, period=PERIOD)
        
        # Padroniza MultiIndex
        if not isinstance(df.columns, pd.MultiIndex):
            df.columns = pd.MultiIndex.from_product([[tickers[0]], df.columns])
        
        df = df.dropna(how="all")
        if df.empty:
            st.error("⚠️ Nenhum dado encontrado.")
        else:
            st.success(f"✅ Dados baixados! Total de tickers: {len(tickers)}")
            final_df = pd.DataFrame()
            
            for ticker in tickers:
                if ticker not in df.columns.get_level_values(0):
                    st.warning(f"{ticker} não encontrado. Pulando.")
                    continue
                
                ticker_df = df[ticker].copy().reset_index()
                
                if 'Close' not in ticker_df.columns:
                    st.warning(f"{ticker} não possui coluna 'Close'. Pulando.")
                    continue
                
                # Remove linhas sem preço
                ticker_df = ticker_df.dropna(subset=['Close'])
                if ticker_df.empty:
                    st.warning(f"{ticker} não possui dados válidos. Pulando.")
                    continue
                
                # Cálculos
                ticker_df[f"{ticker}_Return"] = ticker_df['Close'].pct_change()
                ticker_df[f"{ticker}_SMA20"] = ticker_df['Close'].rolling(20).mean()
                ticker_df[f"{ticker}_EMA20"] = ticker_df['Close'].ewm(span=20, adjust=False).mean()
                ticker_df[f"{ticker}_Volatility"] = ticker_df[f"{ticker}_Return"].rolling(20).std()
                
                ret_series = ticker_df[f"{ticker}_Return"]
                mean_daily = ret_series.mean()
                annual_return = (1 + mean_daily)**252 - 1
                annual_vol = ret_series.std() * np.sqrt(252)
                sharpe = (annual_return - 0.1) / annual_vol
                
                # Regressão linear
                X = np.arange(len(ticker_df)).reshape(-1,1)
                y = ticker_df['Close'].values.reshape(-1,1)
                model = LinearRegression()
                model.fit(X, y)
                ticker_df['Close_Pred'] = model.predict(X)
                ticker_df['Deviation'] = ticker_df['Close'] - ticker_df['Close_Pred']
                ticker_df['Deviation_Change'] = ticker_df['Deviation'].diff()
                
                # Estatísticas
                st.subheader(f"📈 Dashboard - {ticker}")
                st.write(f"**Retorno médio diário:** {mean_daily:.4f}")
                st.write(f"**Retorno anualizado:** {annual_return:.4f}")
                st.write(f"**Volatilidade anualizada:** {annual_vol:.4f}")
                st.write(f"**Índice de Sharpe:** {sharpe:.4f}")
                
                # Candlestick
                fig_candle = go.Figure(data=[go.Candlestick(
                    x=ticker_df['Date'],
                    open=ticker_df['Open'],
                    high=ticker_df['High'],
                    low=ticker_df['Low'],
                    close=ticker_df['Close'],
                    increasing_line_color='green',
                    decreasing_line_color='red'
                )])
                fig_candle.update_layout(
                    title=f"{ticker} - Candlestick",
                    xaxis_rangeslider_visible=False,
                    plot_bgcolor='rgb(20,20,20)',
                    paper_bgcolor='rgb(20,20,20)',
                    font=dict(color='white')
                )
                st.plotly_chart(fig_candle, use_container_width=True)
                
                # Preço + regressão
                fig_reg = go.Figure()
                fig_reg.add_trace(go.Scatter(
                    x=ticker_df['Date'],
                    y=ticker_df['Close'],
                    mode='lines',
                    name='Close Real'
                ))
                fig_reg.add_trace(go.Scatter(
                    x=ticker_df['Date'],
                    y=ticker_df['Close_Pred'],
                    mode='lines',
                    name='Regressão Linear',
                    line=dict(dash='dash', color='orange')
                ))
                fig_reg.update_layout(
                    title=f"{ticker} - Preço + Regressão",
                    plot_bgcolor='rgb(20,20,20)',
                    paper_bgcolor='rgb(20,20,20)',
                    font=dict(color='white')
                )
                st.plotly_chart(fig_reg, use_container_width=True)
                
                # Volatilidade
                fig_vol = go.Figure()
                fig_vol.add_trace(go.Scatter(
                    x=ticker_df['Date'],
                    y=ticker_df[f"{ticker}_Volatility"],
                    mode='lines',
                    name='Volatilidade 20d'
                ))
                fig_vol.update_layout(
                    title=f"{ticker} - Volatilidade",
                    plot_bgcolor='rgb(20,20,20)',
                    paper_bgcolor='rgb(20,20,20)',
                    font=dict(color='white')
                )
                st.plotly_chart(fig_vol, use_container_width=True)
                
                # Desvio da regressão
                fig_dev = go.Figure()
                fig_dev.add_trace(go.Scatter(
                    x=ticker_df['Date'],
                    y=ticker_df['Deviation_Change'],
                    mode='lines',
                    name='Variação do Desvio'
                ))
                fig_dev.update_layout(
                    title=f"{ticker} - Variação da Distância do Preço até a Regressão",
                    plot_bgcolor='rgb(20,20,20)',
                    paper_bgcolor='rgb(20,20,20)',
                    font=dict(color='white')
                )
                st.plotly_chart(fig_dev, use_container_width=True)
                
                # Adiciona ao DataFrame final
                cols_to_add = [col for col in ticker_df.columns if ticker in col or col in ['Close_Pred','Deviation','Deviation_Change']]
                if final_df.empty:
                    final_df = ticker_df[cols_to_add].copy()
                else:
                    final_df = final_df.join(ticker_df[cols_to_add], how='outer')
            
            # Mostrar e baixar
            st.subheader("📄 Dados finais com cálculos por ticker")
            st.dataframe(final_df.tail())
            nome_arquivo = "tickers_dados_compostos.csv"
            final_df.to_csv(nome_arquivo)
            st.download_button(
                label="📥 Baixar CSV",
                data=final_df.to_csv().encode('utf-8'),
                file_name=nome_arquivo,
                mime="text/csv"
            )

