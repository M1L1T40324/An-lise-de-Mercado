import streamlit as st
import pandas as pd
import yfinance as yf

st.set_page_config(page_title="Análise de Tickers", layout="wide")

st.title("📈 Coletor de Dados de Tickers")

# =========================
# INPUT DE DADOS
# =========================

st.subheader("Entrada de Tickers")

texto_input = st.text_input(
    "Digite os tickers separados por vírgula (ex: PETR4.SA, VALE3.SA)"
)

arquivo = st.file_uploader("Ou envie um arquivo .txt com tickers")

# =========================
# FUNÇÃO DE PROCESSAMENTO
# =========================

def processar_tickers(texto, arquivo):
    tickers = []

    if texto:
        tickers.extend([t.strip().upper() for t in texto.split(",") if t.strip()])

    if arquivo:
        conteudo = arquivo.read().decode("utf-8")
        tickers.extend([t.strip().upper() for t in conteudo.split(",") if t.strip()])

    # remove duplicatas
    tickers = list(set(tickers))

    return tickers

# =========================
# DOWNLOAD DOS DADOS
# =========================

def baixar_dados(tickers):
    dados = {}

    for ticker in tickers:
        try:
            df = yf.download(ticker, progress=False)

            if df.empty:
                st.warning(f"{ticker} não retornou dados.")
                continue

            df["Ticker"] = ticker
            dados[ticker] = df

        except Exception as e:
            st.error(f"Erro ao baixar {ticker}: {e}")

    return dados

# =========================
# EXECUÇÃO
# =========================

if st.button("Baixar Dados"):

    tickers = processar_tickers(texto_input, arquivo)

    if not tickers:
        st.warning("Nenhum ticker válido fornecido.")
    else:
        st.write("Tickers processados:", tickers)

        dados = baixar_dados(tickers)

        if dados:
            df_consolidado = pd.concat(dados.values())
            df_consolidado.reset_index(inplace=True)

            st.success("Dados baixados com sucesso!")

            st.dataframe(df_consolidado)

            st.session_state["dados"] = df_consolidado
        else:
            st.error("Nenhum dado foi carregado.")
