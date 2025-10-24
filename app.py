# app.py
"""
Investidor App Inteligente (Streamlit)
- Salve como app.py
- Execute: streamlit run app.py
Dependências: streamlit pandas numpy yfinance plotly scikit-learn
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="Investidor Inteligente", layout="wide", page_icon="💹")

# -------------------------
# UTILITÁRIOS / INDICADORES
# -------------------------
def calcula_indicadores(df):
    df = df.copy()
    df['Retorno'] = df['Adj Close'].pct_change()
    df['LogRetorno'] = np.log(df['Adj Close'] / df['Adj Close'].shift(1))
    df['MA5'] = df['Adj Close'].rolling(5).mean()
    df['MA20'] = df['Adj Close'].rolling(20).mean()
    df['Volatilidade'] = df['Retorno'].rolling(20).std()
    delta = df['Adj Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    ema12 = df['Adj Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Adj Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    return df

def projeta_preco(df):
    df = df.dropna()
    if len(df) < 5:
        return np.array([df['Adj Close'].iloc[-1]] * len(df))
    X = np.arange(len(df)).reshape(-1, 1)
    y = df['Adj Close'].values
    modelo = LinearRegression().fit(X, y)
    return modelo.predict(X)

def calcula_score(df):
    score = 0
    if df['MA5'].iloc[-1] > df['MA20'].iloc[-1]:
        score += 1
    if df['RSI'].iloc[-1] < 70:
        score += 1
    if df['MACD'].iloc[-1] > df['Signal'].iloc[-1]:
        score += 1
    if df['Volatilidade'].iloc[-1] < 0.03:
        score += 1
    return score

def motivos_join(lista):
    return "; ".join(lista) if lista else "—"

def sinal_inteligente(df):
    """
    Regras interpretáveis:
    - RSI >= 80: sobrecompra forte
    - RSI >= 70: sobrecompra
    - MACD < Signal: tendência enfraquecendo
    - Drawdown >= 10%: queda desde pico de 30 dias
    Recomendações: Venda Total / Venda Parcial (50%) / Mover Lucro p/ Reserva / Manter
    """
    rsi = float(df['RSI'].iloc[-1])
    macd = float(df['MACD'].iloc[-1])
    signal = float(df['Signal'].iloc[-1])
    preco = float(df['Adj Close'].iloc[-1])
    recent_max = float(df['Adj Close'].rolling(30, min_periods=1).max().iloc[-1])
    drawdown = (recent_max - preco) / recent_max if recent_max > 0 else 0

    motivos = []
    if rsi >= 80:
        motivos.append("RSI ≥ 80 (sobrecompra forte)")
    elif rsi >= 70:
        motivos.append("RSI ≥ 70 (sobrecompra)")
    if macd < signal:
        motivos.append("MACD < Signal (tendência enfraquecendo)")
    if drawdown >= 0.10:
        motivos.append("Queda ≥10% do pico dos últimos 30d")

    # prioridade
    if macd < signal and rsi >= 70:
        recomendacao = "Venda Total"
    elif macd < signal:
        recomendacao = "Venda Parcial (50%)"
    elif rsi >= 75:
        recomendacao = "Venda Parcial (50%)"
    elif drawdown >= 0.10:
        recomendacao = "Avaliar - possível stop ou venda parcial"
    else:
        recomendacao = "Manter"

    sinais = {'RSI': rsi, 'MACD': macd, 'Signal': signal, 'Drawdown30': drawdown}
    return recomendacao, motivos_join(motivos), sinais

# -------------------------
# ESTADO DO APP (session_state)
# -------------------------
if 'positions' not in st.session_state:
    st.session_state['positions'] = {}  # ticker -> {buy_price, shares, cost_basis, capital_allocated}
if 'reserve' not in st.session_state:
    st.session_state['reserve'] = 0.0
if 'trade_history' not in st.session_state:
    st.session_state['trade_history'] = []

# -------------------------
# SIDEBAR - CONFIGURAÇÕES
# -------------------------
st.sidebar.title("Configurações")
tickers_input = st.sidebar.text_input("Tickers (ex: GMAT3.SA,MGLU3.SA,LREN3.SA,VVAR3.SA)")
capital_total = st.sidebar.number_input("Capital Total (R$)", min_value=1.0, value=1000.0, step=100.0)
btn_init = st.sidebar.button("Inicializar carteira sugerida (simulação)")

# -------------------------
# FUNÇÃO: inicializar carteira simulada
# -------------------------
def init_portfolio(resultados):
    """
    Usa preço atual como preço de compra e capital alocado para criar posição simulada.
    Se já há posição para o ticker, soma com custo médio.
    """
    for res in resultados:
        ticker = res['Ação']
        capital_alocado = res['Capital_Alocado']
        preco = res['Preco_Atual']
        if preco <= 0:
            continue
        shares = capital_alocado / preco
        pos = st.session_state['positions'].get(ticker)
        if pos:
            # recalcular custo médio
            total_cost = pos['cost_basis'] * pos['shares'] + capital_alocado
            total_shares = pos['shares'] + shares
            pos['shares'] = total_shares
            pos['cost_basis'] = total_cost / total_shares
            pos['capital_allocated'] = pos['cost_basis'] * total_shares
        else:
            st.session_state['positions'][ticker] = {
                'buy_price': preco,
                'shares': shares,
                'cost_basis': preco,
                'capital_allocated': capital_alocado
            }

# -------------------------
# COLETAR DADOS E MONTAR RESULTADOS
# -------------------------
st.title("💹 Investidor Inteligente — Simulação de Gestão de Lucros")
if not tickers_input:
    st.info("Digite tickers na barra lateral (ex: GMAT3.SA,MGLU3.SA,LREN3.SA,VVAR3.SA) e clique em 'Inicializar carteira sugerida' ou aguarde a análise.")
else:
    tickers = [t.strip() for t in tickers_input.split(",") if t.strip()]
    resultados = []

    # coletar e analisar
    for ticker in tickers:
        df = yf.download(ticker, period="6mo", interval="1d", progress=False)
        if df.empty:
            st.warning(f"Ticker {ticker} não retornou dados.")
            continue
        df = calcula_indicadores(df)
        df['Preco_Previsto'] = projeta_preco(df)
        score = calcula_score(df)
        perc_capital = score / 4 * 0.4 + 0.1  # alocação sugerida (10% - ~50%)
        capital_alocado = capital_total * perc_capital
        preco_atual = float(df['Adj Close'].iloc[-1])
        preco_previsto = float(df['Preco_Previsto'].iloc[-1])
        rendimento_previsto = (preco_previsto - preco_atual) / preco_atual * capital_alocado

        resultados.append({
            "Ação": ticker,
            "Score": score,
            "Perc_Alocacao": perc_capital,
            "Capital_Alocado": capital_alocado,
            "Preco_Atual": preco_atual,
            "Preco_Previsto": preco_previsto,
            "Rendimento_Previsto": rendimento_previsto,
            "DataFrame": df
        })

    # botão para inicializar
    if btn_init:
        init_portfolio(resultados)
        st.success("Carteira inicializada (simulação). Verifique 'Carteira & Ações'.")

    # -------------------------
    # TABS
    # -------------------------
    tab1, tab2, tab3, tab4 = st.tabs(["Análise", "Dashboard", "Projeção", "Carteira & Ações"])

    # -------------------------
    # TAB 1: ANÁLISE E RECOMENDAÇÕES
    # -------------------------
    with tab1:
        st.header("Análises e Recomendações Inteligentes")
        st.write("Sinais técnicos: RSI, MACD, drawdown (30d). Recomendações geradas por regras interpretáveis.")
        for res in resultados:
            ticker = res['Ação']
            df = res['DataFrame']
            recomendacao, motivo, sinais = sinal_inteligente(df)

            st.subheader(f"{ticker}  — Preço atual: R$ {res['Preco_Atual']:.2f}")
            c1, c2, c3, c4 = st.columns([1, 2, 3, 2])
            c1.metric("Score", res['Score'])
            c2.markdown(f"**Recomendação:** {recomendacao}")
            c3.markdown(f"**Motivos:** {motivo}")
            c4.markdown(f"**RSI:** {sinais['RSI']:.1f}  •  **MACD:** {sinais['MACD']:.4f}")

            # Ações simuladas
            a1, a2, a3, a4 = st.columns(4)
            if a1.button(f"Venda Parcial 50% — {ticker}"):
                pos = st.session_state['positions'].get(ticker)
                if pos:
                    shares_sell = pos['shares'] * 0.5
                    valor_recebido = shares_sell * res['Preco_Atual']
                    pos['shares'] -= shares_sell
                    pos['capital_allocated'] = pos['shares'] * pos['cost_basis']
                    st.session_state['reserve'] += valor_recebido
                    st.session_state['trade_history'].append({
                        'timestamp': datetime.now().isoformat(),
                        'ticker': ticker,
                        'action': 'Venda Parcial 50%',
                        'shares': shares_sell,
                        'price': res['Preco_Atual'],
                        'value': valor_recebido
                    })
                    st.success(f"Venda parcial simulada: R$ {valor_recebido:.2f} para reserva.")
                else:
                    st.warning("Posição simulada não encontrada. Inicialize a carteira primeiro.")

            if a2.button(f"Venda Total — {ticker}"):
                pos = st.session_state['positions'].get(ticker)
                if pos:
                    shares_sell = pos['shares']
                    valor_recebido = shares_sell * res['Preco_Atual']
                    del st.session_state['positions'][ticker]
                    st.session_state['reserve'] += valor_recebido
                    st.session_state['trade_history'].append({
                        'timestamp': datetime.now().isoformat(),
                        'ticker': ticker,
                        'action': 'Venda Total',
                        'shares': shares_sell,
                        'price': res['Preco_Atual'],
                        'value': valor_recebido
                    })
                    st.success(f"Venda total simulada: R$ {valor_recebido:.2f} para reserva.")
                else:
                    st.warning("Posição simulada não encontrada.")

            if a3.button(f"Mover Lucro p/ Reserva — {ticker}"):
                pos = st.session_state['positions'].get(ticker)
                if pos:
                    market_value = pos['shares'] * res['Preco_Atual']
                    cost = pos['shares'] * pos['cost_basis']
                    lucros = market_value - cost
                    if lucros <= 0:
                        st.warning("Sem lucro realizável.")
                    else:
                        shares_to_sell = lucros / res['Preco_Atual']
                        shares_to_sell = min(shares_to_sell, pos['shares'])
                        valor_recebido = shares_to_sell * res['Preco_Atual']
                        pos['shares'] -= shares_to_sell
                        pos['capital_allocated'] = pos['shares'] * pos['cost_basis']
                        st.session_state['reserve'] += valor_recebido
                        st.session_state['trade_history'].append({
                            'timestamp': datetime.now().isoformat(),
                            'ticker': ticker,
                            'action': 'Mover Lucro p/ Reserva',
                            'shares': shares_to_sell,
                            'price': res['Preco_Atual'],
                            'value': valor_recebido
                        })
                        st.success(f"Lucro de R$ {valor_recebido:.2f} movido para reserva.")
                else:
                    st.warning("Posição simulada não encontrada.")

            if a4.button(f"Manter — {ticker}"):
                st.info("Decisão: manter posição simulada (nenhuma ação).")
            st.markdown("---")

    # -------------------------
    # TAB 2: DASHBOARD (gráficos interativos)
    # -------------------------
    with tab2:
        st.header("Dashboard — Preço, Médias e Projeção")
        colors = ["#3498db", "#2ecc71", "#9b59b6", "#e67e22"]
        for i, res in enumerate(resultados):
            df = res['DataFrame']
            color = colors[i % len(colors)]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df['Adj Close'], mode='lines', name='Preço', line=dict(color=color)))
            fig.add_trace(go.Scatter(x=df.index, y=df['MA5'], mode='lines', name='MA5', line=dict(color=color, dash='dash')))
            fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], mode='lines', name='MA20', line=dict(color=color, dash='dot')))
            fig.add_trace(go.Scatter(x=df.index, y=df['Preco_Previsto'], mode='lines', name='Projeção', line=dict(color=color, dash='dashdot')))
            # seta de tendência
            last_idx = df.index[-1]
            last_proj = res['Preco_Previsto']
            last_price = res['Preco_Atual']
            arrow_color = 'lime' if last_proj > last_price else 'orange'
            fig.add_annotation(x=last_idx, y=last_proj,
                               text='↑' if last_proj > last_price else '↓',
                               showarrow=False, font=dict(color=arrow_color, size=18))
            fig.update_layout(template='plotly_dark', height=360, margin=dict(l=10, r=10, t=30, b=10))
            st.subheader(res['Ação'])
            st.plotly_chart(fig, use_container_width=True)

    # -------------------------
    # TAB 3: PROJEÇÃO
    # -------------------------
    with tab3:
        st.header("Projeção de Capital (estado atual)")
        # Valor atual da carteira
        total_positions_value = 0.0
        for ticker, pos in st.session_state['positions'].items():
            # pegar preço atual de resultados se presente
            cur_price = None
            for r in resultados:
                if r['Ação'] == ticker:
                    cur_price = r['Preco_Atual']
                    break
            if cur_price is None:
                dd = yf.download(ticker, period="5d", interval="1d", progress=False)
                cur_price = float(dd['Adj Close'].iloc[-1]) if not dd.empty else pos['cost_basis']
            total_positions_value += pos['shares'] * cur_price

        total_capital_now = total_positions_value + st.session_state['reserve']
        st.metric("Capital total atual (posições + reserva)", f"R$ {total_capital_now:,.2f}")

        # estimativa de retorno mensal média (a partir dos ativos)
        returns = []
        for r in resultados:
            df = r['DataFrame']
            mean_daily = df['Retorno'].mean()
            if pd.isna(mean_daily):
                continue
            monthly_est = (1 + mean_daily) ** 21 - 1
            returns.append(monthly_est)
        estimated_monthly_return = np.nanmean(returns) if returns else 0.005

        st.write(f"Estimativa média de retorno mensal (histórico dos ativos): {estimated_monthly_return:.4f} (~{estimated_monthly_return*100:.2f}%/mês)")

        periods = [3, 6, 12]  # meses
        col1, col2, col3 = st.columns(3)
        for idx, p in enumerate(periods):
            projected = total_capital_now * ((1 + estimated_monthly_return) ** p)
            if idx == 0:
                col1.metric(f"{p} meses", f"R$ {projected:,.2f}")
            elif idx == 1:
                col2.metric(f"{p} meses", f"R$ {projected:,.2f}")
            else:
                col3.metric(f"{p} meses", f"R$ {projected:,.2f}")

        st.markdown("**Projeção detalhada por ativo (composto simples usando rendimento previsto do ativo):**")
        for r in resultados:
            st.write(f"- {r['Ação']} — Capital Alocado: R$ {r['Capital_Alocado']:.2f}")
            for p in periods:
                # evitar divisão por zero; usar rendimento_previsto como taxa efetiva (pouco ideal mas simples)
                try:
                    rate_month = r['Rendimento_Previsto'] / max(r['Capital_Alocado'], 1e-6)
                    projected_asset = r['Capital_Alocado'] * ((1 + rate_month) ** p)
                except:
                    projected_asset = r['Capital_Alocado']
                st.write(f"   • {p} meses: R$ {projected_asset:,.2f}")

    # -------------------------
    # TAB 4: CARTEIRA, RESERVA E HISTÓRICO
    # -------------------------
    with tab4:
        st.header("Carteira Simulada — Posições Atuais & Reserva")
        st.subheader("Posições simuladas")
        if st.session_state['positions']:
            pos_df = []
            for ticker, pos in st.session_state['positions'].items():
                pos_df.append({
                    "Ticker": ticker,
                    "Shares": pos['shares'],
                    "Cost Basis (R$)": pos['cost_basis'],
                    "Valor Alocado (R$)": pos['capital_allocated'],
                    "Market Value (R$)": pos['shares'] * (next((r['Preco_Atual'] for r in resultados if r['Ação'] == ticker), pos['cost_basis']))
                })
            pos_df = pd.DataFrame(pos_df)
            st.dataframe(pos_df)
        else:
            st.info("Nenhuma posição simulada. Clique em 'Inicializar carteira sugerida' na barra lateral.")

        st.subheader("Reserva (lucros realizados)")
        st.metric("Reserva em caixa", f"R$ {st.session_state['reserve']:.2f}")

        st.subheader("Histórico de operações simuladas")
        if st.session_state['trade_history']:
            hist_df = pd.DataFrame(st.session_state['trade_history']).sort_values('timestamp', ascending=False)
            st.dataframe(hist_df)
        else:
            st.info("Nenhuma operação simulada registrada.")

        colA, colB = st.columns(2)
        with colA:
            if st.button("Reinvestir Reserva proporcionalmente às alocações atuais"):
                reserve = st.session_state['reserve']
                if reserve <= 0:
                    st.warning("Reserva vazia.")
                else:
                    total_alloc = sum([pos['capital_allocated'] for pos in st.session_state['positions'].values()]) or 0
                    if total_alloc == 0:
                        st.warning("Sem posições para reinvestir. Inicialize carteira primeiro.")
                    else:
                        for ticker, pos in st.session_state['positions'].items():
                            frac = pos['capital_allocated'] / total_alloc
                            invest = reserve * frac
                            shares_new = invest / pos['cost_basis'] if pos['cost_basis'] > 0 else 0
                            pos['shares'] += shares_new
                            pos['capital_allocated'] = pos['shares'] * pos['cost_basis']
                        st.session_state['trade_history'].append({
                            'timestamp': datetime.now().isoformat(),
                            'ticker': 'MULTI',
                            'action': 'Reinvest Reserve',
                            'shares': None,
                            'price': None,
                            'value': reserve
                        })
                        st.session_state['reserve'] = 0.0
                        st.success("Reserva reinvestida proporcionalmente.")
        with colB:
            if st.button("Sacar Reserva (simulação)"):
                st.session_state['trade_history'].append({
                    'timestamp': datetime.now().isoformat(),
                    'ticker': 'CASH',
                    'action': 'Sacar Reserva',
                    'shares': None,
                    'price': None,
                    'value': st.session_state['reserve']
                })
                st.session_state['reserve'] = 0.0
                st.success("Reserva sacada (simulada).")

# -------------------------
# RODAPÉ / AVISO
# -------------------------
st.markdown("---")
st.caption(
    "⚠️ Este sistema é uma SIMULAÇÃO educativa. Não envia ordens para corretoras. "
    "Use as recomendações como apoio à decisão e confirme operações na sua corretora. "
    "A lógica de sinais é heurística: combine com análise fundamentalista e gestão de risco."
)
