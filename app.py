def analyze_ticker(ticker, horizon, strategy):

    data = load_data(ticker)

    if data is None:
        return None

    close = data["Close"]

    log_ret = compute_log_returns(close)

    mu = estimate_drift(log_ret)

    sigma = estimate_volatility_ewma(log_ret)

    # VOLATILIDADE ANUAL
    vol_annual = sigma * np.sqrt(252)

    if vol_annual < 0.20:
        return None

    # TENDÊNCIA
    ma100 = close.rolling(100).mean().iloc[-1]

    if close.iloc[-1] < ma100:
        return None

    # LIQUIDEZ
    if data["Volume"].mean() < 200000:
        return None

    atr = compute_atr(data)

    atr_pct = atr / close.iloc[-1]

    best, df_all = optimize_tp_sl(
        mu,
        sigma,
        horizon,
        strategy,
        atr_pct
    )

    if best is None:
        return None

    tp = best["TP"]
    sl = best["SL"]

    pnl = simulate_trade_distribution(
        mu,
        sigma,
        tp,
        sl,
        horizon
    )

    EV, sharpe, max_dd, cvar, skew = risk_metrics(pnl)

    prob_win = (pnl > 0).mean()

    return {
        "Ticker":ticker,
        "Score":best["Score"],
        "ProbWin":prob_win,
        "EV":EV,
        "Sharpe":sharpe,
        "TP":tp,
        "SL":sl,
        "Kelly":best["Kelly"],
        "VolAnual":vol_annual
    }
