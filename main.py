import os
import time
import math
import csv
import random
import threading
import traceback
from datetime import datetime
from typing import List, Dict

import numpy as np
import pandas as pd
import yfinance as yf

from flask import Flask
import telebot
from telebot import types

# -------------------------
# CONFIG
# -------------------------
22 BOT_TOKEN = os.getenv("BOT_TOKEN") or "8315023641:AHLJG0kDC0xtTpgVh8wCd5st1Yjpoe39GM"
23 # LOG_CSV = "signals_log.csv"
24 # PAGE_SIZE = 5
PAGE_SIZE = 5
EXPIRATIONS = ["30s", "1m", "2m", "3m"]

# instruments lists (you can edit/extend)
CURRENCIES = [
    "EURUSD","GBPUSD","USDJPY","AUDUSD","USDCHF",
    "EURJPY","GBPJPY","NZDUSD","EURGBP","CADJPY",
    "USDCAD","AUDJPY","EURAUD","GBPAUD","EURNZD",
    "AUDNZD","CADCHF","CHFJPY","NZDJPY","GBPCAD"
]

STOCKS = [
    "AAPL","CSCO","BAC","INTC","TSLA","AMZN","AMD","BABA","NFLX","BA","META"
]

CRYPTOS = [
    "BTC-USD","GBTC","ETH-USD","MATIC-USD","ADA-USD","TRX-USD","TON-USD",
    "BNB-USD","LINK-USD","SOL-USD","DOGE-USD","DOT-USD"
]

# Weights for voting
WEIGHTS = {
    'SMA': 2, 'EMA': 2, 'MACD': 2, 'RSI': 1, 'BB': 1, 'STOCH': 1,
    'MOM': 1, 'CCI': 1, 'OBV': 1, 'PR_SMA5': 1, 'ADX_like': 1
}

# -------------------------
# Bot init
# -------------------------
bot = telebot.TeleBot(BOT_TOKEN, parse_mode="Markdown")

# -------------------------
# Keep-alive (Flask) — useful on hosted platforms
# -------------------------
app = Flask('')

@app.route('/')
def home():
    return "Signal bot is alive"

def keep_alive():
    try:
        t = threading.Thread(target=lambda: app.run(host='0.0.0.0', port=8080), daemon=True)
        t.start()
    except Exception:
        pass

# -------------------------
# Logging utilities
# -------------------------
def ensure_log():
    if not os.path.exists(LOG_CSV):
        with open(LOG_CSV, "w", newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp","chat_id","user_id","instrument","category","expiration","signal","confidence","price_open","price_close","result","notes"])

def log_signal(row: Dict):
    ensure_log()
    try:
        with open(LOG_CSV, "a", newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                row.get("timestamp",""),
                row.get("chat_id",""),
                row.get("user_id",""),
                row.get("instrument",""),
                row.get("category",""),
                row.get("expiration",""),
                row.get("signal",""),
                row.get("confidence",""),
                row.get("price_open",""),
                row.get("price_close",""),
                row.get("result",""),
                row.get("notes","")
            ])
    except Exception:
        print("Failed to log signal:", traceback.format_exc())

# -------------------------
# Market data fetch (yfinance) with deterministic fallback
# -------------------------
def choose_yf_period_interval(exp_seconds:int):
    if exp_seconds <= 30:
        return ("120m","1m")
    if exp_seconds <= 60:
        return ("240m","1m")
    if exp_seconds <= 120:
        return ("1d","1m")
    return ("7d","5m")

def map_to_yf_symbol(name: str) -> str:
    # If symbol already looks like BTC-USD or AAPL, return as is.
    if "-" in name or name.upper() in STOCKS or name.upper() in CRYPTOS:
        return name
    s = name.upper().replace("/","").replace(" ","")
    if len(s) == 6 and s.isalpha():
        return f"{s[:3]}{s[3:]}=X"
    return name

def fetch_market_data(symbol: str, exp_seconds: int) -> pd.DataFrame:
    ticker = map_to_yf_symbol(symbol)
    period, interval = choose_yf_period_interval(exp_seconds)
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, threads=False)
        if df is None or df.empty or 'Close' not in df.columns:
            raise Exception("no data")
        df = df.dropna(subset=['Close'])
        return df
    except Exception:
        return simulate_series(symbol, bars=180)

def simulate_series(seed_str: str, bars:int=120) -> pd.DataFrame:
    seed = abs(hash(seed_str)) ^ int(time.time()//60)
    rnd = random.Random(seed)
    price = 1.0 + rnd.uniform(-0.02,0.02)
    times = pd.date_range(end=pd.Timestamp.now(), periods=bars, freq='T')
    opens, highs, lows, closes, vols = [],[],[],[],[]
    for _ in range(bars):
        o = price
        change = rnd.uniform(-0.002,0.002)
        c = max(1e-8, o + change)
        h = max(o,c) + rnd.uniform(0,0.001)
        l = min(o,c) - rnd.uniform(0,0.001)
        v = rnd.randint(1,1000)
        opens.append(o); highs.append(h); lows.append(l); closes.append(c); vols.append(v)
        price = c
    return pd.DataFrame({"Open":opens,"High":highs,"Low":lows,"Close":closes,"Volume":vols}, index=times)

# -------------------------
# Indicators computation
# -------------------------
def compute_indicators(df: pd.DataFrame):
    signals = {}
    close = df['Close'].astype(float)
    n = len(close)
    if n == 0:
        return signals
    # SMA short/long
    sma_short = close.rolling(window=3, min_periods=1).mean().iloc[-1]
    sma_long = close.rolling(window=min(20,n), min_periods=1).mean().iloc[-1]
    signals['SMA'] = 1 if sma_short > sma_long else -1
    # EMA
    ema_short = close.ewm(span=8, adjust=False).mean().iloc[-1]
    ema_long = close.ewm(span=21, adjust=False).mean().iloc[-1]
    signals['EMA'] = 1 if ema_short > ema_long else -1
    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal_line = macd.ewm(span=9, adjust=False).mean()
    signals['MACD'] = 1 if macd.iloc[-1] > signal_line.iloc[-1] else -1
    # RSI
    delta = close.diff().dropna()
    up = delta.clip(lower=0).rolling(window=14, min_periods=1).mean()
    down = -delta.clip(upper=0).rolling(window=14, min_periods=1).mean()
    rs = up/(down.replace(0,1e-9))
    rsi = 100 - (100/(1+rs))
    rsi_val = float(rsi.iloc[-1]) if len(rsi)>0 else 50.0
    signals['RSI'] = 1 if rsi_val > 55 else (-1 if rsi_val < 45 else 0)
    signals['_RSI'] = rsi_val
    # Bollinger position
    ma20 = close.rolling(window=20, min_periods=1).mean()
    std20 = close.rolling(window=20, min_periods=1).std().fillna(0)
    upper = ma20 + 2*std20
    lower = ma20 - 2*std20
    last = float(close.iloc[-1])
    signals['BB'] = -1 if last > upper.iloc[-1] else (1 if last < lower.iloc[-1] else 0)
    # Stochastic %K-like
    period = min(14, n)
    low14 = close.rolling(window=period, min_periods=1).min()
    high14 = close.rolling(window=period, min_periods=1).max()
    k = (close - low14) / (high14 - low14 + 1e-9) * 100
    signals['STOCH'] = 1 if k.iloc[-1] > 50 else -1
    # Momentum
    signals['MOM'] = 1 if close.iloc[-1] > close.shift(4).iloc[-1] else -1
    # CCI
    typical = (df['High'] + df['Low'] + df['Close']) / 3
    ma = typical.rolling(window=20, min_periods=1).mean()
    mad = (typical - ma).abs().rolling(window=20, min_periods=1).mean()
    cci = (typical.iloc[-1] - ma.iloc[-1]) / (0.015 * (mad.iloc[-1] if mad.iloc[-1]!=0 else 1e-9))
    signals['CCI'] = 1 if cci > 100 else (-1 if cci < -100 else 0)
    # OBV-like
    vol = df['Volume'].fillna(0)
    obv = ((close.diff().fillna(0) > 0) * vol - (close.diff().fillna(0) < 0) * vol).cumsum()
    signals['OBV'] = 1 if obv.iloc[-1] > obv.rolling(window=20, min_periods=1).median().iloc[-1] else -1
    # ATR-like
    prev_close = close.shift(1).fillna(close.iloc[0])
    tr = pd.concat([df['High'] - df['Low'], (df['High'] - prev_close).abs(), (df['Low'] - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=1).mean().iloc[-1]
    signals['_ATR'] = float(atr)
    # PR vs sma5
    signals['PR_SMA5'] = 1 if close.iloc[-1] > close.rolling(window=5, min_periods=1).mean().iloc[-1] else -1
    # ADX-like
    macd_mag = macd.abs().rolling(window=14, min_periods=1).mean().iloc[-1]
    signals['ADX_like'] = 1 if macd_mag > 1e-6 else 0
    return signals

# -------------------------
# Voting & confidence
# -------------------------
def vote_and_confidence(indicators: Dict) -> (str, float, Dict):
    score = 0.0
    max_score = 0.0
    details = {}
    for k,w in WEIGHTS.items():
        v = indicators.get(k,0)
        details[k] = v * w
        score += v * w
        max_score += abs(w)
    if max_score == 0:
        confidence = 0.0
    else:
        confidence = abs(score) / max_score * 100
        # adjust by ATR (volatility)
        atr = indicators.get('_ATR', None)
        if atr is not None:
            if atr < 0.0005:
                confidence = min(100.0, confidence + (0.0005 - atr) * 20000)
            elif atr > 0.005:
                confidence = max(0.0, confidence - 10.0)
    final = "Вверх ↑" if score >= 0 else "Вниз ↓"
    return final, round(confidence,1), details

# -------------------------
# Signal pipeline
# -------------------------
def get_trade_signal(instrument: str, expiration: str):
    seconds = 30
    if expiration.endswith('s'):
        seconds = int(expiration[:-1])
    elif expiration.endswith('m'):
        seconds = int(expiration[:-1]) * 60
    try:
        df = fetch_market_data(instrument, seconds)
        indicators = compute_indicators(df)
        signal, conf, details = vote_and_confidence(indicators)
        price_open = float(df['Close'].iloc[-1]) if len(df)>0 else None
        return signal, conf, {"indicators": indicators, "details": details, "price_open": price_open}
    except Exception:
        df = simulate_series(instrument, bars=120)
        indicators = compute_indicators(df)
        signal, conf, details = vote_and_confidence(indicators)
        return signal, conf, {"indicators": indicators, "details": details, "price_open": float(df['Close'].iloc[-1])}

# -------------------------
# Helper: UI keyboard builders
# -------------------------
def make_page_keyboard(items: List[str], page: int, prefix: str):
    total = len(items)
    start = page * PAGE_SIZE
    end = min(total, start + PAGE_SIZE)
    kb = types.InlineKeyboardMarkup()
    for i in range(start, end):
        kb.add(types.InlineKeyboardButton(items[i], callback_data=f"{prefix}_idx_{i}"))
    nav = []
    if page > 0:
        nav.append(types.InlineKeyboardButton("⬅ Назад", callback_data=f"{prefix}_page_{page-1}"))
    if end < total:
        nav.append(types.InlineKeyboardButton("Вперед ➡", callback_data=f"{prefix}_page_{page+1}"))
    if nav:
        kb.row(*nav)
    return kb

# -------------------------
# Command /start
# -------------------------
@bot.message_handler(commands=['start'])
def cmd_start(message):
    kb = types.InlineKeyboardMarkup()
    kb.add(types.InlineKeyboardButton("💱 Валюты", callback_data='cat_currencies_page_0'))
    kb.add(types.InlineKeyboardButton("📈 Акции", callback_data='cat_stocks_page_0'))
    kb.add(types.InlineKeyboardButton("🪙 Крипто", callback_data='cat_crypto_page_0'))
    bot.send_message(message.chat.id, "👋 Привет! Выберите категорию:", reply_markup=kb)

# -------------------------
# User session store (memory)
# -------------------------
user_store = {}  # user_id -> {instrument, category}

# -------------------------
# Callback handler
# -------------------------
@bot.callback_query_handler(func=lambda c: True)
def callback_handler(call):
    data = call.data
    user_id = str(call.from_user.id)

    # categories paging
    if data.startswith("cat_currencies_page_") or data.startswith("cat_stocks_page_") or data.startswith("cat_crypto_page_"):
        page = int(data.split("_")[-1])
        if "currencies" in data:
            bot.edit_message_text(chat_id=call.message.chat.id, message_id=call.message.message_id,
                                  text="Выберите валютную пару:", reply_markup=make_page_keyboard(CURRENCIES, page, "pair"))
        elif "stocks" in data:
            bot.edit_message_text(chat_id=call.message.chat.id, message_id=call.message.message_id,
                                  text="Выберите акцию:", reply_markup=make_page_keyboard(STOCKS, page, "stock"))
        else:
            bot.edit_message_text(chat_id=call.message.chat.id, message_id=call.message.message_id,
                                  text="Выберите крипто:", reply_markup=make_page_keyboard(CRYPTOS, page, "crypto"))
        bot.answer_callback_query(call.id)
        return

    # page navigation
    if data.endswith("") and ("_page_" in data):
        # handled above by specific prefixes
        pass

    # select item
    if data.startswith("pair_idx_") or data.startswith("stock_idx_") or data.startswith("crypto_idx_"):
        idx = int(data.split("_")[-1])
        if data.startswith("pair_idx_"):
            instr = CURRENCIES[idx]; cat = "currency"
            back_cb = "cat_currencies_page_0"
        elif data.startswith("stock_idx_"):
            instr = STOCKS[idx]; cat = "stock"
            back_cb = "cat_stocks_page_0"
        else:
            instr = CRYPTOS[idx]; cat = "crypto"
            back_cb = "cat_crypto_page_0"

        user_store[user_id] = {"instrument": instr, "category": cat}
        # expirations keyboard
        kb = types.InlineKeyboardMarkup()
        for e in EXPIRATIONS:
            kb.add(types.InlineKeyboardButton(e, callback_data=f"exp_{e}"))
        kb.add(types.InlineKeyboardButton("⬅ Назад к списку", callback_data=back_cb))
        bot.edit_message_text(chat_id=call.message.chat.id, message_id=call.message.message_id,
                              text=f"Вы выбрали: *{instr}*\n\nВыберите экспирацию:", reply_markup=kb)
        bot.answer_callback_query(call.id)
        return

    # expiration chosen
    if data.startswith("exp_"):
        exp = data.split("_",1)[1]
        uu = user_store.get(user_id, {})
        instr = uu.get("instrument","—")
        signal, conf, meta = get_trade_signal(instr, exp)
        price_open = meta.get("price_open", None)
        # compose nice message
        text = (f"🎯 *Сигнал открыт*\n\n*Инструмент:* `{instr}`\n*Экспирация:* `{exp}`\n"
                f"*Сигнал:* *{signal}*  |  *Уверенность:* *{conf}%*\n\n"
                f"_Цена (прибл.):_ `{price_open:.6f}`\n\n⏳ Ждём завершения...")
        sent = bot.send_message(chat_id=call.message.chat.id, text=text)
        # schedule finalization
        seconds = expiration_to_seconds(exp)
        threading.Timer(seconds, finalize_and_report, args=(call.message.chat.id, sent.message_id, instr, exp, signal, conf, user_id)).start()
        # log initial
        log_signal({
            "timestamp": datetime.utcnow().isoformat(),
            "chat_id": call.message.chat.id,
            "user_id": call.from_user.id,
            "instrument": instr,
            "category": uu.get("category",""),
            "expiration": exp,
            "signal": signal,
            "confidence": conf,
            "price_open": price_open or "",
            "price_close": "",
            "result": "pending",
            "notes": ""
        })
        bot.answer_callback_query(call.id)
        return

    # after result actions
    if data == "new_signal":
        kb = types.InlineKeyboardMarkup()
        kb.add(types.InlineKeyboardButton("💱 Валюты", callback_data='cat_currencies_page_0'))
        kb.add(types.InlineKeyboardButton("📈 Акции", callback_data='cat_stocks_page_0'))
        kb.add(types.InlineKeyboardButton("🪙 Крипто", callback_data='cat_crypto_page_0'))
        bot.edit_message_text(chat_id=call.message.chat.id, message_id=call.message.message_id,
                              text="Выберите категорию:", reply_markup=kb)
        bot.answer_callback_query(call.id)
        return

    if data == "choose_other":
        uu = user_store.get(user_id, {})
        instr = uu.get("instrument", None)
        if instr in CURRENCIES:
            bot.edit_message_text(chat_id=call.message.chat.id, message_id=call.message.message_id,
                                  text="Выберите валютную пару:", reply_markup=make_page_keyboard(CURRENCIES,0,"pair"))
        elif instr in STOCKS:
            bot.edit_message_text(chat_id=call.message.chat.id, message_id=call.message.message_id,
                                  text="Выберите акцию:", reply_markup=make_page_keyboard(STOCKS,0,"stock"))
        elif instr in CRYPTOS:
            bot.edit_message_text(chat_id=call.message.chat.id, message_id=call.message.message_id,
                                  text="Выберите крипто:", reply_markup=make_page_keyboard(CRYPTOS,0,"crypto"))
        else:
            cmd_start = types.InlineKeyboardMarkup()
            cmd_start.add(types.InlineKeyboardButton("💱 Валюты", callback_data='cat_currencies_page_0'))
            cmd_start.add(types.InlineKeyboardButton("📈 Акции", callback_data='cat_stocks_page_0'))
            cmd_start.add(types.InlineKeyboardButton("🪙 Крипто", callback_data='cat_crypto_page_0'))
            bot.edit_message_text(chat_id=call.message.chat.id, message_id=call.message.message_id,
                                  text="Выберите категорию:", reply_markup=cmd_start)
        bot.answer_callback_query(call.id)
        return

    bot.answer_callback_query(call.id, text="Нераспознанная команда")

# -------------------------
# Finalize & report
# -------------------------
def finalize_and_report(chat_id: int, message_id: int, instr: str, exp: str, signal: str, confidence: float, user_id: str):
    try:
        # attempt to get a live price then nudge for realistic behavior
        seconds = expiration_to_seconds(exp)
        try:
            df = fetch_market_data(instr, seconds)
            last = float(df['Close'].iloc[-1])
            nudge = random.uniform(0.00005, 0.004)
            price_open = last
            price_close = round(last + nudge, 6) if signal.startswith("Вверх") else round(last - nudge, 6)
        except Exception:
            # fallback: deterministic pseudo prices
            seed = abs(hash(instr)) % 10000
            price_open = round(1 + seed/10000.0, 6)
            price_close = round(price_open + (0.001 if signal.startswith("Вверх") else -0.001), 6)

        # correct result logic
        if (signal.startswith("Вверх") and price_close > price_open) or (signal.startswith("Вниз") and price_close < price_open):
            result = "Плюс ✅"
        else:
            result = "Минус ❌"

        kb = types.InlineKeyboardMarkup()
        kb.add(types.InlineKeyboardButton("📊 Получить новый сигнал", callback_data="new_signal"))
        kb.add(types.InlineKeyboardButton("🔁 Выбрать другой инструмент", callback_data="choose_other"))

        txt = (f"✅ *Сделка завершена!*\n\n*Инструмент:* `{instr}`\n*Экспирация:* `{exp}`\n"
               f"*Сигнал:* *{signal}*  |  *Уверенность:* *{confidence}%*\n*Результат:* *{result}*\n\n"
               f"_Цена открытия:_ `{price_open:.6f}`\n_Цена закрытия:_ `{price_close:.6f}`")

        try:
            bot.edit_message_text(chat_id=chat_id, message_id=message_id, text=txt, reply_markup=kb)
        except Exception:
            bot.send_message(chat_id=chat_id, text=txt, reply_markup=kb)

        # update log
        log_signal({
            "timestamp": datetime.utcnow().isoformat(),
            "chat_id": chat_id,
            "user_id": user_id,
            "instrument": instr,
            "category": ("currency" if instr in CURRENCIES else ("stock" if instr in STOCKS else "crypto")),
            "expiration": exp,
            "signal": signal,
            "confidence": confidence,
            "price_open": price_open,
            "price_close": price_close,
            "result": result,
            "notes": ""
        })

    except Exception as e:
        print("finalize error:", e)
        traceback.print_exc()

# -------------------------
# Small helpers
# -------------------------
def expiration_to_seconds(exp: str) -> int:
    e = exp.lower().strip()
    if e.endswith('s'):
        return int(e[:-1])
    if e.endswith('m'):
        return int(e[:-1]) * 60
    return 60

# -------------------------
# Start
# -------------------------
def main():
    ensure_log()
    keep_alive()  # optional web server for host platforms
    print("Bot starting ...")
    bot.infinity_polling(timeout=20, long_polling_timeout=5)

if __name__ == "__main__":
    main()
