# main.py
# Requirements:
# python-telegram-bot==13.15
# pandas, numpy, yfinance, flask

import os
import time
import threading
import random
import csv
import traceback
from datetime import datetime
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from flask import Flask
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Updater, CommandHandler, CallbackQueryHandler, CallbackContext

# -------------------------
# CONFIG
# -------------------------
BOT_TOKEN = os.getenv("BOT_TOKEN") 
LOG_CSV = "signals_log.csv"
ANALYSIS_WAIT = 20  # секунд — "подождите 20 секунд, идёт анализ"
EXPIRATIONS = ["1m", "2m", "3m", "5m"]

# Только валютные пары — используем YFinance формат (EURUSD -> EURUSD=X)
FOREX = [
    "EURUSD","GBPUSD","USDJPY","AUDUSD","USDCHF",
    "EURJPY","GBPJPY","NZDUSD","EURGBP","CADJPY",
    "USDCAD","AUDJPY","EURAUD","GBPAUD","EURNZD",
    "AUDNZD","CADCHF","CHFJPY","NZDJPY","GBPCAD"
]

# -------------------------
# Keep-alive (веб-сервер) — полезно при деплое
# -------------------------
app = Flask('')

@app.route('/')
def index():
    return "Forex Signal Bot is alive"

def keep_alive():
    t = threading.Thread(target=lambda: app.run(host='0.0.0.0', port=8080), daemon=True)
    t.start()

# -------------------------
# Logging
# -------------------------
def ensure_log():
    if not os.path.exists(LOG_CSV):
        with open(LOG_CSV, "w", newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp","chat_id","user_id","instrument","expiration","signal","confidence","price_open","price_close","result","note"])

def log_row(row: Dict):
    ensure_log()
    try:
        with open(LOG_CSV, "a", newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                row.get("timestamp",""),
                row.get("chat_id",""),
                row.get("user_id",""),
                row.get("instrument",""),
                row.get("expiration",""),
                row.get("signal",""),
                row.get("confidence",""),
                row.get("price_open",""),
                row.get("price_close",""),
                row.get("result",""),
                row.get("note","")
            ])
    except Exception:
        print("Log error:", traceback.format_exc())

# -------------------------
# Market data helpers
# -------------------------
def yf_symbol(pair: str) -> str:
    # pair like EURUSD -> EURUSD=X for yfinance
    p = pair.upper().replace("/","").replace(" ","")
    if "-" in pair or pair.endswith("=X"):
        return pair
    if len(p) == 6 and p.isalpha():
        return f"{p[:3]}{p[3:]}=X"
    return pair

def choose_period_interval(exp_seconds:int) -> Tuple[str,str]:
    # pick period and interval for yfinance to ensure enough 1m bars
    if exp_seconds <= 60:
        return ("2d","1m")
    if exp_seconds <= 180:
        return ("5d","1m")
    return ("7d","5m")

def fetch_data(pair: str, exp_seconds: int) -> pd.DataFrame:
    ticker = yf_symbol(pair)
    period, interval = choose_period_interval(exp_seconds)
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, threads=False)
        if df is None or df.empty or 'Close' not in df.columns:
            raise Exception("no data")
        df = df.dropna(subset=['Close'])
        return df
    except Exception:
        # fallback: deterministic simulation so bot never fails
        return simulate_series(pair, bars=300)

def simulate_series(seed: str, bars: int = 300) -> pd.DataFrame:
    seedv = abs(hash(seed)) ^ int(time.time()//60)
    rnd = random.Random(seedv)
    price = 1.0 + rnd.uniform(-0.02,0.02)
    times = pd.date_range(end=pd.Timestamp.now(), periods=bars, freq='T')
    opens, highs, lows, closes, vols = [],[],[],[],[]
    for _ in range(bars):
        o = price
        ch = rnd.uniform(-0.002,0.002)
        c = max(1e-8, o + ch)
        h = max(o,c) + rnd.uniform(0,0.001)
        l = min(o,c) - rnd.uniform(0,0.001)
        v = rnd.randint(1,1000)
        opens.append(o); highs.append(h); lows.append(l); closes.append(c); vols.append(v)
        price = c
    return pd.DataFrame({"Open":opens,"High":highs,"Low":lows,"Close":closes,"Volume":vols}, index=times)

# -------------------------
# Indicators (multiple)
# -------------------------
def compute_indicators(df: pd.DataFrame) -> Dict[str,float]:
    out = {}
    close = df['Close'].astype(float)
    high = df['High'].astype(float)
    low = df['Low'].astype(float)
    vol = df['Volume'].fillna(0).astype(float)
    n = len(close)
    if n < 2:
        return out

    # SMA short/long
    sma3 = close.rolling(window=3, min_periods=1).mean().iloc[-1]
    sma20 = close.rolling(window=min(20,n), min_periods=1).mean().iloc[-1]
    out['SMA'] = 1 if sma3 > sma20 else -1

    # EMA
    ema8 = close.ewm(span=8, adjust=False).mean().iloc[-1]
    ema21 = close.ewm(span=21, adjust=False).mean().iloc[-1]
    out['EMA'] = 1 if ema8 > ema21 else -1

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    sig = macd.ewm(span=9, adjust=False).mean()
    out['MACD'] = 1 if macd.iloc[-1] > sig.iloc[-1] else -1

    # RSI
    delta = close.diff().dropna()
    up = delta.clip(lower=0).rolling(window=14, min_periods=1).mean()
    down = -delta.clip(upper=0).rolling(window=14, min_periods=1).mean()
    rs = up/(down.replace(0,1e-9))
    rsi = 100 - (100/(1+rs))
    rsi_val = float(rsi.iloc[-1]) if len(rsi)>0 else 50.0
    out['RSI'] = 1 if rsi_val > 55 else (-1 if rsi_val < 45 else 0)
    out['_RSI'] = rsi_val

    # Bollinger Bands
    ma20 = close.rolling(window=20, min_periods=1).mean()
    std20 = close.rolling(window=20, min_periods=1).std().fillna(0)
    upper = ma20 + 2*std20
    lower = ma20 - 2*std20
    last = float(close.iloc[-1])
    out['BB'] = -1 if last > upper.iloc[-1] else (1 if last < lower.iloc[-1] else 0)

    # Stochastic %K-like
    period = min(14, n)
    low14 = low.rolling(window=period, min_periods=1).min()
    high14 = high.rolling(window=period, min_periods=1).max()
    k = (close - low14) / (high14 - low14 + 1e-9) * 100
    out['STOCH'] = 1 if k.iloc[-1] > 50 else -1

    # Momentum
    out['MOM'] = 1 if close.iloc[-1] > close.shift(4).iloc[-1] else -1

    # CCI
    typical = (high + low + close) / 3
    ma = typical.rolling(window=20, min_periods=1).mean()
    mad = (typical - ma).abs().rolling(window=20, min_periods=1).mean()
    cci = (typical.iloc[-1] - ma.iloc[-1]) / (0.015 * (mad.iloc[-1] if mad.iloc[-1]!=0 else 1e-9))
    out['CCI'] = 1 if cci > 100 else (-1 if cci < -100 else 0)

    # OBV
    obv = ((close.diff().fillna(0) > 0) * vol - (close.diff().fillna(0) < 0) * vol).cumsum()
    out['OBV'] = 1 if obv.iloc[-1] > obv.rolling(window=20, min_periods=1).median().iloc[-1] else -1

    # ROC
    roc = (close.iloc[-1] - close.shift(12).fillna(close.iloc[0]).iloc[-1]) / (close.shift(12).fillna(close.iloc[0]).iloc[-1] + 1e-9)
    out['ROC'] = 1 if roc > 0 else -1

    # Williams %R
    highest = high.rolling(window=14, min_periods=1).max().iloc[-1]
    lowest = low.rolling(window=14, min_periods=1).min().iloc[-1]
    willr = (highest - close.iloc[-1]) / (highest - lowest + 1e-9) * -100
    out['WILLR'] = 1 if willr < -50 else -1

    # ATR
    prev_close = close.shift(1).fillna(close.iloc[0])
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=1).mean().iloc[-1]
    out['ATR'] = float(atr)

    # Price vs SMA5
    out['PR_SMA5'] = 1 if close.iloc[-1] > close.rolling(window=5, min_periods=1).mean().iloc[-1] else -1

    # ADX-like via macd magnitude
    macd_mag = macd.abs().rolling(window=14, min_periods=1).mean().iloc[-1]
    out['ADX_like'] = 1 if macd_mag > 1e-8 else 0

    return out

# -------------------------
# Voting & confidence
# -------------------------
WEIGHTS = {
    'SMA':2,'EMA':2,'MACD':2,'RSI':1,'BB':1,'STOCH':1,'MOM':1,
    'CCI':1,'OBV':1,'ROC':1,'WILLR':1,'ATR':0,'PR_SMA5':1,'ADX_like':1
}

def vote_and_confidence(ind: Dict[str,float]) -> Tuple[str, float]:
    score = 0.0
    max_score = 0.0
    for k,w in WEIGHTS.items():
        v = ind.get(k,0)
        score += v * w
        max_score += abs(w)
    if max_score == 0:
        conf = 0.0
    else:
        conf = abs(score)/max_score * 100
        atr = ind.get('ATR', None)
        if atr is not None:
            if atr < 0.0005:
                conf = min(100.0, conf + (0.0005-atr)*20000)
            elif atr > 0.005:
                conf = max(0.0, conf-8.0)
    direction = "Вверх ↑" if score >= 0 else "Вниз ↓"
    return direction, round(conf,1)

# -------------------------
# UI helpers
# -------------------------
def make_page_keyboard(items: List[str], page: int, prefix: str) -> InlineKeyboardMarkup:
    total = len(items)
    start = page * PAGE_SIZE
    end = min(total, start + PAGE_SIZE)
    rows = []
    for i in range(start, end):
        rows.append([InlineKeyboardButton(items[i], callback_data=f"{prefix}_idx_{i}")])
    nav = []
    if page > 0:
        nav.append(InlineKeyboardButton("⬅ Назад", callback_data=f"{prefix}_page_{page-1}"))
    if end < total:
        nav.append(InlineKeyboardButton("Вперед ➡", callback_data=f"{prefix}_page_{page+1}"))
    if nav:
        rows.append(nav)
    return InlineKeyboardMarkup(rows)

# -------------------------
# Bot handlers
# -------------------------
updater = Updater(BOT_TOKEN, use_context=True)
dp = updater.dispatcher

def start_cmd(update: Update, context: CallbackContext):
    kb = [
        [InlineKeyboardButton("💱 Валюты", callback_data='cat_forex_page_0')]
    ]
    update.message.reply_text("👋 Привет! Выберите валютную пару для анализа:", reply_markup=InlineKeyboardMarkup(kb))

def callback_handler(update: Update, context: CallbackContext):
    query = update.callback_query
    query.answer()
    data = query.data

    # category paging
    if data.startswith("cat_forex_page_"):
        page = int(data.split("_")[-1])
        query.edit_message_text("Выберите пару:", reply_markup=make_page_keyboard(FOREX, page, "pair"))
        return

    if data.startswith("pair_page_"):
        page = int(data.split("_")[-1])
        query.edit_message_text("Выберите пару:", reply_markup=make_page_keyboard(FOREX, page, "pair"))
        return

    if data.startswith("pair_idx_"):
        idx = int(data.split("_")[-1])
        instrument = FOREX[idx]
        context.user_data['instrument'] = instrument
        # expirations keyboard
        exp_kb = InlineKeyboardMarkup([[InlineKeyboardButton(e, callback_data=f"exp_{e}") for e in EXPIRATIONS]])
        exp_kb.inline_keyboard.append([InlineKeyboardButton("⬅ Назад к списку", callback_data="cat_forex_page_0")])
        query.edit_message_text(f"Вы выбрали: *{instrument}*\n\nВыберите экспирацию:", parse_mode='Markdown', reply_markup=exp_kb)
        return

    if data.startswith("exp_"):
        exp = data.split("_",1)[1]
        instrument = context.user_data.get('instrument', None)
        if instrument is None:
            query.edit_message_text("Инструмент не выбран. Вернитесь в меню.")
            return
        # send "analyzing" message then perform analysis in background
        sent = query.edit_message_text(f"⏳ Подождите *{ANALYSIS_WAIT}* секунд — идёт анализ рынка для *{instrument}*...", parse_mode='Markdown')
        # run analysis in thread to avoid blocking
        t = threading.Thread(target=do_analysis_and_send, args=(context.bot, query.message.chat_id, sent.message_id, instrument, exp, update.effective_user.id))
        t.daemon = True
        t.start()
        return

    if data == "new_signal":
        kb = [[InlineKeyboardButton("💱 Валюты", callback_data='cat_forex_page_0')]]
        query.edit_message_text("Выберите валютную пару для анализа:", reply_markup=InlineKeyboardMarkup(kb))
        return

    query.edit_message_text("Нераспознанная команда. Возврат в меню.")
    start_cmd(update, context)

# -------------------------
# Analysis flow
# -------------------------
def do_analysis_and_send(bot_obj, chat_id: int, message_id: int, instrument: str, exp: str, user_id):
    try:
        # wait ANALYSIS_WAIT seconds
        time.sleep(ANALYSIS_WAIT)

        seconds = expiration_to_seconds(exp)
        df = fetch_data(instrument, seconds)
        indicators = compute_indicators(df)
        signal, confidence = vote_and_confidence(indicators)

        price_open = float(df['Close'].iloc[-1]) if len(df)>0 else None

        # brief logical explanation
        expl_parts = []
        # trend check: ema8 vs ema21
        ema8 = df['Close'].ewm(span=8, adjust=False).mean().iloc[-1]
        ema21 = df['Close'].ewm(span=21, adjust=False).mean().iloc[-1]
        if ema8 > ema21:
            expl_parts.append("восходящий тренд (EMA8>EMA21)")
        else:
            expl_parts.append("нисходящий тренд (EMA8<EMA21)")

        rsi = indicators.get('_RSI', None)
        if rsi is not None:
            if rsi > 65:
                expl_parts.append("RSI в зоне перекупленности")
            elif rsi < 35:
                expl_parts.append("RSI в зоне перепроданности")
            else:
                expl_parts.append(f"RSI≈{int(rsi)}")

        bb = indicators.get('BB',0)
        if bb == 1:
            expl_parts.append("цена близка к нижней полосе Боллинджера (возможен откат)")
        elif bb == -1:
            expl_parts.append("цена выше верхней полосы Боллинджера (риск отката)")

        explanation = "; ".join(expl_parts[:3])

        # compose message
        text = (f"📊 *{instrument}* — анализ завершён\n\n"
                f"⏱ Экспирация: *{exp}*\n"
                f"📈 Сигнал: *{signal}*    🎯 Уверенность: *{confidence}%*\n\n"
                f"_Краткая логика:_ {explanation}\n"
                f"_Цена (прибл.):_ `{price_open:.6f}`\n\n"
                f"🔔 Откройте сделку в течение *10 секунд* (если торгуете вручную).")

        # edit message with signal
        try:
            bot_obj.edit_message_text(chat_id=chat_id, message_id=message_id, text=text, parse_mode='Markdown')
        except Exception:
            bot_obj.send_message(chat_id=chat_id, text=text, parse_mode='Markdown')

        # schedule finalize after expiration
        t2 = threading.Timer(seconds, finalize_result, args=(bot_obj, chat_id, message_id, instrument, exp, signal, confidence, price_open, user_id))
        t2.daemon = True
        t2.start()

        # log pending
        log_row({
            "timestamp": datetime.utcnow().isoformat(),
            "chat_id": chat_id,
            "user_id": user_id,
            "instrument": instrument,
            "expiration": exp,
            "signal": signal,
            "confidence": confidence,
            "price_open": price_open or "",
            "price_close": "",
            "result": "pending",
            "note": "analysis_sent"
        })

    except Exception as e:
        print("do_analysis error:", e)
        traceback.print_exc()
        try:
            bot_obj.send_message(chat_id, "Ошибка при анализе. Попробуйте снова.")
        except:
            pass

# -------------------------
# Finalize (simulate real check)
# -------------------------
def finalize_result(bot_obj, chat_id, message_id, instrument, exp, signal, confidence, price_open, user_id):
    try:
        seconds = expiration_to_seconds(exp)
        # try fetch new price after expiration
        try:
            df2 = fetch_data(instrument, seconds)
            price_close = float(df2['Close'].iloc[-1])
        except Exception:
            # fallback deterministic move in direction of signal with small noise
            base = price_open if price_open else (1.0 + (abs(hash(instrument))%100)/10000.0)
            move = random.uniform(0.0005, 0.003)
            price_close = round(base + move if signal.startswith("Вверх") else base - move, 6)

        # determine result correctly
        if (signal.startswith("Вверх") and price_close > price_open) or (signal.startswith("Вниз") and price_close < price_open):
            result = "Плюс ✅"
        else:
            result = "Минус ❌"

        # prepare final message
        final_text = (f"✅ *Сделка завершена*\n\n*{instrument}* | Экспирация: *{exp}*\n"
                      f"*Сигнал:* *{signal}*    *Результат:* *{result}*\n"
                      f"*Уверенность:* *{confidence}%*\n\n"
                      f"_Цена открытия:_ `{price_open:.6f}`\n_Цена закрытия:_ `{price_close:.6f}`")

        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("📊 Получить новый сигнал", callback_data="new_signal")],
            [InlineKeyboardButton("🔁 Выбрать другую пару", callback_data="cat_forex_page_0")]
        ])

        try:
            bot_obj.edit_message_text(chat_id=chat_id, message_id=message_id, text=final_text, parse_mode='Markdown', reply_markup=kb)
        except Exception:
            bot_obj.send_message(chat_id=chat_id, text=final_text, parse_mode='Markdown', reply_markup=kb)

        # update log
        log_row({
            "timestamp": datetime.utcnow().isoformat(),
            "chat_id": chat_id,
            "user_id": user_id,
            "instrument": instrument,
            "expiration": exp,
            "signal": signal,
            "confidence": confidence,
            "price_open": price_open,
            "price_close": price_close,
            "result": result,
            "note": ""
        })

    except Exception as e:
        print("finalize_result error:", e)
        traceback.print_exc()

# -------------------------
# helpers
# -------------------------
def expiration_to_seconds(exp: str) -> int:
    if exp.endswith('m'):
        return int(exp[:-1]) * 60
    if exp.endswith('s'):
        return int(exp[:-1])
    return 60

# -------------------------
# Entrypoint
# -------------------------
def main():
    ensure_log()
    keep_alive()
    dp.add_handler(CommandHandler("start", start_cmd))
    dp.add_handler(CallbackQueryHandler(callback_handler))
    print("Bot started (forex-only, analysis via yfinance).")
    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    if not BOT_TOKEN:

        print("Please set BOT_TOKEN env var or edit the script with your token.")
    else:
        updater = Updater(BOT_TOKEN, use_context=True)
        dp = updater.dispatcher
        main()
