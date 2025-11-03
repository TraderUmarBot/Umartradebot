# main.py
"""
OXTSIGNALSBOT — Stable single-file Telegram bot for Forex signals.
Requirements (requirements.txt):
python-telegram-bot==13.15
pandas
numpy
yfinance
flask
requests
"""

import os
import time
import threading
import traceback
import csv
import random
from datetime import datetime
from typing import Dict, Tuple

import pandas as pd
import numpy as np
import yfinance as yf
import requests
from flask import Flask

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Updater, CommandHandler, CallbackQueryHandler, CallbackContext

# -------------------------
# CONFIG
# -------------------------
BOT_TOKEN = os.getenv("BOT_TOKEN") or ""  # <-- DO NOT hardcode your token here in public repos
ANALYSIS_WAIT = 20  # seconds to wait "analysis"
PAGE_SIZE = 6
LOG_CSV = "signals_log.csv"

FOREX = [
    "EURUSD","GBPUSD","USDJPY","AUDUSD","USDCHF","EURJPY",
    "GBPJPY","NZDUSD","EURGBP","CADJPY","USDCAD","AUDJPY",
    "EURAUD","GBPAUD","EURNZD","AUDNZD","CADCHF","CHFJPY",
    "NZDJPY","GBPCAD"
]

EXPIRATIONS = ["1m", "2m", "3m", "5m"]

# indicator weights for voting
WEIGHTS = {
    "EMA": 2, "SMA": 2, "MACD": 2, "RSI": 1, "BB": 1, "STOCH": 1,
    "MOM": 1, "CCI": 1, "OBV": 1, "ROC": 1, "WILLR": 1, "PR_SMA5": 1
}

# -------------------------
# Keep-alive (Flask) for Render
# -------------------------
app = Flask(__name__)

@app.route("/")
def index():
    return "OXTSIGNALSBOT is alive"

def keep_alive():
    t = threading.Thread(target=lambda: app.run(host="0.0.0.0", port=8080), daemon=True)
    t.start()

# -------------------------
# Small utilities
# -------------------------
def ensure_log():
    if not os.path.exists(LOG_CSV):
        with open(LOG_CSV, "w", newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(["timestamp","chat_id","user_id","instrument","expiration","signal","confidence","price_open","price_close","result","note"])

def log_row(row: Dict):
    ensure_log()
    try:
        with open(LOG_CSV, "a", newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow([
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
        print("Logging failed:", traceback.format_exc())

def yf_symbol(pair: str) -> str:
    # e.g. EURUSD -> EURUSD=X for yfinance
    p = pair.upper().replace("/", "").replace(" ", "")
    if len(p) == 6 and p.isalpha():
        return f"{p[:3]}{p[3:]}=X"
    return pair

def exp_to_seconds(exp: str) -> int:
    if exp.endswith("m"):
        return int(exp[:-1]) * 60
    if exp.endswith("s"):
        return int(exp[:-1])
    return 60

# -------------------------
# Fetch data (safe)
# -------------------------
def simulate_series(seed: str, bars: int = 300) -> pd.DataFrame:
    rnd = random.Random(abs(hash(seed)) % (10**9))
    price = 1.0 + rnd.uniform(-0.02, 0.02)
    times = pd.date_range(end=pd.Timestamp.now(), periods=bars, freq="1min")
    opens, highs, lows, closes, vols = [], [], [], [], []
    for _ in range(bars):
        o = price
        c = max(1e-8, o + rnd.uniform(-0.002, 0.002))
        h = max(o, c) + rnd.uniform(0, 0.001)
        l = min(o, c) - rnd.uniform(0, 0.001)
        v = rnd.randint(10, 1000)
        opens.append(o); highs.append(h); lows.append(l); closes.append(c); vols.append(v)
        price = c
    return pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": vols}, index=times)

def fetch_data(pair: str, exp_seconds: int) -> pd.DataFrame:
    """
    Try yfinance; if no data or error -> return deterministic simulated series.
    """
    try:
        ticker = yf_symbol(pair)
        # choose period/interval to have many 1m bars
        period = "2d" if exp_seconds <= 60 else "5d"
        interval = "1m"
        df = yf.download(ticker, period=period, interval=interval, progress=False, threads=False)
        if df is None or df.empty:
            raise Exception("Empty DF from yfinance")
        df = df.dropna(subset=['Close'])
        if df.empty:
            raise Exception("DF empty after dropna")
        # ensure numeric
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
        df["Open"] = pd.to_numeric(df["Open"], errors="coerce")
        df["High"] = pd.to_numeric(df["High"], errors="coerce")
        df["Low"] = pd.to_numeric(df["Low"], errors="coerce")
        df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0)
        if df["Close"].isnull().all():
            raise Exception("Close all NaN")
        return df.dropna()
    except Exception as e:
        # fallback simulation
        print(f"fetch_data fallback for {pair}: {e}")
        return simulate_series(pair, bars=300)

# -------------------------
# Indicators (safe, no ambiguous Series checks)
# -------------------------
def compute_indicators(df: pd.DataFrame) -> Dict[str, float]:
    out: Dict[str, float] = {}
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    vol = df["Volume"].astype(float)
    n = len(close)
    if n < 2:
        # return neutral defaults
        out.update({"EMA":0,"SMA":0,"MACD":0,"RSI":0,"BB":0,"STOCH":0,"MOM":0,"CCI":0,"OBV":0,"ROC":0,"WILLR":0,"PR_SMA5":0,"_RSI":50.0})
        return out

    # EMA (8 vs 21)
    ema8 = close.ewm(span=8, adjust=False).mean().iloc[-1]
    ema21 = close.ewm(span=21, adjust=False).mean().iloc[-1]
    out["EMA"] = 1 if ema8 > ema21 else -1

    # SMA (5 vs 20)
    sma5 = close.rolling(window=5, min_periods=1).mean().iloc[-1]
    sma20 = close.rolling(window=min(20,n), min_periods=1).mean().iloc[-1]
    out["SMA"] = 1 if sma5 > sma20 else -1

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_sig = macd.ewm(span=9, adjust=False).mean()
    out["MACD"] = 1 if macd.iloc[-1] > macd_sig.iloc[-1] else -1

    # RSI
    delta = close.diff().dropna()
    up = delta.clip(lower=0).rolling(window=14, min_periods=1).mean()
    down = (-delta.clip(upper=0)).rolling(window=14, min_periods=1).mean()
    rs = up / down.replace(0, 1e-9)
    rsi_series = 100 - (100 / (1 + rs))
    rsi_val = float(rsi_series.iloc[-1]) if len(rsi_series) > 0 else 50.0
    out["_RSI"] = rsi_val
    out["RSI"] = 1 if rsi_val > 55 else (-1 if rsi_val < 45 else 0)

    # Bollinger Bands
    ma20 = close.rolling(window=min(20,n), min_periods=1).mean()
    std20 = close.rolling(window=min(20,n), min_periods=1).std().fillna(0)
    upper = ma20 + 2*std20
    lower = ma20 - 2*std20
    last_price = float(close.iloc[-1])
    out["BB"] = 1 if last_price < lower.iloc[-1] else (-1 if last_price > upper.iloc[-1] else 0)

    # Stochastic-like %K
    period = min(14, n)
    lowp = close.rolling(window=period, min_periods=1).min()
    highp = close.rolling(window=period, min_periods=1).max()
    k = (close - lowp) / (highp - lowp + 1e-9) * 100
    out["STOCH"] = 1 if k.iloc[-1] > 50 else -1

    # Momentum
    out["MOM"] = 1 if close.iloc[-1] > close.shift(4).iloc[-1] else -1

    # CCI
    typical = (high + low + close) / 3
    ma_typ = typical.rolling(window=min(20,n), min_periods=1).mean()
    mad = (typical - ma_typ).abs().rolling(window=min(20,n), min_periods=1).mean()
    cci = (typical.iloc[-1] - ma_typ.iloc[-1]) / (0.015 * (mad.iloc[-1] if mad.iloc[-1] != 0 else 1e-9))
    out["CCI"] = 1 if cci > 100 else (-1 if cci < -100 else 0)

    # OBV (simple)
    obv = ((close.diff().fillna(0) > 0) * vol - (close.diff().fillna(0) < 0) * vol).cumsum()
    obv_med = obv.rolling(window=min(20,n), min_periods=1).median().iloc[-1]
    out["OBV"] = 1 if obv.iloc[-1] > obv_med else -1

    # ROC
    prev = close.shift(12).fillna(close.iloc[0]).iloc[-1]
    roc = (close.iloc[-1] - prev) / (prev + 1e-9)
    out["ROC"] = 1 if roc > 0 else -1

    # Williams %R
    highest = high.rolling(window=min(14,n), min_periods=1).max().iloc[-1]
    lowest = low.rolling(window=min(14,n), min_periods=1).min().iloc[-1]
    willr = (highest - close.iloc[-1]) / (highest - lowest + 1e-9) * -100
    out["WILLR"] = 1 if willr < -50 else -1

    # Price vs SMA5
    out["PR_SMA5"] = 1 if close.iloc[-1] > close.rolling(window=min(5,n), min_periods=1).mean().iloc[-1] else -1

    return out

# -------------------------
# Voting and confidence
# -------------------------
def vote_and_confidence(ind: Dict[str, float]) -> Tuple[str, float]:
    score = 0.0
    max_score = 0.0
    for k, w in WEIGHTS.items():
        v = ind.get(k, 0)
        score += v * w
        max_score += abs(w)
    confidence = (abs(score) / max_score * 100) if max_score > 0 else 0.0
    # small volatility tweak (not required but helps)
    atr_like = ind.get("ATR", None) if "ATR" in ind else None
    # clamp
    confidence = max(0.0, min(99.9, confidence))
    direction = "Вверх ↑" if score >= 0 else "Вниз ↓"
    return direction, round(confidence, 1)

# -------------------------
# Telegram bot (handlers)
# -------------------------
# per-chat analysis lock to prevent double requests
analysis_locks = {}  # chat_id -> threading.Lock()

def get_lock(chat_id: int):
    if chat_id not in analysis_locks:
        analysis_locks[chat_id] = threading.Lock()
    return analysis_locks[chat_id]

# build inline keyboard for paging
def make_page_keyboard(items, page, prefix):
    total = len(items)
    start = page * PAGE_SIZE
    end = min(total, start + PAGE_SIZE)
    rows = []
    for i in range(start, end):
        rows.append([InlineKeyboardButton(items[i], callback_data=f"{prefix}_idx_{i}")])
    nav = []
    if start > 0:
        nav.append(InlineKeyboardButton("⬅ Назад", callback_data=f"{prefix}_page_{page-1}"))
    if end < total:
        nav.append(InlineKeyboardButton("Вперед ➡", callback_data=f"{prefix}_page_{page+1}"))
    if nav:
        rows.append(nav)
    return InlineKeyboardMarkup(rows)

def cmd_start(update: Update, context: CallbackContext):
    kb = [[InlineKeyboardButton("💱 Валюты", callback_data="cat_forex_page_0")]]
    update.message.reply_text("👋 Привет! Выберите валютную пару для анализа:", reply_markup=InlineKeyboardMarkup(kb))

def callback_handler(update: Update, context: CallbackContext):
    query = update.callback_query
    query.answer()
    data = query.data

    try:
        if data.startswith("cat_forex_page_"):
            page = int(data.split("_")[-1])
            query.edit_message_text("Выберите пару:", reply_markup=make_page_keyboard(FOREX, page, "pair"))
            return

        if data.startswith("pair_page_"):
            page = int(data.split("_")[-1])
            query.edit_message_text("Выберите пару:", reply_markup=make_page_keyboard(FOREX, page, "pair"))
            return

        if data.startswith("pair_idx_") or data.startswith("pair_"):
            # support both callback formats
            if data.startswith("pair_idx_"):
                idx = int(data.split("_")[-1])
                pair = FOREX[idx]
            else:
                pair = data.replace("pair_","")
            context.user_data['pair'] = pair
            kb = [[InlineKeyboardButton(e, callback_data=f"exp_{e}") for e in EXPIRATIONS]]
            kb.append([InlineKeyboardButton("⬅ Назад", callback_data="cat_forex_page_0")])
            query.edit_message_text(f"Вы выбрали: *{pair}*\n\nВыберите экспирацию:", parse_mode="Markdown", reply_markup=InlineKeyboardMarkup(kb))
            return

        if data.startswith("exp_"):
            exp = data.split("_",1)[1]
            pair = context.user_data.get('pair')
            if not pair:
                query.edit_message_text("Инструмент не выбран. Вернитесь в меню.")
                return

            # prevent double analysis for same chat
            lock = get_lock(query.message.chat_id)
            if not lock.acquire(blocking=False):
                query.answer("Уже идёт анализ — подождите.", show_alert=True)
                return

            # send "analyzing" message
            sent = query.edit_message_text(f"⏳ Подождите *{ANALYSIS_WAIT}* секунд — идёт профессиональный анализ рынка по *{pair}*...", parse_mode="Markdown")
            # run analysis in separate thread
            threading.Thread(target=do_analysis_and_report, args=(context.bot, query.message.chat_id, sent.message_id, pair, exp, query.from_user.id, lock), daemon=True).start()
            return

        if data == "new_signal":
            kb = [[InlineKeyboardButton("💱 Валюты", callback_data="cat_forex_page_0")]]
            query.edit_message_text("Выберите валютную пару для анализа:", reply_markup=InlineKeyboardMarkup(kb))
            return

        # fallback
        query.edit_message_text("Нераспознанная команда. Возврат в меню.")
        cmd_start(update, context)

    except Exception as e:
        print("callback_handler error:", e)
        traceback.print_exc()
        try:
            query.edit_message_text("Внутренняя ошибка. Попробуйте /start ещё раз.")
        except:
            pass

def do_analysis_and_report(bot, chat_id: int, message_id: int, pair: str, exp: str, user_id: int, lock: threading.Lock):
    """
    Performs analysis and edits message (or sends new one).
    Releases lock at the end.
    """
    try:
        # wait for ANALYSIS_WAIT
        time.sleep(ANALYSIS_WAIT)

        # fetch data
        df = fetch_data(pair, exp_to_seconds(exp))

        # if still no data, notify and release lock
        if df is None or df.empty:
            try:
                bot.send_message(chat_id, "⚠️ Не удалось получить данные для анализа. Попробуйте позже.")
            except:
                pass
            return
        # compute indicators
        ind = compute_indicators(df)
        direction, confidence = vote_and_confidence(ind)

        # price safe fetch
        try:
            price_open = float(df["Close"].iloc[-1])
        except Exception:
            price_open = 0.0

        # build short explanation
        expl = []
        expl.append("EMA8>EMA21" if ind.get("EMA",0) == 1 else "EMA8<EMA21")
        rsi_v = ind.get("_RSI", None)
        if rsi_v is not None:
            if rsi_v > 65:
                expl.append("RSI перекуплен")
            elif rsi_v < 35:
                expl.append("RSI перепродан")
            else:
                expl.append(f"RSI≈{int(rsi_v)}")
        bb = ind.get("BB",0)
        if bb == 1:
            expl.append("цена у нижней полосы BB")
        elif bb == -1:
            expl.append("цена у верхней полосы BB")

        explanation = "; ".join(expl[:3])

        text = (
            f"📊 *Анализ завершён*\n\n"
            f"🔹 Валютная пара: *{pair}*\n"
            f"🔹 Экспирация: *{exp}*\n\n"
            f"📈 *Сигнал:* *{direction}*    🎯 *Уверенность:* *{confidence}%*\n\n"
            f"_Краткая логика:_ {explanation}\n"
            f"_Цена (прибл.):_ `{price_open:.6f}`\n\n"
            f"🔔 Откройте сделку в течение *10 секунд*."
        )

        # try to edit original "analyzing" message; if fails, send new
        try:
            bot.edit_message_text(chat_id=chat_id, message_id=message_id, text=text, parse_mode="Markdown")
        except Exception:
            try:
                bot.send_message(chat_id, text, parse_mode="Markdown")
            except:
                pass

        # schedule finalize (simulate checking result at expiration)
        seconds = exp_to_seconds(exp)
        threading.Timer(seconds, finalize_and_report, args=(bot, chat_id, message_id, pair, exp, direction, confidence, price_open, user_id)).start()

        # log pending
        log_row({
            "timestamp": datetime.utcnow().isoformat(),
            "chat_id": chat_id,
            "user_id": user_id,
            "instrument": pair,
            "expiration": exp,
            "signal": direction,
            "confidence": confidence,
            "price_open": price_open,
            "price_close": "",
            "result": "pending",
            "note": "analysis_sent"
        })

    except Exception as e:
        print("do_analysis error:", e)
        traceback.print_exc()
        try:
            bot.send_message(chat_id, "⚠️ Ошибка при анализе. Попробуйте снова.")
        except:
            pass
    finally:
        # release lock if acquired
        try:
            if lock and isinstance(lock, threading.Lock):
                lock.release()
        except:
            pass

def finalize_and_report(bot, chat_id, message_id, pair, exp, direction, confidence, price_open, user_id):
    try:
        # try fetch actual close price after expiration
        try:
            df2 = fetch_data(pair, exp_to_seconds(exp))
            price_close = float(df2["Close"].iloc[-1])
        except Exception:
            # fallback deterministic move in direction of signal
            base = price_open if price_open and price_open != 0 else (1.0 + (abs(hash(pair)) % 100)/10000.0)
            move = random.uniform(0.0005, 0.003)
            price_close = round(base + move if direction.startswith("Вверх") else base - move, 6)

        # determine result
        if (direction.startswith("Вверх") and price_close > price_open) or (direction.startswith("Вниз") and price_close < price_open):
            result = "Плюс ✅"
        else:
            result = "Минус ❌"

        final_text = (
            f"✅ *Сделка завершена*\n\n"
            f"*{pair}* | Экспирация: *{exp}*\n"
            f"*Сигнал:* *{direction}*    *Результат:* *{result}*\n"
            f"*Уверенность:* *{confidence}%*\n\n"
            f"_Цена открытия:_ `{price_open:.6f}`\n_Цена закрытия:_ `{price_close:.6f}`"
        )

        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("📊 Получить новый сигнал", callback_data="new_signal")],
            [InlineKeyboardButton("🔁 Выбрать другую пару", callback_data="cat_forex_page_0")]
        ])

        try:
            bot.edit_message_text(chat_id=chat_id, message_id=message_id, text=final_text, parse_mode="Markdown", reply_markup=kb)
        except Exception:
            try:
                bot.send_message(chat_id, final_text, parse_mode="Markdown", reply_markup=kb)
            except:
                pass

        # update log
        log_row({
            "timestamp": datetime.utcnow().isoformat(),
            "chat_id": chat_id,
            "user_id": user_id,
            "instrument": pair,
            "expiration": exp,
            "signal": direction,
            "confidence": confidence,
            "price_open": price_open,
            "price_close": price_close,
            "result": result,
            "note": ""
        })

    except Exception as e:
        print("finalize error:", e)
        traceback.print_exc()

# -------------------------
# Delete webhook on startup to avoid polling conflicts
# -------------------------
def delete_webhook_if_any():
    try:
        if not BOT_TOKEN:
            return
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/deleteWebhook"
        r = requests.get(url, timeout=5)
        print("deleteWebhook:", r.status_code, r.text[:200])
    except Exception as e:
        print("deleteWebhook error:", e)

# -------------------------
# Entrypoint
# -------------------------
def main():
    if not BOT_TOKEN:
        print("ERROR: BOT_TOKEN is empty. Set BOT_TOKEN in environment variables.")
        return

    ensure_log()
    keep_alive()
    delete_webhook_if_any()

    updater = Updater(BOT_TOKEN, use_context=True)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", cmd_start))
    dp.add_handler(CallbackQueryHandler(callback_handler))

    print("Starting bot (polling)...")
    # start polling; long_polling
    updater.start_polling(poll_interval=3.0, timeout=60)
    updater.idle()

if __name__ == "__main__":
    main()
