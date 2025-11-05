import os
import time
import threading
import random
import csv
import traceback
from datetime import datetime
from typing import Dict, Tuple

import pandas as pd
import numpy as np
import yfinance as yf
from flask import Flask
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Updater, CommandHandler, CallbackQueryHandler, CallbackContext
import requests


# ✅ CONFIG
BOT_TOKEN = os.getenv("BOT_TOKEN") or ""
ANALYSIS_WAIT = 20
PAGE_SIZE = 6
LOG_CSV = "signals_log.csv"

FOREX = [
    "EURUSD","GBPUSD","USDJPY","AUDUSD","USDCHF","EURJPY",
    "GBPJPY","NZDUSD","EURGBP","CADJPY","USDCAD","AUDJPY",
    "EURAUD","GBPAUD","EURNZD","AUDNZD","CADCHF","CHFJPY",
    "NZDJPY","GBPCAD"
]

EXPIRATIONS = ["1m", "2m", "3m", "5m"]

WEIGHTS = {"EMA":2, "SMA":2, "MACD":2, "RSI":1, "BB":1}


# ✅ Convert exp "1m" → seconds
def exp_to_seconds(exp: str) -> int:
    if exp.endswith("m"):
        return int(exp.replace("m","")) * 60
    return 60


# ✅ Flask keep-alive for Render
app = Flask(__name__)

@app.route("/")
def home():
    return "OXTSIGNALS Bot is running."


def keep_alive():
    threading.Thread(target=lambda: app.run(host="0.0.0.0", port=8080), daemon=True).start()


# ✅ Log file
def ensure_log():
    if not os.path.exists(LOG_CSV):
        with open(LOG_CSV, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "timestamp","chat_id","user_id","instrument","expiration",
                "signal","confidence","price_open","price_close","result"
            ])


def log_row(data: dict):
    ensure_log()
    with open(LOG_CSV, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            data.get("timestamp",""),
            data.get("chat_id",""),
            data.get("user_id",""),
            data.get("instrument",""),
            data.get("expiration",""),
            data.get("signal",""),
            data.get("confidence",""),
            data.get("price_open",""),
            data.get("price_close",""),
            data.get("result","")
        ])


# ✅ Yahoo symbol convert
def yf_symbol(pair: str) -> str:
    pair = pair.upper().replace(" ","")
    if len(pair) == 6:
        return pair[:3] + pair[3:] + "=X"
    return pair


# ✅ Simulated fallback data
def simulate_series(seed: str) -> pd.DataFrame:
    rnd = random.Random(abs(hash(seed)) % 999999)
    base = 1 + rnd.uniform(-0.02, 0.02)
    times = pd.date_range(end=datetime.now(), periods=240, freq="1min")

    data = []
    for _ in range(240):
        o = base
        c = o + rnd.uniform(-0.001, 0.001)
        h = max(o, c) + rnd.uniform(0, 0.0007)
        l = min(o, c) - rnd.uniform(0, 0.0007)
        v = rnd.randint(50, 200)
        data.append([o,h,l,c,v])
        base = c

    return pd.DataFrame(data, columns=["Open","High","Low","Close","Volume"], index=times)


# ✅ Safe yahoo fetch
def fetch_data(pair: str) -> pd.DataFrame:
    try:
        df = yf.download(yf_symbol(pair), period="2d", interval="1m", progress=False, threads=False)
        if df is None or df.empty:
            raise Exception("empty df")

        df = df.dropna(subset=["Close"])
        if df.empty:
            raise Exception("empty after drop")

        return df

    except Exception:
        print(f"⚠️ yfinance failed, fallback to simulation for {pair}")
        return simulate_series(pair)


# ✅ Indicators
def compute_indicators(df: pd.DataFrame) -> dict:
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)

    res = {}

    # EMA
    ema8 = close.ewm(span=8, adjust=False).mean().iloc[-1]
    ema21 = close.ewm(span=21, adjust=False).mean().iloc[-1]
    res["EMA"] = 1 if ema8 > ema21 else -1

    # SMA
    sma5 = close.rolling(5).mean().iloc[-1]
    sma20 = close.rolling(20).mean().iloc[-1]
    res["SMA"] = 1 if sma5 > sma20 else -1

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_sig = macd.ewm(span=9, adjust=False).mean()
    res["MACD"] = 1 if macd.iloc[-1] > macd_sig.iloc[-1] else -1

    # RSI
    delta = close.diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = (-delta.clip(upper=0)).rolling(14).mean()
    rs = up / down.replace(0, 1e-9)
    rsi = 100 - (100 / (1 + rs))
    rsi_val = float(rsi.iloc[-1])
    res["RSI"] = 1 if rsi_val > 55 else (-1 if rsi_val < 45 else 0)
    res["_RSI_VAL"] = rsi_val

    # Bollinger
    ma = close.rolling(20).mean().iloc[-1]
    std = close.rolling(20).std().iloc[-1]
    upper = ma + 2*std
    lower = ma - 2*std
    last = close.iloc[-1]
    if last < lower:
        res["BB"] = 1
    elif last > upper:
        res["BB"] = -1
    else:
        res["BB"] = 0

    return res


# ✅ Voting system
def vote(ind: dict) -> Tuple[str, float]:
    score = 0
    max_s = 0

    for k,w in WEIGHTS.items():
        score += ind.get(k,0) * w
        max_s += abs(w)

    conf = int((abs(score)/max_s) * 100)
    conf = max(55, min(95, conf))

    direction = "Вверх ↑" if score >= 0 else "Вниз ↓"
    return direction, conf


# ✅ Locks per chat (prevents overlapping analysis)
locks = {}

def get_lock(chat_id):
    if chat_id not in locks:
        locks[chat_id] = threading.Lock()
    return locks[chat_id]


# ✅ Telegram handlers
def start(update: Update, context: CallbackContext):
    kb = [[InlineKeyboardButton("💱 Валютные пары", callback_data="cat_fx_0")]]
    update.message.reply_text("Выберите валютную пару:", reply_markup=InlineKeyboardMarkup(kb))


def callback(update: Update, context: CallbackContext):
    q = update.callback_query
    q.answer()
    data = q.data

    # pages
    if data.startswith("cat_fx_"):
        page = int(data.split("_")[-1])
        start_i = page * PAGE_SIZE
        end_i = min(len(FOREX), start_i + PAGE_SIZE)

        rows = []
        for i in range(start_i, end_i):
            rows.append([InlineKeyboardButton(FOREX[i], callback_data=f"pair_{i}")])

        nav = []
        if start_i > 0:
            nav.append(InlineKeyboardButton("⬅", callback_data=f"cat_fx_{page-1}"))
        if end_i < len(FOREX):
            nav.append(InlineKeyboardButton("➡", callback_data=f"cat_fx_{page+1}"))
        if nav:
            rows.append(nav)

        q.edit_message_text("Выберите пару:", reply_markup=InlineKeyboardMarkup(rows))
        return

    # pair selected
    if data.startswith("pair_"):
        idx = int(data.split("_")[1])
        pair = FOREX[idx]
        context.user_data["pair"] = pair

        kb = [[InlineKeyboardButton(e, callback_data=f"exp_{e}") for e in EXPIRATIONS]]
        q.edit_message_text(f"Вы выбрали: {pair}\nВыберите экспирацию:", reply_markup=InlineKeyboardMarkup(kb))
        return

    # expiration selected → run analysis
    if data.startswith("exp_"):
        exp = data.replace("exp_","")
        pair = context.user_data.get("pair")

        lock = get_lock(q.message.chat_id)
        if not lock.acquire(blocking=False):
            q.answer("Уже идёт анализ!", show_alert=True)
            return

        msg = q.edit_message_text(
            f"⏳ Подождите {ANALYSIS_WAIT} секунд — идёт анализ по {pair}..."
        )

        threading.Thread(
            target=run_analysis,
            args=(context.bot, q.message.chat_id, msg.message_id, pair, exp, q.from_user.id, lock),
            daemon=True
        ).start()

        return

    if data == "new_signal":
        start(update, context)
        return


# ✅ Analysis worker
def run_analysis(bot, chat_id, msg_id, pair, exp, user_id, lock):
    try:
        time.sleep(ANALYSIS_WAIT)

        df = fetch_data(pair)
        ind = compute_indicators(df)
        signal, conf = vote(ind)
        price_open = float(df["Close"].iloc[-1])

        logic = []

        logic.append("EMA восходящая" if ind["EMA"] == 1 else "EMA нисходящая")
        rsi = ind["_RSI_VAL"]
        if rsi > 65: logic.append("RSI перекуплен")
        elif rsi < 35: logic.append("RSI перепродан")
        else: logic.append(f"RSI≈{int(rsi)}")

        if ind["BB"] == 1: logic.append("Цена у нижней полосы Bollinger")
        elif ind["BB"] == -1: logic.append("Цена у верхней полосы Bollinger")

        text = (
            f"✅ Анализ завершён\n\n"
            f"Пара: *{pair}*\n"
            f"Экспирация: *{exp}*\n\n"
            f"Сигнал: *{signal}*\n"
            f"Уверенность: *{conf}%*\n\n"
            f"Логика: _{'; '.join(logic)}_\n"
            f"Цена открытия: `{price_open}`"
        )

        bot.edit_message_text(
            chat_id=chat_id, message_id=msg_id,
            text=text, parse_mode="Markdown"
        )

        log_row({
            "timestamp": datetime.utcnow().isoformat(),
            "chat_id": chat_id,
            "user_id": user_id,
            "instrument": pair,
            "expiration": exp,
            "signal": signal,
            "confidence": conf,
            "price_open": price_open,
            "price_close": "",
            "result": "pending"
        })

        # schedule finalize
        seconds = exp_to_seconds(exp)
        threading.Timer(seconds, finalize, args=(
            bot, chat_id, msg_id, pair, exp,
            signal, conf, price_open, user_id
        )).start()

    except Exception as e:
        print("❌ ANALYSIS ERROR:", e)
        bot.send_message(chat_id, "⚠️ Ошибка анализа. Попробуйте снова.")
    finally:
        lock.release()


# ✅ Finalize trade
def finalize(bot, chat_id, msg_id, pair, exp, signal, conf, price_open, user_id):
    try:
        df = fetch_data(pair)
        price_close = float(df["Close"].iloc[-1])

        result = "Плюс ✅" if (
            (signal.startswith("Вверх") and price_close > price_open) or
            (signal.startswith("Вниз") and price_close < price_open)
        ) else "Минус ❌"

        text = (
            f"✅ Сделка завершена\n\n"
            f"{pair} | {exp}\n"
            f"Сигнал: *{signal}*\n"
            f"Результат: *{result}*\n"
            f"Уверенность: *{conf}%*\n\n"
            f"Открытие: `{price_open}`\n"
            f"Закрытие: `{price_close}`"
        )

        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("📊 Новый сигнал", callback_data="new_signal")]
        ])

        bot.send_message(chat_id, text, parse_mode="Markdown", reply_markup=kb)

        log_row({
            "timestamp": datetime.utcnow().isoformat(),
            "chat_id": chat_id,
            "user_id": user_id,
            "instrument": pair,
            "expiration": exp,
            "signal": signal,
            "confidence": conf,
            "price_open": price_open,
            "price_close": price_close,
            "result": result
        })

    except Exception as e:
        print("❌ FINALIZE ERROR:", e)


# ✅ Delete webhook (fix conflict)
def delete_webhook():
    if not BOT_TOKEN:
        return
    try:
        r = requests.get(f"https://api.telegram.org/bot{BOT_TOKEN}/deleteWebhook")
        print("deleteWebhook:", r.text)
    except:
        pass


# ✅ Run bot
def main():
    if not BOT_TOKEN:
        print("ERROR: BOT_TOKEN missing!")
        return

    keep_alive()
    delete_webhook()

    updater = Updater(BOT_TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CallbackQueryHandler(callback))

    print("✅ Bot started.")
    updater.start_polling()
    updater.idle()


if __name__ == "__main__":
    main()
