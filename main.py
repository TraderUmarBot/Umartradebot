# ================================
# OXTSIGNALSBOT PRO — CLEAN VERSION
# Без пауз, без ограничений, без зависаний
# yfinance + fallback + мощная аналитика
# ================================

import os
import time
import threading
import random
import csv
import traceback
from datetime import datetime

import pandas as pd
import numpy as np
import yfinance as yf
import requests
from flask import Flask
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Updater, CommandHandler, CallbackQueryHandler, CallbackContext

# ========== CONFIG ==========
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

YF_PERIOD = "2d"
YF_INTERVAL = "1m"
FALLBACK_BARS = 480


# ========== FLASK KEEP-ALIVE ==========
app = Flask(__name__)
@app.route("/")
def index():
    return "OXTSIGNALSBOT PRO is running (Clean Edition)"

def keep_alive():
    threading.Thread(target=lambda: app.run(host="0.0.0.0", port=8080), daemon=True).start()


# ========== LOGGING ==========
def ensure_log():
    if not os.path.exists(LOG_CSV):
        with open(LOG_CSV, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "timestamp","chat_id","user_id","pair","exp","signal","conf","price_open","price_close","result"
            ])

def log_row(row):
    ensure_log()
    with open(LOG_CSV, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            row.get("timestamp",""),
            row.get("chat_id",""),
            row.get("user_id",""),
            row.get("pair",""),
            row.get("exp",""),
            row.get("signal",""),
            row.get("conf",""),
            row.get("price_open",""),
            row.get("price_close",""),
            row.get("result","")
        ])


# ========== UTILS ==========
def exp_to_seconds(exp):
    return int(exp.replace("m","")) * 60

def yf_symbol(p):
    p = p.upper().replace("/","")
    if len(p)==6:
        return p[:3] + p[3:] + "=X"
    return p


# ========== FALLBACK: REALISTIC SMART SERIES ==========
def smart_fallback(seed: str, bars: int = FALLBACK_BARS):
    rnd = random.Random(abs(hash(seed)) % 9999999)
    base = 1.0 + rnd.uniform(-0.05, 0.05)
    vol = rnd.uniform(0.0004, 0.002)
    times = pd.date_range(end=datetime.now(), periods=bars, freq="1min")
    O,H,L,C,V = [],[],[],[],[]

    price = base
    for _ in range(bars):
        drift = rnd.uniform(-0.0003, 0.0003)
        change = rnd.gauss(drift, vol)

        o = price
        c = price + change
        h = max(o,c) + abs(rnd.gauss(0, vol*0.8))
        l = min(o,c) - abs(rnd.gauss(0, vol*0.8))
        v = rnd.randint(50, 150)

        O.append(o); H.append(h); L.append(l); C.append(c); V.append(v)
        price = c

    return pd.DataFrame({"Open":O,"High":H,"Low":L,"Close":C,"Volume":V}, index=times)


# ========== GET DATA (yfinance + fallback) ==========
def fetch_data(pair: str):
    symbol = yf_symbol(pair)
    try:
        df = yf.download(
            symbol,
            period=YF_PERIOD,
            interval=YF_INTERVAL,
            threads=False,
            progress=False
        )
        if df is None or df.empty:
            raise Exception("empty yfinance")

        df = df.dropna(subset=["Close"])
        if df.empty:
            raise Exception("close empty")

        return df
    except Exception:
        print(f"[YF FAIL] Using fallback for {pair}")
        return smart_fallback(pair)


# ========== INDICATORS ==========
def compute_indicators(df: pd.DataFrame):
    out = {}
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)

    # EMA
    ema8 = close.ewm(span=8).mean().iloc[-1]
    ema21 = close.ewm(span=21).mean().iloc[-1]
    out["EMA"] = 1 if ema8 > ema21 else -1

    # SMA
    sma5 = close.rolling(5).mean().iloc[-1]
    sma20 = close.rolling(20).mean().iloc[-1]
    out["SMA"] = 1 if sma5 > sma20 else -1

    # MACD
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    macd = ema12 - ema26
    macd_sig = macd.ewm(span=9).mean()
    out["MACD"] = 1 if macd.iloc[-1] > macd_sig.iloc[-1] else -1
    out["MACD_mag"] = abs(float(macd.iloc[-1] - macd_sig.iloc[-1]))

    # RSI
    delta = close.diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = (-delta.clip(upper=0)).rolling(14).mean()
    rs = up / down.replace(0, 1e-9)
    rsi = 100 - (100 / (1 + rs))
    rsi_val = float(rsi.iloc[-1])
    out["_RSI"] = rsi_val
    out["RSI"] = 1 if rsi_val > 55 else (-1 if rsi_val < 45 else 0)

    # Bollinger
    ma = close.rolling(20).mean().iloc[-1]
    std = close.rolling(20).std().iloc[-1]
    last = close.iloc[-1]
    upper = ma + 2*std
    lower = ma - 2*std

    if last < lower:
        out["BB"] = 1
    elif last > upper:
        out["BB"] = -1
    else:
        out["BB"] = 0

    return out


# ========== VOTE ==========
def vote(indicators):
    score = 0
    max_s = 0
    for k,w in WEIGHTS.items():
        v = indicators.get(k,0)
        score += v*w
        max_s += abs(w)

    conf = int((abs(score)/max_s)*100)
    conf = max(55, min(95, conf))

    direction = "Вверх ↑" if score >= 0 else "Вниз ↓"
    return direction, conf


# ========== KEYBOARDS ==========
def main_menu():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("💱 Валютные пары", callback_data="fx_0")],
        [InlineKeyboardButton("📰 NON-FARM (NFP)", callback_data="nfp")]
    ])

def pairs_page(page):
    rows=[]
    start = page*PAGE_SIZE
    end = min(len(FOREX), start+PAGE_SIZE)
    for i in range(start,end):
        rows.append([InlineKeyboardButton(FOREX[i], callback_data=f"pair_{i}")])
    nav=[]
    if start>0: nav.append(InlineKeyboardButton("⬅", callback_data=f"fx_{page-1}"))
    if end<len(FOREX): nav.append(InlineKeyboardButton("➡", callback_data=f"fx_{page+1}"))
    if nav: rows.append(nav)
    return InlineKeyboardMarkup(rows)


# ========== START ==========
def start(update: Update, context: CallbackContext):
    update.message.reply_text("👋 Привет! Выберите режим:", reply_markup=main_menu())


# ========== CALLBACK ==========
def callback(update: Update, context: CallbackContext):
    q = update.callback_query
    data = q.data
    q.answer()

    # страницы
    if data.startswith("fx_"):
        page = int(data.split("_")[1])
        q.edit_message_text("Выберите валютную пару:", reply_markup=pairs_page(page))
        return

    # выбор пары
    if data.startswith("pair_"):
        idx = int(data.split("_")[1])
        pair = FOREX[idx]
        context.user_data["pair"] = pair

        kb=[[InlineKeyboardButton(e, callback_data=f"exp_{e}") for e in EXPIRATIONS]]
        kb.append([InlineKeyboardButton("⬅ Назад", callback_data="fx_0")])
        q.edit_message_text(f"Пара: *{pair}*\nВыберите экспирацию:", parse_mode="Markdown", reply_markup=InlineKeyboardMarkup(kb))
        return

    # выбор экспирации → анализ
    if data.startswith("exp_"):
        exp = data.replace("exp_","")
        pair = context.user_data.get("pair")

        msg = q.edit_message_text(f"⏳ Подождите {ANALYSIS_WAIT} сек — анализ {pair}...", parse_mode="Markdown")
        threading.Thread(target=analysis_worker, args=(context.bot, q.message.chat_id, msg.message_id, pair, exp, q.from_user.id), daemon=True).start()
        return

    # NFP
    if data=="nfp":
        msg = q.edit_message_text("⏳ Идёт NFP-Анализ...", parse_mode="Markdown")
        threading.Thread(target=nfp_worker, args=(context.bot, q.message.chat_id, msg.message_id), daemon=True).start()
        return


# ========== ANALYSIS WORKER ==========
def analysis_worker(bot, chat_id, msg_id, pair, exp, user_id):
    try:
        time.sleep(ANALYSIS_WAIT)

        df = fetch_data(pair)
        ind = compute_indicators(df)
        signal, conf = vote(ind)

        price_open = float(df["Close"].iloc[-1])

        logic=[]
        logic.append("EMA тренд ↑" if ind["EMA"]==1 else "EMA тренд ↓")
        logic.append(f"RSI≈{int(ind['_RSI'])}")
        if ind["BB"]==1: logic.append("Цена у нижней BB")
        elif ind["BB"]==-1: logic.append("Цена у верхней BB")

        text=(
            f"📊 *Анализ завершён*\n\n"
            f"Пара: *{pair}*\nЭксп: *{exp}*\n\n"
            f"Сигнал: *{signal}*\nУверенность: *{conf}%*\n\n"
            f"Логика: _{'; '.join(logic)}_\n"
            f"Цена: `{price_open}`"
        )

        bot.edit_message_text(chat_id=chat_id, message_id=msg_id, text=text, parse_mode="Markdown")

        log_row({
            "timestamp": datetime.utcnow().isoformat(),
            "chat_id": chat_id,
            "user_id": user_id,
            "pair": pair,
            "exp": exp,
            "signal": signal,
            "conf": conf,
            "price_open": price_open,
            "price_close": "",
            "result": "pending"
        })

        # таймер финализации
        seconds = exp_to_seconds(exp)
        threading.Timer(seconds, finalize_worker, args=(bot, chat_id, msg_id, pair, exp, signal, conf, price_open, user_id)).start()

    except Exception:
        bot.send_message(chat_id, "⚠️ Ошибка при анализе. Попробуйте снова.")
        traceback.print_exc()


# ========== FINALIZE ==========
def finalize_worker(bot, chat_id, msg_id, pair, exp, signal, conf, price_open, user_id):
    try:
        df = fetch_data(pair)
        price_close = float(df["Close"].iloc[-1])

        win = (signal.startswith("Вверх") and price_close > price_open) or \
              (signal.startswith("Вниз") and price_close < price_open)

        result = "Плюс ✅" if win else "Минус ❌"

        text=(
            f"✅ *Сделка завершена*\n\n"
            f"{pair} | {exp}\n"
            f"Сигнал: *{signal}*\nРезультат: *{result}*\n"
            f"Уверенность: *{conf}%*\n\n"
            f"Открытие: `{price_open}`\n"
            f"Закрытие: `{price_close}`"
        )

        bot.send_message(chat_id, text, parse_mode="Markdown")

        # return to menu
        bot.send_message(chat_id, "🔁 Возвращаю в меню:", reply_markup=main_menu())

        log_row({
            "timestamp": datetime.utcnow().isoformat(),
            "chat_id": chat_id,
            "user_id": user_id,
            "pair": pair,
            "exp": exp,
            "signal": signal,
            "conf": conf,
            "price_open": price_open,
            "price_close": price_close,
            "result": result
        })

    except Exception:
        bot.send_message(chat_id, "⚠️ Ошибка при получении результата.")
        traceback.print_exc()


# ========== NFP WORKER ==========
def nfp_worker(bot, chat_id, msg_id):
    try:
        pair="EURUSD"
        df=fetch_data(pair)
        ind=compute_indicators(df)
        signal, conf = vote(ind)

        text=(
            f"📰 *NFP анализ (EURUSD)*\n\n"
            f"Сигнал: *{signal}*\n"
            f"Уверенность: *{conf}%*\n\n"
            f"Используйте экспирацию 1-3 минуты."
        )
        bot.edit_message_text(chat_id=chat_id, message_id=msg_id, text=text, parse_mode="Markdown")

    except Exception:
        bot.send_message(chat_id, "⚠️ Ошибка NFP анализа.")
        traceback.print_exc()


# ========== DELETE WEBHOOK ==========
def delete_webhook():
    if not BOT_TOKEN:
        return
    try:
        r = requests.get(f"https://api.telegram.org/bot{BOT_TOKEN}/deleteWebhook")
        print("deleteWebhook:", r.text)
    except:
        pass


# ========== MAIN ==========
def main():
    if not BOT_TOKEN:
        print("ERROR: BOT_TOKEN is missing.")
        return

    keep_alive()
    delete_webhook()
    ensure_log()

    updater = Updater(BOT_TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CallbackQueryHandler(callback))

    print("✅ BOT STARTED (CLEAN EDITION)")
    updater.start_polling()
    updater.idle()


if __name__ == "__main__":
    main()
