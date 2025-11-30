# ============================
# POCKET OPTION SIGNAL BOT (Render 24/7 - POLLING)
# ============================

import os
import sys
import types
import logging
import pandas as pd
import pandas_ta as ta
import yfinance as yf
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Updater,
    CommandHandler,
    CallbackQueryHandler,
    CallbackContext
)

# ============================
# Патч для Python 3.13 (но мы используем 3.10 — просто запас)
# ============================
if sys.version_info >= (3, 13):
    sys.modules['imghdr'] = types.ModuleType('imghdr')
    sys.modules['imghdr'].what = lambda *args, **kwargs: None

# ============================
# Настройки
# ============================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

PAIRS = [
    "EUR/USD","GBP/USD","USD/JPY","AUD/USD","USD/CAD","USD/CHF",
    "EUR/JPY","GBP/JPY","AUD/JPY","EUR/GBP","EUR/AUD","GBP/AUD",
    "CAD/JPY","CHF/JPY","EUR/CAD","GBP/CAD","AUD/CAD","AUD/CHF","CAD/CHF"
]

EXPIRATIONS = [1, 3, 5, 10]
NUM_CANDLES = 200
ANALYSIS_DELAY = 1

USER_STATE = {}
TRADE_HISTORY = {}

# ============================
# Логирование
# ============================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================
# Функции анализа
# ============================
def pair_to_ticker(pair):
    base, quote = pair.split("/")
    return f"{base}{quote}=X"

def fetch_ohlc(pair, interval_minutes, num_candles):
    ticker = pair_to_ticker(pair)
    interval = "1m" if interval_minutes == 1 else ("5m" if interval_minutes in [3,5] else "15m")
    hours = int((num_candles * interval_minutes) / 60) + 1
    period = f"{hours}h"
    df = yf.Ticker(ticker).history(period=period, interval=interval)
    if df.empty:
        raise Exception("Не удалось загрузить данные для анализа.")
    df = df.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"})
    return df[["open","high","low","close","volume"]].tail(num_candles)

def analyze_indicators(df):
    votes = {}
    close = df["close"]

    ema8 = ta.ema(close, length=8)
    ema21 = ta.ema(close, length=21)
    votes["EMA"] = 1 if ema8.iloc[-1] > ema21.iloc[-1] else -1

    rsi = ta.rsi(close, length=14)
    if rsi.iloc[-1] > 60: votes["RSI"] = 1
    elif rsi.iloc[-1] < 40: votes["RSI"] = -1
    else: votes["RSI"] = 0

    macd = ta.macd(close)
    votes["MACD"] = 1 if macd["MACDh_12_26_9"].iloc[-1] > 0 else -1

    bb = ta.bbands(close)
    votes["BB"] = 1 if close.iloc[-1] > bb["BBM_20_2.0"].iloc[-1] else -1

    stoch = ta.stoch(df["high"], df["low"], close)
    votes["STOCH"] = 1 if stoch.iloc[-1]["STOCHk_14_3_3"] > stoch.iloc[-1]["STOCHd_14_3_3"] else -1

    return votes

def build_signal(votes):
    total = len(votes)
    bullish = list(votes.values()).count(1)
    bearish = list(votes.values()).count(-1)
    direction = "📈 Вверх" if bullish > bearish else "📉 Вниз"
    raw = bullish/total if bullish > bearish else bearish/total
    confidence = min(max(int(70 + raw*25), 70), 95)
    return direction, confidence

# ============================
# Главное меню
# ============================
def main_menu(update, context):
    keyboard = [
        [InlineKeyboardButton("📊 Выбор валюты", callback_data="menu_currency")],
        [InlineKeyboardButton("📂 История сделок", callback_data="menu_history")]
    ]
    update.message.reply_text("🚀 Главное меню:", reply_markup=InlineKeyboardMarkup(keyboard))

# ============================
# Обработка кнопок
# ============================
def button(update, context):
    q = update.callback_query
    q.answer()
    chat = q.message.chat_id
    data = q.data

    if data == "menu_currency":
        keyboard = []
        row = []
        for p in PAIRS:
            row.append(InlineKeyboardButton(p, callback_data=f"pair:{p}"))
            if len(row) == 3:
                keyboard.append(row)
                row = []
        if row: keyboard.append(row)
        q.message.reply_text("Выберите валютную пару:", reply_markup=InlineKeyboardMarkup(keyboard))
        return

    if data == "menu_history":
        history = TRADE_HISTORY.get(chat, [])
        if not history:
            q.message.reply_text("📂 История пуста.")
            return
        msg = "📂 История:\n\n"
        for t in history[-10:]:
            msg += f"{t['pair']} | {t['exp']} мин | {t['direction']} | {t['confidence']}% | Результат: {t['result']}\n"
        q.message.reply_text(msg)
        return

    if data.startswith("pair:"):
        pair = data.split(":")[1]
        USER_STATE[chat] = {"pair": pair}
        keyboard = [[InlineKeyboardButton(f"{e} мин", callback_data=f"exp:{e}") for e in EXPIRATIONS]]
        q.message.reply_text(f"Пара выбрана: *{pair}*\nВыберите экспирацию:", parse_mode="Markdown", reply_markup=InlineKeyboardMarkup(keyboard))
        return

    if data.startswith("exp:"):
        exp = int(data.split(":")[1])
        pair = USER_STATE[chat]["pair"]
        USER_STATE[chat]["exp"] = exp
        q.message.reply_text("⏳ Анализирую сигнал…")
        context.job_queue.run_once(run_signal, 1, context={"chat": chat, "pair": pair, "exp": exp})
        return

    if data.startswith("result:"):
        result = data.split(":")[1]
        trade = USER_STATE[chat].get("last")
        if trade:
            TRADE_HISTORY.setdefault(chat, []).append({**trade, "result": result})
            q.message.reply_text("✅ Результат сохранён!")
        USER_STATE[chat] = {}
        main_menu(update, context)

# ============================
# Анализ
# ============================
def run_signal(context):
    data = context.job.context
    chat = data["chat"]
    pair = data["pair"]
    exp = data["exp"]

    try:
        df = fetch_ohlc(pair, exp, NUM_CANDLES)
        votes = analyze_indicators(df)
        direction, conf = build_signal(votes)

        USER_STATE[chat]["last"] = {"pair": pair, "exp": exp, "direction": direction, "confidence": conf}

        msg = f"""
📊 *Сигнал готов!*

💹 Пара: *{pair}*
⏱ Экспирация: *{exp} мин*

➡ Направление: *{direction}*
🔥 Уверенность: *{conf}%*
"""
        context.bot.send_message(chat, msg, parse_mode="Markdown")

        keyboard = [
            [InlineKeyboardButton("👍 Профит", callback_data="result:+"),
             InlineKeyboardButton("👎 Лосс", callback_data="result:-")]
        ]
        context.bot.send_message(chat, "Отметьте результат:", reply_markup=InlineKeyboardMarkup(keyboard))

    except Exception as e:
        context.bot.send_message(chat, f"⚠ Ошибка анализа: {e}")

# ============================
# MAIN
# ============================
def main():
    if not TELEGRAM_TOKEN:
        print("TELEGRAM_TOKEN не установлен!")
        return

    updater = Updater(TELEGRAM_TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", main_menu))
    dp.add_handler(CallbackQueryHandler(button))

    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()
