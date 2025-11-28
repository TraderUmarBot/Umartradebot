# ============================
#     POCKET OPTION SIGNAL BOT
#       SINGLE FILE VERSION
# ============================

import logging
import time
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import numpy as np

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Updater, CommandHandler, CallbackQueryHandler, CallbackContext


# =====================================
#               НАСТРОЙКИ
# =====================================

TELEGRAM_TOKEN = "PUT_YOUR_TELEGRAM_BOT_TOKEN_HERE"   # <<< ВСТАВЬ СВОЙ ТОКЕН

PAIRS = [
    "EUR/USD","GBP/USD","USD/JPY","AUD/USD","USD/CAD","USD/CHF",
    "EUR/JPY","GBP/JPY","AUD/JPY","EUR/GBP","EUR/AUD","GBP/AUD",
    "CAD/JPY","CHF/JPY","EUR/CAD","GBP/CAD","AUD/CAD","AUD/CHF","CAD/CHF"
]

EXPIRATIONS = [1, 3, 5, 10]

NUM_CANDLES = 200
ANALYSIS_DELAY = 10


# =====================================
#             ЛОГИРОВАНИЕ
# =====================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

USER_STATE = {}


# =====================================
#        ПРЕОБРАЗОВАНИЕ В ТИКЕР
# =====================================

def pair_to_ticker(pair: str):
    base, quote = pair.split("/")
    return f"{base}{quote}=X"


# =====================================
#      ЗАГРУЗКА СВЕЧЕЙ С YFINANCE
# =====================================

def fetch_ohlc(pair: str, interval_minutes: int, num_candles: int):
    ticker = pair_to_ticker(pair)

    if interval_minutes == 1:
        interval = "1m"
    elif interval_minutes in [3, 5]:
        interval = "5m"
    else:
        interval = "15m"

    hours = int((num_candles * interval_minutes) / 60) + 1
    period = f"{hours}h"

    df = yf.Ticker(ticker).history(period=period, interval=interval)

    if df.empty:
        raise RuntimeError("Невозможно получить данные для анализа.")

    df = df.rename(columns={
        "Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"
    })[["open","high","low","close","volume"]]

    return df.tail(num_candles)


# =====================================
#         АНАЛИЗ с ИНДИКАТОРАМИ
# =====================================

def analyze_indicators(df: pd.DataFrame):
    votes = {}
    close = df["close"]

    # EMA
    ema8 = ta.ema(close, length=8)
    ema21 = ta.ema(close, length=21)
    votes["EMA"] = 1 if ema8.iloc[-1] > ema21.iloc[-1] else -1

    # RSI
    rsi = ta.rsi(close, length=14)
    if rsi.iloc[-1] > 60:
        votes["RSI"] = 1
    elif rsi.iloc[-1] < 40:
        votes["RSI"] = -1
    else:
        votes["RSI"] = 0

    # MACD
    macd = ta.macd(close)
    votes["MACD"] = 1 if macd["MACDh_12_26_9"].iloc[-1] > 0 else -1

    # BBANDS
    bb = ta.bbands(close)
    votes["BB"] = 1 if close.iloc[-1] > bb["BBM_20_2.0"].iloc[-1] else -1

    # STOCHASTIC
    stoch = ta.stoch(df["high"], df["low"], close)
    k = stoch["STOCHk_14_3_3"].iloc[-1]
    d = stoch["STOCHd_14_3_3"].iloc[-1]
    votes["STOCH"] = 1 if k > d else -1

    return votes


# =====================================
#        ПОСТРОЕНИЕ СИГНАЛА
# =====================================

def build_signal(votes):
    total = len(votes)
    bullish = list(votes.values()).count(1)
    bearish = list(votes.values()).count(-1)

    if bullish > bearish:
        direction = "Вверх"
        raw = bullish / total
    else:
        direction = "Вниз"
        raw = bearish / total

    confidence = int(70 + raw * 25)
    if confidence > 95: confidence = 95
    if confidence < 70: confidence = 70

    return direction, confidence


# =====================================
#           TELEGRAM HANDLERS
# =====================================

def start(update: Update, context: CallbackContext):
    keyboard = [[InlineKeyboardButton("Начать", callback_data="start")]]
    update.message.reply_text(
        "Привет трейдер! Нажми кнопку чтобы начать.",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )


def button(update: Update, context: CallbackContext):
    query = update.callback_query
    query.answer()
    chat = query.message.chat_id
    data = query.data

    # Выбор пары
    if data == "start":
        keyboard = []
        row = []
        for p in PAIRS:
            row.append(InlineKeyboardButton(p, callback_data=f"pair:{p}"))
            if len(row) == 3:
                keyboard.append(row)
                row = []
        if row:
            keyboard.append(row)

        query.message.reply_text(
            "Выбери валютную пару:",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
        return

    # Выбор экспирации
    if data.startswith("pair:"):
        pair = data.split(":")[1]
        USER_STATE[chat] = {"pair": pair}

        keyboard = [[InlineKeyboardButton(f"{e} мин", callback_data=f"exp:{e}") for e in EXPIRATIONS]]
        query.message.reply_text(
            f"Пара выбрана: {pair}\nТеперь выбери экспирацию:",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
        return

    # Анализ и вывод сигнала
    if data.startswith("exp:"):
        exp = int(data.split(":")[1])
        pair = USER_STATE[chat]["pair"]
        USER_STATE[chat]["exp"] = exp

        query.message.reply_text(f"Пара: {pair}\nЭкспирация: {exp} мин.\nПодождите, делаю анализ...")

        time.sleep(ANALYSIS_DELAY)

        try:
            df = fetch_ohlc(pair, exp, NUM_CANDLES)
            votes = analyze_indicators(df)
            direction, conf = build_signal(votes)

            msg = f"""
📊 *Сигнал готов!*

Пара: *{pair}*
Экспирация: *{exp} мин*

Направление: *{direction}*
Уверенность: *{conf}%*
"""
            query.message.reply_markdown(msg)

        except Exception as e:
            query.message.reply_text(f"Ошибка анализа: {e}")


# =====================================
#               MAIN
# =====================================

def main():
    updater = Updater(TELEGRAM_TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CallbackQueryHandler(button))

    updater.start_polling()
    updater.idle()


if __name__ == "__main__":
    main()
