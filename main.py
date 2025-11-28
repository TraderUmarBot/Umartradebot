# ============================
# POCKET OPTION SIGNAL BOT - FULL 24/7
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
# Патч для Python 3.13 (имитация imghdr)
# ============================
if sys.version_info >= (3, 13):
    sys.modules['imghdr'] = types.ModuleType('imghdr')
    def what(filename, h=None):
        return None
    sys.modules['imghdr'].what = what

# =====================================
# Настройки
# =====================================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")  # Устанавливаем через переменные окружения

PAIRS = [
    "EUR/USD","GBP/USD","USD/JPY","AUD/USD","USD/CAD","USD/CHF",
    "EUR/JPY","GBP/JPY","AUD/JPY","EUR/GBP","EUR/AUD","GBP/AUD",
    "CAD/JPY","CHF/JPY","EUR/CAD","GBP/CAD","AUD/CAD","AUD/CHF","CAD/CHF"
]

EXPIRATIONS = [1, 3, 5, 10]
NUM_CANDLES = 200
ANALYSIS_DELAY = 1  # задержка перед анализом в секундах для JobQueue

# Состояние пользователей
USER_STATE = {}
TRADE_HISTORY = {}  # структура: {chat_id: [{"pair":..,"exp":..,"direction":..,"confidence":..,"result":..}, ...]}

# =====================================
# Логирование
# =====================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="bot.log",
    filemode="a"
)
logger = logging.getLogger(__name__)

# =====================================
# Вспомогательные функции
# =====================================
def pair_to_ticker(pair: str):
    base, quote = pair.split("/")
    return f"{base}{quote}=X"

def fetch_ohlc(pair: str, interval_minutes: int, num_candles: int):
    ticker = pair_to_ticker(pair)
    interval = "1m" if interval_minutes==1 else ("5m" if interval_minutes in [3,5] else "15m")
    hours = int((num_candles * interval_minutes)/60)+1
    period = f"{hours}h"

    df = yf.Ticker(ticker).history(period=period, interval=interval)
    if df.empty:
        raise RuntimeError("Невозможно получить данные для анализа.")

    df = df.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"})
    return df[["open","high","low","close","volume"]].tail(num_candles)

def analyze_indicators(df: pd.DataFrame):
    votes = {}
    close = df["close"]

    # EMA
    ema8 = ta.ema(close, length=8)
    ema21 = ta.ema(close, length=21)
    votes["EMA"] = 1 if ema8.iloc[-1] > ema21.iloc[-1] else -1

    # RSI
    rsi = ta.rsi(close, length=14)
    if rsi.iloc[-1] > 60: votes["RSI"] = 1
    elif rsi.iloc[-1] < 40: votes["RSI"] = -1
    else: votes["RSI"] = 0

    # MACD
    macd = ta.macd(close)
    votes["MACD"] = 1 if macd["MACDh_12_26_9"].iloc[-1] > 0 else -1

    # Bollinger Bands
    bb = ta.bbands(close)
    votes["BB"] = 1 if close.iloc[-1] > bb["BBM_20_2.0"].iloc[-1] else -1

    # Stochastic
    stoch = ta.stoch(df["high"], df["low"], close)
    k = stoch["STOCHk_14_3_3"].iloc[-1]
    d = stoch["STOCHd_14_3_3"].iloc[-1]
    votes["STOCH"] = 1 if k > d else -1

    return votes

def build_signal(votes):
    total = len(votes)
    bullish = list(votes.values()).count(1)
    bearish = list(votes.values()).count(-1)

    direction = "📈 Вверх" if bullish > bearish else "📉 Вниз"
    raw = bullish/total if bullish > bearish else bearish/total
    confidence = min(max(int(70 + raw*25),70),95)
    return direction, confidence

# =====================================
# Главное меню
# =====================================
def main_menu(update: Update, context: CallbackContext):
    keyboard = [
        [InlineKeyboardButton("📊 Выбор валюты", callback_data="menu_currency")],
        [InlineKeyboardButton("📂 История сделок", callback_data="menu_history")]
    ]
    update.message.reply_text(
        "🚀 Главное меню:",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

# =====================================
# Обработка кнопок
# =====================================
def button(update: Update, context: CallbackContext):
    query = update.callback_query
    query.answer()
    chat = query.message.chat_id
    data = query.data

    # Главное меню
    if data == "menu_currency":
        keyboard = []
        row = []
        for p in PAIRS:
            row.append(InlineKeyboardButton(p, callback_data=f"pair:{p}"))
            if len(row) == 3:
                keyboard.append(row)
                row = []
        if row: keyboard.append(row)
        query.message.reply_text("Выберите валютную пару:", reply_markup=InlineKeyboardMarkup(keyboard))
        return

    if data == "menu_history":
        history = TRADE_HISTORY.get(chat, [])
        if not history:
            query.message.reply_text("📂 История пуста.")
        else:
            msg = "📂 История сделок:\n\n"
            for t in history[-10:]:
                msg += f"💹 {t['pair']} | ⏱ {t['exp']} мин | {t['direction']} | 🔥 {t['confidence']}% | Результат: {t['result']}\n"
            query.message.reply_text(msg)
        return

    # Выбор пары
    if data.startswith("pair:"):
        pair = data.split(":")[1]
        USER_STATE[chat] = {"pair": pair}
        keyboard = [[InlineKeyboardButton(f"{e} мин", callback_data=f"exp:{e}") for e in EXPIRATIONS]]
        query.message.reply_text(f"✅ Пара выбрана: *{pair}*\nВыберите экспирацию:", reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="Markdown")
        return

    # Выбор экспирации
    if data.startswith("exp:"):
        exp = int(data.split(":")[1])
        pair = USER_STATE[chat]["pair"]
        USER_STATE[chat]["exp"] = exp
        query.message.reply_text(f"⏳ Пара: *{pair}*\n⏱ Экспирация: *{exp} мин*\nАнализируем сигнал...", parse_mode="Markdown")
        context.job_queue.run_once(run_analysis, ANALYSIS_DELAY, context={"chat_id": chat, "pair": pair, "exp": exp})
        return

    # Отметка результата 👍/👎
    if data.startswith("result:"):
        result = data.split(":")[1]
        last_trade = USER_STATE[chat].get("last_signal")
        if last_trade:
            TRADE_HISTORY.setdefault(chat, []).append({**last_trade, "result": result})
            query.message.reply_text(f"✅ Результат {result} сохранён.")
        USER_STATE[chat] = {}  # сброс состояния
        main_menu(update, context)
        return

# =====================================
# Асинхронный анализ сигнала
# =====================================
def run_analysis(context: CallbackContext):
    job = context.job
    chat_id = job.context["chat_id"]
    pair = job.context["pair"]
    exp = job.context["exp"]

    try:
        df = fetch_ohlc(pair, exp, NUM_CANDLES)
        votes = analyze_indicators(df)
        direction, confidence = build_signal(votes)

        # Сохраняем в состояние, чтобы потом кнопки 👍/👎 могли записать результат
        USER_STATE[chat_id]["last_signal"] = {
            "pair": pair,
            "exp": exp,
            "direction": direction,
            "confidence": confidence
        }

        msg = f"""
📊 *Сигнал готов!*

💹 Пара: *{pair}*
⏱ Экспирация: *{exp} мин*

➡ Направление: *{direction}*
🔥 Уверенность: *{confidence}%*
"""
        context.bot.send_message(chat_id, msg, parse_mode="Markdown")

        # Кнопки 👍 и 👎
        keyboard = [
            [InlineKeyboardButton("👍 Профит", callback_data="result:+"),
             InlineKeyboardButton("👎 Лосс", callback_data="result:-")]
        ]
        context.bot.send_message(chat_id, "Отметьте результат сигнала:", reply_markup=InlineKeyboardMarkup(keyboard))

    except Exception as e:
        logger.error(f"Ошибка анализа: {e}")
        context.bot.send_message(chat_id, f"⚠ Ошибка анализа: {e}")

# =====================================
# Main
# =====================================
def main():
    if not TELEGRAM_TOKEN:
        logger.error("TELEGRAM_TOKEN не задан! Используй переменные окружения.")
        return

    updater = Updater(TELEGRAM_TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", main_menu))
    dp.add_handler(CallbackQueryHandler(button))

    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()
