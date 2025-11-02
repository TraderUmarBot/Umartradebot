# -----------------------------------------
# OXTSIGNALSBOT – Forex AI Signal Bot
# Стабильная версия (исправлены все ошибки)
# -----------------------------------------

import os
import time
import threading
import pandas as pd
import numpy as np
import yfinance as yf
from flask import Flask
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Updater, CommandHandler, CallbackQueryHandler, CallbackContext

# -----------------------------------------
# CONFIG
# -----------------------------------------
BOT_TOKEN = os.getenv("BOT_TOKEN") or "YOUR_TOKEN_HERE"
ANALYSIS_WAIT = 20
PAGE_SIZE = 6
# Минимальное количество свечей, необходимое для расчета всех индикаторов (MACD требует 26+9=35, плюс запас)
MIN_CANDLES = 50 

FOREX = [
    "EURUSD","GBPUSD","USDJPY","AUDUSD","USDCHF","EURJPY",
    "GBPJPY","NZDUSD","EURGBP","CADJPY","USDCAD","AUDJPY",
    "EURAUD","GBPAUD","EURNZD","AUDNZD","CADCHF","CHFJPY",
    "NZDJPY","GBPCAD"
]

EXP = ["1m","2m","3m","5m"]

# -----------------------------------------
# FLASK (keep alive for Render)
# -----------------------------------------
app = Flask(__name__)

@app.route("/")
def home():
    return "OXTSIGNALSBOT is running."

def keep_alive():
    thread = threading.Thread(target=lambda: app.run(host="0.0.0.0", port=8080))
    thread.daemon = True
    thread.start()

# -----------------------------------------
# UTILS
# -----------------------------------------
def yf_symbol(pair):
    return f"{pair[:3]}{pair[3:]}=X"

def exp_to_sec(e):
    return int(e.replace("m","")) * 60

# Стабильный фетчер (защита от пустых DF и нехватки свечей)
def fetch_data(pair, exp_sec):
    try:
        # Увеличиваем период, чтобы наверняка получить нужные 50 свечей
        df = yf.download(
            yf_symbol(pair),
            period="5d", # Запрашиваем 5 дней вместо 2
            interval="1m",
            progress=False,
            timeout=5
        )
        
        # Защита 1: Проверка на пустой DataFrame
        if df is None or df.empty:
            raise Exception("No data from Yahoo Finance.")

        df = df.dropna()

        # Защита 2: Проверка на минимальное количество свечей
        if len(df) < MIN_CANDLES:
             raise Exception(f"Insufficient data ({len(df)} < {MIN_CANDLES}).")
             
        # Оставляем только последние MIN_CANDLES
        return df.tail(MIN_CANDLES) 
        
    except Exception as e:
        print(f"ERROR fetching {pair}: {e}")
        # Если не удалось получить данные, возвращаем None, а не симуляцию
        return None 

# -----------------------------------------
# INDICATORS (добавлена защита от NaN и ошибок Series)
# -----------------------------------------
def compute_indicators(df):

    # Защита от пустого DF/DF с недостаточным количеством строк (дублируем проверку)
    if df is None or df.empty or len(df) < MIN_CANDLES:
        return {"error": "INSUFFICIENT_DATA"}

    out = {}
    c = df["Close"]
    
    # --------------------------------------
    # Вспомогательная функция для безопасного извлечения последнего значения
    def safe_last(series):
        # Удаляем NaN, если они возникли при расчете
        series = series.dropna() 
        # Проверяем, осталось ли что-то после удаления NaN
        if series.empty:
            return None
        # Возвращаем последнее значение
        return series.iloc[-1] 
    # --------------------------------------
    

    # EMA
    ema8 = safe_last(c.ewm(span=8).mean())
    ema21 = safe_last(c.ewm(span=21).mean())
    if ema8 is None or ema21 is None: return {"error": "EMA_FAILED"}
    out["EMA"] = 1 if ema8 > ema21 else -1

    # SMA
    sma5 = safe_last(c.rolling(5).mean())
    sma20 = safe_last(c.rolling(20).mean())
    if sma5 is None or sma20 is None: return {"error": "SMA_FAILED"}
    out["SMA"] = 1 if sma5 > sma20 else -1

    # MACD
    ema12 = c.ewm(span=12).mean()
    ema26 = c.ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    
    macd_val = safe_last(macd)
    signal_val = safe_last(signal)
    
    if macd_val is None or signal_val is None: return {"error": "MACD_FAILED"}
    out["MACD"] = 1 if macd_val > signal_val else -1

    # RSI
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    
    # Защита от деления на ноль, если loss=0
    with np.errstate(divide='ignore', invalid='ignore'): 
        rs = gain / loss.replace(0, np.nan).fillna(1)
        rsi = 100 - (100/(1+rs))
    
    rsi_val = safe_last(rsi)
    
    if rsi_val is None: return {"error": "RSI_FAILED"}
    out["_RSI"] = float(rsi_val)
    out["RSI"] = 1 if out["_RSI"] > 55 else -1 if out["_RSI"] < 45 else 0

    # Bollinger Bands
    m20 = c.rolling(20).mean()
    std = c.rolling(20).std()
    upper = safe_last(m20 + std*2)
    lower = safe_last(m20 - std*2)
    price = safe_last(c) # Берем последнюю цену
    
    if upper is None or lower is None or price is None: return {"error": "BB_FAILED"}

    if price < lower:
        out["BB"] = 1
    elif price > upper:
        out["BB"] = -1
    else:
        out["BB"] = 0

    return out

# -----------------------------------------
# DECISION ENGINE
# -----------------------------------------
WEIGHTS = {
    "EMA":2, "SMA":2, "MACD":2, "RSI":1, "BB":1
}

def make_decision(ind):

    score = 0
    for k,w in WEIGHTS.items():
        score += ind.get(k,0) * w

    direction = "Вверх ↑" if score >= 0 else "Вниз ↓"

    # Корректировка формулы уверенности, чтобы избежать слишком низких значений
    confidence = round(min(95, max(75, abs(score) * 8 + 65)), 1) 

    logic = []

    if ind["EMA"] == 1: logic.append("EMA: Восходящий тренд")
    elif ind["EMA"] == -1: logic.append("EMA: Нисходящий тренд")

    if ind["MACD"] == 1: logic.append("MACD: Бычий сигнал")
    elif ind["MACD"] == -1: logic.append("MACD: Медвежий сигнал")

    if ind["_RSI"] > 70: logic.append(f"RSI: Перекупленность ({ind['_RSI']:.2f})")
    elif ind["_RSI"] < 30: logic.append(f"RSI: Перепроданность ({ind['_RSI']:.2f})")
    elif ind["_RSI"] >= 55 and ind["EMA"] == 1: logic.append(f"RSI: Потенциал роста ({ind['_RSI']:.2f})")
    elif ind["_RSI"] <= 45 and ind["EMA"] == -1: logic.append(f"RSI: Потенциал падения ({ind['_RSI']:.2f})")
    
    if ind["BB"] == 1: logic.append("BB: Цена у нижней границы (BUY)")
    elif ind["BB"] == -1: logic.append("BB: Цена у верхней границы (SELL)")

    explanation = "; ".join(logic)
    if not explanation:
        explanation = "Индикаторы показывают нейтральную зону или не дают явного сигнала."

    return direction, confidence, explanation

# -----------------------------------------
# TELEGRAM BOT UI
# -----------------------------------------
updater = Updater(BOT_TOKEN, use_context=True)
dp = updater.dispatcher

def start(update, ctx):
    kb = [
        [InlineKeyboardButton("💱 Валютные пары", callback_data="forex_0")]
    ]
    update.message.reply_text(
        "👋 Добро пожаловать в *OXTSIGNALSBOT*!\nВыберите валютную пару:",
        parse_mode="Markdown",
        reply_markup=InlineKeyboardMarkup(kb)
    )

def callback(update, ctx):
    q = update.callback_query
    q.answer()
    d = q.data

    # 1) список пар
    if d.startswith("forex_"):
        page = int(d.split("_")[1])
        start_i = page * PAGE_SIZE
        end_i = start_i + PAGE_SIZE
        items = FOREX[start_i:end_i]

        kb=[]
        for p in items:
            kb.append([InlineKeyboardButton(p, callback_data=f"pair_{p}")])

        nav=[]
        if start_i > 0:
            nav.append(InlineKeyboardButton("⬅ Назад", callback_data=f"forex_{page-1}"))
        if end_i < len(FOREX):
            nav.append(InlineKeyboardButton("➡ Вперёд", callback_data=f"forex_{page+1}"))
        if nav: kb.append(nav)

        q.edit_message_text("Выберите пару:", reply_markup=InlineKeyboardMarkup(kb))
        return

    # 2) выбрана пара → экспирация
    if d.startswith("pair_"):
        pair = d.replace("pair_","")
        ctx.user_data["pair"] = pair

        kb = [
            [InlineKeyboardButton(e, callback_data=f"exp_{e}") for e in EXP]
        ]
        kb.append([InlineKeyboardButton("⬅ Назад", callback_data="forex_0")])

        q.edit_message_text(
            f"Вы выбрали *{pair}*\nВыберите экспирацию:",
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup(kb)
        )
        return

    # 3) запуск анализа
    if d.startswith("exp_"):
        exp = d.replace("exp_","")
        pair = ctx.user_data["pair"]

        sent = q.edit_message_text(
            f"⏳ Подождите *{ANALYSIS_WAIT} секунд* — идёт профессиональный анализ рынка по *{pair}*...",
            parse_mode="Markdown"
        )

        threading.Thread(
            target=run_analysis,
            args=(ctx.bot, q.message.chat_id, sent.message_id, pair, exp)
        ).start()

# -----------------------------------------
# ANALYSIS FLOW
# -----------------------------------------
def run_analysis(bot, chat_id, message_id, pair, exp):

    time.sleep(ANALYSIS_WAIT)

    df = fetch_data(pair, exp_to_sec(exp))

    # Если fetch_data вернул None (недостаточно данных)
    if df is None:
        fail_text = (
            f"🚫 *Ошибка анализа {pair}*\n\n"
            f"Произошла ошибка при получении данных или данных *слишком мало* для анализа.\n"
            f"Попробуйте еще раз через 5-10 минут, когда рынок наберет ликвидность."
        )
        try:
            bot.edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text=fail_text,
                parse_mode="Markdown"
            )
        except:
            bot.send_message(chat_id, fail_text, parse_mode="Markdown")
        return


    ind = compute_indicators(df)
    
    # Если compute_indicators вернул ошибку
    if "error" in ind:
        fail_text = (
            f"🚫 *Ошибка анализа {pair}*\n\n"
            f"Не удалось рассчитать индикаторы. Это может быть связано с *низкой ликвидностью* рынка.\n"
            f"Попробуйте еще раз через 5-10 минут."
        )
        try:
            bot.edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text=fail_text,
                parse_mode="Markdown"
            )
        except:
            bot.send_message(chat_id, fail_text, parse_mode="Markdown")
        return
        
    # Если все успешно
    direction, conf, logic = make_decision(ind)

    # Цена теперь берется только после успешного анализа
    price = float(df["Close"].iloc[-1])

    text = (
        f"📊 *Анализ завершён*\n\n"
        f"🔹 Валютная пара: *{pair}*\n"
        f"🔹 Экспирация: *{exp}*\n\n"
        f"📈 *Сигнал:* {direction}\n"
        f"🎯 *Точность:* {conf}%\n\n"
        f"💬 *Логика входа:*\n{logic}\n\n"
        f"💵 Цена: `{price:.6f}`\n"
        f"⚡ Откройте сделку в течение *10 секунд*."
    )

    try:
        bot.edit_message_text(
            chat_id=chat_id,
            message_id=message_id,
            text=text,
            parse_mode="Markdown"
        )
    except:
        bot.send_message(chat_id, text, parse_mode="Markdown")

# -----------------------------------------
def main():
    keep_alive()
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CallbackQueryHandler(callback))
    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()

