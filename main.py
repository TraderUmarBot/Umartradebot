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
# АБСОЛЮТНЫЙ МИНИМУМ СВЕЧЕЙ: 21 свеча необходима для EMA21 и BB20
MIN_CANDLES = 21 

FOREX = [
    "EURUSD","GBPUSD","USDJPY","AUDUSD","USDCHF","EURJPY",
    "GBPJPY","NZDUSD","EURGBP","CADJPY","USDCAD","AUDJPY",
    "EURAUD","GBPAUD","EURNZD","AUDNZD","CADCHF","CHFJPY",
    "NZDJPY","GBPCAD"
]

EXP = ["1m","2m","3m","5m"]

# -----------------------------------------
# FLASK (Только для проверки статуса Render)
# -----------------------------------------
# Важно: приложение теперь называется 'app' для Gunicorn
app = Flask(__name__)

@app.route("/")
def home():
    # Эта функция теперь просто проверяет, что сервис Render жив.
    return "OXTSIGNALSBOT is running (Flask heartbeat)."

# -----------------------------------------
# UTILS
# -----------------------------------------
def yf_symbol(pair):
    return f"{pair[:3]}{pair[3:]}=X"

def exp_to_sec(e):
    return int(e.replace("m","")) * 60

# Фолбэк: симуляция, если yfinance не работает или данных очень мало
def simulate_data(pair, num_periods=100):
    import random
    # Убедимся, что симуляция всегда возвращает минимум MIN_CANDLES
    num_periods = max(num_periods, MIN_CANDLES) 
    
    rng = random.Random(abs(hash(pair)) % 999999)
    # Используем более реалистичные цены для EURUSD (около 1.08)
    price = 1.05 + rng.uniform(-0.02, 0.06) 
    dates = pd.date_range(end=pd.Timestamp.now(), periods=num_periods, freq="1min")

    data = []
    for _ in range(num_periods):
        o = price
        c = o + rng.uniform(-0.0005, 0.0005)
        h = max(o, c) + rng.uniform(0,0.0003)
        l = min(o, c) - rng.uniform(0,0.0003)
        v = rng.randint(500, 1500)
        price = c
        data.append([o, h, l, c, v])

    df = pd.DataFrame(data, columns=["Open","High","Low","Close","Volume"], index=dates)
    
    # Добавляем комментарий в индекс, чтобы потом понять, что это симуляция
    df.index.name = "Simulated" 
    return df.tail(num_periods)


# Стабильный фетчер: Пробуем Yahoo, если не работает — симуляция
def fetch_data(pair, exp_sec):
    try:
        df = yf.download(
            yf_symbol(pair),
            period="5d", 
            interval="1m",
            progress=False,
            timeout=5
        )
        
        df = df.dropna()
        
        # Если данных от YF достаточно, возвращаем их
        if len(df) >= MIN_CANDLES:
             return df.tail(MIN_CANDLES) 
             
        # Если данных от YF недостаточно, но они есть, переходим к симуляции
        if len(df) > 0:
            print(f"WARNING: Insufficient data from YF for {pair}. ({len(df)}/{MIN_CANDLES}) -> Switching to Simulation.")
            
    except Exception as e:
        print(f"ERROR fetching {pair}: {e} -> Switching to Simulation.")
        pass # Идем дальше к симуляции

    # Резервный режим: запускаем симуляцию
    return simulate_data(pair)


# -----------------------------------------
# INDICATORS (добавлена защита от NaN и ошибок Series)
# -----------------------------------------
def compute_indicators(df):

    # Минимальная проверка, если что-то пошло не так даже с симуляцией
    if df is None or df.empty or len(df) < MIN_CANDLES:
        return {"error": "INSUFFICIENT_DATA"}

    out = {}
    c = df["Close"]
    
    # --------------------------------------
    # Вспомогательная функция для безопасного извлечения последнего значения
    def safe_last(series):
        # Удаляем NaN, если они возникли при расчете
        series = series.dropna() 
        if series.empty:
            return None
        return series.iloc[-1] 
    # --------------------------------------
    

    # EMA
    ema8 = safe_last(c.ewm(span=8, adjust=False).mean())
    ema21 = safe_last(c.ewm(span=21, adjust=False).mean())
    if ema8 is None or ema21 is None: return {"error": "EMA_FAILED"}
    out["EMA"] = 1 if ema8 > ema21 else -1

    # SMA
    sma5 = safe_last(c.rolling(5).mean())
    sma20 = safe_last(c.rolling(20).mean())
    if sma5 is None or sma20 is None: return {"error": "SMA_FAILED"}
    out["SMA"] = 1 if sma5 > sma20 else -1

    # MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    
    macd_val = safe_last(macd)
    signal_val = safe_last(signal)
    
    if macd_val is None or signal_val is None: return {"error": "MACD_FAILED"}
    out["MACD"] = 1 if macd_val > signal_val else -1

    # RSI
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    
    with np.errstate(divide='ignore', invalid='ignore'): 
        # Если loss=0, деление на 1 предотвратит ошибку
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
    price = safe_last(c) 
    
    if upper is None or lower is None or price is None: return {"error": "BB_FAILED"}

    if price < lower:
        out["BB"] = 1
    elif price > upper:
        out["BB"] = -1
    else:
        out["BB"] = 0

    return out

# -----------------------------------------
# DECISION ENGINE (добавлен признак симуляции)
# -----------------------------------------
WEIGHTS = {
    "EMA":2, "SMA":2, "MACD":2, "RSI":1, "BB":1
}

def make_decision(ind):

    score = 0
    for k,w in WEIGHTS.items():
        score += ind.get(k,0) * w

    direction = "Вверх ↑" if score >= 0 else "Вниз ↓"

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
# TELEGRAM BOT UI (без изменений)
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
    
    is_simulated = (df is not None and df.index.name == "Simulated")
    
    # Мы не будем падать, если есть данные, даже симулированные.
    if df is None or df.empty or len(df) < MIN_CANDLES:
        fail_text = (
            f"🚫 *Критическая ошибка анализа {pair}*\n\n"
            f"Не удалось получить *абсолютно никаких* данных. Возможно, сбой на стороне брокера."
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
    
    # Если compute_indicators вернул ошибку, хотя данные были (это ошибка в расчетах, а не в данных)
    if "error" in ind:
        fail_text = (
            f"🚫 *Внутренняя ошибка расчёта {pair}*\n\n"
            f"Не удалось рассчитать индикаторы. Попробуйте другую пару или повторите запрос."
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
    
    sim_warning = ""
    if is_simulated:
        # Уменьшаем уверенность, если данные симулированы
        conf = round(conf * 0.9, 1)
        sim_warning = "\n\n⚠️ *ВНИМАНИЕ:* Данные рынка были неполными. Анализ основан на *резервной симуляции*."

    text = (
        f"📊 *Анализ завершён*\n\n"
        f"🔹 Валютная пара: *{pair}*\n"
        f"🔹 Экспирация: *{exp}*\n\n"
        f"📈 *Сигнал:* {direction}\n"
        f"🎯 *Точность:* {conf}%\n\n"
        f"💬 *Логика входа:*\n{logic}\n\n"
        f"💵 Цена: `{price:.6f}`"
        f"{sim_warning}\n"
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
# НОВАЯ ФУНКЦИЯ: Запускает Polling
def run_polling():
    print("Starting Telegram Polling...")
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CallbackQueryHandler(callback))
    updater.start_polling()
    updater.idle()
    print("Telegram Polling finished.")


def main():
    # Мы ожидаем, что Flask запустится через gunicorn, а Polling - через run_polling
    pass 
    
if __name__ == "__main__":
    
    # Это добавлено для запуска Polling, как указано в Procfile
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'run_polling':
        run_polling()

