import asyncio
import pandas as pd
import yfinance as yf
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ApplicationBuilder, CallbackQueryHandler, CommandHandler, ContextTypes
import datetime
import random

# Токен твоего бота
BOT_TOKEN = "8316818247:AAEZYEhSxDeixKNGvY2G4HYQEjdfaj5Un54"

# Валютные пары для анализа
PAIRS = [
    "EURUSD=X","GBPUSD=X","USDJPY=X","AUDUSD=X","USDCAD=X","USDCHF=X",
    "EURJPY=X","GBPJPY=X","AUDJPY=X","EURGBP=X","EURAUD=X","GBPAUD=X",
    "CADJPY=X","CHFJPY=X","EURCAD=X","GBPCAD=X","AUDCAD=X","AUDCHF=X","CADCHF=X"
]

# История сигналов
signal_history = []

# Простейший анализ RSI
def analyze_rsi(df):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Простейший паттерн анализа свечей
def candle_pattern(df):
    patterns = []
    last = df.iloc[-1]
    if last['Close'] > last['Open']:
        patterns.append("bullish")
    else:
        patterns.append("bearish")
    return patterns

# Определение направления сигнала
def generate_signal(df):
    rsi = analyze_rsi(df).iloc[-1]
    patterns = candle_pattern(df)
    
    # Простое правило: RSI + свечной паттерн
    if rsi < 30 and "bullish" in patterns:
        return "ВВЕРХ", random.randint(70, 95)
    elif rsi > 70 and "bearish" in patterns:
        return "ВНИЗ", random.randint(70, 95)
    else:
        return None, 0

# Получение исторических данных
async def get_data(pair, interval="1m", period="1d"):
    try:
        df = yf.download(pair, period=period, interval=interval, progress=False, auto_adjust=True)
        return df
    except Exception as e:
        print(f"Ошибка при загрузке {pair}: {e}")
        return None

# Основная функция генерации сигнала
async def send_signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    pair = random.choice(PAIRS)
    df = await get_data(pair)
    if df is None or df.empty:
        await update.message.reply_text(f"Не удалось получить данные для {pair}")
        return
    
    direction, confidence = generate_signal(df)
    if not direction:
        await update.message.reply_text(f"Сигнал для {pair} не найден.")
        return
    
    # Время экспирации от 1 до 15 минут
    expiry = random.randint(1, 15)
    
    keyboard = [
        [
            InlineKeyboardButton("+", callback_data=f"up_{pair}"),
            InlineKeyboardButton("-", callback_data=f"down_{pair}")
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    message = (
        f"📊 Валютная пара: {pair}\n"
        f"⏱ Время экспирации: {expiry} мин.\n"
        f"📈 Сигнал: {direction}\n"
        f"💯 Уверенность: {confidence}%"
    )
    
    await update.message.reply_text(message, reply_markup=reply_markup)
    
    # Сохраняем в историю
    signal_history.append({
        "pair": pair,
        "direction": direction,
        "expiry": expiry,
        "confidence": confidence,
        "time": datetime.datetime.now()
    })

# Обработка кнопок +/-
async def button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    action, pair = query.data.split("_")
    for s in signal_history:
        if s["pair"] == pair:
            s["feedback"] = action
            break
    await query.edit_message_text(text=f"Спасибо! Ваша оценка: {action}")

# Команда /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Привет! Нажми /signal чтобы получить торговый сигнал.")

# Команда /signal
async def signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await send_signal(update, context)

# Запуск бота
async def main():
    app = ApplicationBuilder().token(BOT_TOKEN).concurrent_updates(True).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("signal", signal))
    app.add_handler(CallbackQueryHandler(button))
    await app.run_polling()

if __name__ == "__main__":
    asyncio.run(main())
