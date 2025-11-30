import logging
import pandas as pd
import pandas_ta as ta
import yfinance as yf
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    ApplicationBuilder, CommandHandler, CallbackQueryHandler,
    ContextTypes
)
import os

logging.basicConfig(level=logging.INFO)

# =====================================================
#                ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ
# =====================================================
user_state = {}      # выбранная пара и время
trade_history = {}   # история сделок

# =====================================================
#                 /start команда
# =====================================================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("📈 Выбрать валютную пару", callback_data="choose_pair")],
        [InlineKeyboardButton("📜 История сделок", callback_data="history")]
    ]
    await update.message.reply_text(
        "👋 Привет! Я торговый бот.\n\n"
        "Выбери действие ниже:",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

# =====================================================
#              Выбор валютной пары
# =====================================================
async def choose_pair(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    keyboard = [
        [InlineKeyboardButton("EURUSD", callback_data="pair_EURUSD")],
        [InlineKeyboardButton("GBPUSD", callback_data="pair_GBPUSD")],
        [InlineKeyboardButton("USDJPY", callback_data="pair_USDJPY")],
        [InlineKeyboardButton("⬅ Назад", callback_data="back_to_menu")]
    ]
    await query.edit_message_text(
        "⚡ Выберите валютную пару:",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

# =====================================================
#          Выбор экспирации после выбора пары
# =====================================================
async def choose_expiration(update: Update, context: ContextTypes.DEFAULT_TYPE, pair):
    keyboard = [
        [InlineKeyboardButton("1 мин", callback_data=f"exp_1_{pair}")],
        [InlineKeyboardButton("3 мин", callback_data=f"exp_3_{pair}")],
        [InlineKeyboardButton("5 мин", callback_data=f"exp_5_{pair}")],
        [InlineKeyboardButton("10 мин", callback_data=f"exp_10_{pair}")],
        [InlineKeyboardButton("⬅ Назад", callback_data="choose_pair")]
    ]

    await update.callback_query.edit_message_text(
        f"Пара: *{pair}*\nТеперь выберите экспирацию:",
        parse_mode="Markdown",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

# =====================================================
#                 Генерация сигнала
# =====================================================
def generate_signal(pair, timeframe):
    try:
        data = yf.download(pair, period="1d", interval="1m")

        if data.empty:
            return None

        data["rsi"] = ta.rsi(data["Close"], length=14)
        last_rsi = data["rsi"].iloc[-1]

        if last_rsi < 30:
            return "⬆ CALL (покупка)"
        elif last_rsi > 70:
            return "⬇ PUT (продажа)"
        else:
            return "❕ Нет чёткого сигнала"

    except Exception as e:
        return None

# =====================================================
#            После сигнала → ПЛЮС / МИНУС
# =====================================================
async def ask_result(update: Update, context: ContextTypes.DEFAULT_TYPE, pair, expiration):
    query = update.callback_query
    user_id = query.from_user.id

    signal = generate_signal(pair, expiration)

    if not signal:
        await query.edit_message_text("❌ Не удалось получить сигнал.")
        return

    user_state[user_id] = {"pair": pair, "exp": expiration}

    keyboard = [
        [
            InlineKeyboardButton("🟢 Плюс", callback_data="result_plus"),
            InlineKeyboardButton("🔴 Минус", callback_data="result_minus")
        ]
    ]

    await query.edit_message_text(
        f"📊 Сигнал для *{pair}*\n"
        f"⏱ Экспирация: *{expiration} мин*\n"
        f"📈 Сигнал: *{signal}*\n\n"
        f"Отметьте результат:",
        parse_mode="Markdown",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

# =====================================================
#                 Запись результата
# =====================================================
async def save_result(update: Update, context: ContextTypes.DEFAULT_TYPE, result):
    query = update.callback_query
    user_id = query.from_user.id

    if user_id not in trade_history:
        trade_history[user_id] = []

    pair = user_state[user_id]["pair"]
    exp = user_state[user_id]["exp"]

    trade_history[user_id].append(f"{pair} | {exp} мин — {result}")

    keyboard = [
        [InlineKeyboardButton("📈 Сделать новый сигнал", callback_data="choose_pair")],
        [InlineKeyboardButton("📜 История", callback_data="history")]
    ]

    await query.edit_message_text(
        f"Записано: *{result}*\n\nВыберите действие:",
        parse_mode="Markdown",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

# =====================================================
#                     История
# =====================================================
async def history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user_id = query.from_user.id

    if user_id not in trade_history or len(trade_history[user_id]) == 0:
        await query.edit_message_text("📭 История пустая.")
        return

    text = "📜 *Ваша история сделок:*\n\n"
    for trade in trade_history[user_id]:
        text += f"• {trade}\n"

    keyboard = [[InlineKeyboardButton("⬅ Назад", callback_data="back_to_menu")]]

    await query.edit_message_text(
        text,
        parse_mode="Markdown",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

# =====================================================
#                Обработчик Callback
# =====================================================
async def callbacks(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    data = query.data

    if data == "choose_pair":
        await choose_pair(update, context)

    elif data.startswith("pair_"):
        pair = data.split("_")[1]
        await choose_expiration(update, context, pair)

    elif data.startswith("exp_"):
        _, exp, pair = data.split("_")
        await ask_result(update, context, pair, int(exp))

    elif data == "result_plus":
        await save_result(update, context, "🟢 Плюс")

    elif data == "result_minus":
        await save_result(update, context, "🔴 Минус")

    elif data == "history":
        await history(update, context)

    elif data == "back_to_menu":
        await start(update, context)

# =====================================================
#                        MAIN
# =====================================================
TOKEN = os.getenv("TELEGRAM_TOKEN")

application = ApplicationBuilder().token(TOKEN).build()

application.add_handler(CommandHandler("start", start))
application.add_handler(CallbackQueryHandler(callbacks))

print("Бот запущен и работает через polling...")
application.run_polling()
