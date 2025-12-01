import logging
import pandas as pd
import pandas_ta as ta
import yfinance as yf
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    ApplicationBuilder, CommandHandler, CallbackQueryHandler,
    ContextTypes
)
from flask import Flask, request
import os
import re

logging.basicConfig(level=logging.INFO)

# =====================================================
#                ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ
# =====================================================
user_state = {}      # выбранная пара и время
trade_history = {}   # история сделок

# Список всех валютных пар
ALL_PAIRS = [
    "EUR/USD","GBP/USD","USD/JPY","AUD/USD","USD/CAD","USD/CHF",
    "EUR/JPY","GBP/JPY","AUD/JPY","EUR/GBP","EUR/AUD","GBP/AUD",
    "CAD/JPY","CHF/JPY","EUR/CAD","GBP/CAD","AUD/CAD","AUD/CHF","CAD/CHF"
]

PAIRS_PER_PAGE = 6  # сколько пар показывать на одной "странице"

# =====================================================
#                Вспомогательные функции
# =====================================================
def escape_markdown(text: str) -> str:
    """Экранируем спецсимволы для MarkdownV2"""
    return re.sub(r'([_*[\]()~`>#+-=|{}.!])', r'\\\1', text)

def get_pairs_page(page: int):
    start = page * PAIRS_PER_PAGE
    end = start + PAIRS_PER_PAGE
    return ALL_PAIRS[start:end]

def total_pages():
    return (len(ALL_PAIRS) - 1) // PAIRS_PER_PAGE

# =====================================================
#                 /start команда
# =====================================================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("📈 Выбрать валютную пару", callback_data="choose_pair_0")],
        [InlineKeyboardButton("📜 История сделок", callback_data="history")]
    ]
    await update.message.reply_text(
        "👋 Привет! Я торговый бот.\n\nВыбери действие ниже:",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

# =====================================================
#           Выбор валютной пары с пагинацией
# =====================================================
async def choose_pair(update: Update, context: ContextTypes.DEFAULT_TYPE, page=0):
    query = update.callback_query
    await query.answer()

    pairs = get_pairs_page(page)
    keyboard = [[InlineKeyboardButton(pair, callback_data=f"pair_{pair}")] for pair in pairs]

    nav_buttons = []
    if page > 0:
        nav_buttons.append(InlineKeyboardButton("⬅ Назад", callback_data=f"choose_pair_{page-1}"))
    if page < total_pages():
        nav_buttons.append(InlineKeyboardButton("Вперёд ➡", callback_data=f"choose_pair_{page+1}"))
    if nav_buttons:
        keyboard.append(nav_buttons)

    keyboard.append([InlineKeyboardButton("⬅ Главное меню", callback_data="back_to_menu")])

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
        [InlineKeyboardButton("⬅ Назад", callback_data="choose_pair_0")]
    ]

    await update.callback_query.edit_message_text(
        f"Пара: *{escape_markdown(pair)}*\nТеперь выберите экспирацию:",
        parse_mode="MarkdownV2",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

# =====================================================
#                 Генерация сигнала
# =====================================================
def generate_signal(pair, timeframe):
    try:
        data = yf.download(pair.replace("/", ""), period="1d", interval="1m")
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
    except:
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
        f"📊 Сигнал для *{escape_markdown(pair)}*\n⏱ Экспирация: *{expiration} мин*\n📈 Сигнал: *{escape_markdown(signal)}*\n\nОтметьте результат:",
        parse_mode="MarkdownV2",
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
        [InlineKeyboardButton("📈 Сделать новый сигнал", callback_data="choose_pair_0")],
        [InlineKeyboardButton("📜 История", callback_data="history")]
    ]

    await query.edit_message_text(
        f"Записано: *{escape_markdown(result)}*\n\nВыберите действие:",
        parse_mode="MarkdownV2",
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
        text += f"• {escape_markdown(trade)}\n"

    keyboard = [[InlineKeyboardButton("⬅ Главное меню", callback_data="back_to_menu")]]

    await query.edit_message_text(
        text,
        parse_mode="MarkdownV2",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

# =====================================================
#                Обработчик Callback
# =====================================================
async def callbacks(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    data = query.data

    if data.startswith("choose_pair_"):
        page = int(data.split("_")[2])
        await choose_pair(update, context, page)
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
TOKEN = os.getenv("BOT_TOKEN")
WEBHOOK_URL = os.getenv("WEBHOOK_URL")  # ставим URL вашего Render сервиса

app = Flask(__name__)
application = ApplicationBuilder().token(TOKEN).build()
application.add_handler(CommandHandler("start", start))
application.add_handler(CallbackQueryHandler(callbacks))

@app.route(f"/{TOKEN}", methods=["POST"])
def webhook():
    """Обрабатываем POST-запрос от Telegram"""
    update = Update.de_json(request.get_json(force=True), application.bot)
    application.update_queue.put(update)
    return "OK"

if __name__ == "__main__":
    application.bot.set_webhook(WEBHOOK_URL + "/" + TOKEN)
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
