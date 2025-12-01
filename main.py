import logging
import pandas as pd
import yfinance as yf
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application, ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes
)
from flask import Flask, request
import os
import re
import asyncio

logging.basicConfig(level=logging.INFO)

# =====================================================
#                ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ
# =====================================================
user_state = {}
trade_history = {}

ALL_PAIRS = [
    "EUR/USD","GBP/USD","USD/JPY","AUD/USD","USD/CAD","USD/CHF",
    "EUR/JPY","GBP/JPY","AUD/JPY","EUR/GBP","EUR/AUD","GBP/AUD",
    "CAD/JPY","CHF/JPY","EUR/CAD","GBP/CAD","AUD/CAD","AUD/CHF","CAD/CHF"
]
PAIRS_PER_PAGE = 6


# ---------- RSI (без pandas_ta) ----------
def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def escape_md(text: str):
    return re.sub(r"([_*\[\]()~`>#+\-=|{}.!])", r"\\\1", text)

def get_pairs_page(page):
    start = page * PAIRS_PER_PAGE
    end = start + PAIRS_PER_PAGE
    return ALL_PAIRS[start:end]

def total_pages():
    return (len(ALL_PAIRS) - 1) // PAIRS_PER_PAGE


# =============== /start ================================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("📈 Выбрать валютную пару", callback_data="choose_pair_0")],
        [InlineKeyboardButton("📜 История сделок", callback_data="history")]
    ]
    await update.message.reply_text(
        "👋 Привет! Я торговый бот.\n\nВыбери действие:",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )


# =============== Выбор пары ============================
async def choose_pair(update, context, page=0):
    q = update.callback_query
    await q.answer()

    pairs = get_pairs_page(page)
    keyboard = [[InlineKeyboardButton(p, callback_data=f"pair_{p}")] for p in pairs]

    nav = []
    if page > 0:
        nav.append(InlineKeyboardButton("⬅ Назад", callback_data=f"choose_pair_{page-1}"))
    if page < total_pages():
        nav.append(InlineKeyboardButton("Вперёд ➡", callback_data=f"choose_pair_{page+1}"))
    if nav:
        keyboard.append(nav)

    keyboard.append([InlineKeyboardButton("⬅ Главное меню", callback_data="back_to_menu")])

    await q.edit_message_text(
        "⚡ Выберите валютную пару:",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )


# =============== Экспирация ===============================
async def choose_expiration(update, context, pair):
    keyboard = [
        [InlineKeyboardButton("1 мин", callback_data=f"exp_1_{pair}")],
        [InlineKeyboardButton("3 мин", callback_data=f"exp_3_{pair}")],
        [InlineKeyboardButton("5 мин", callback_data=f"exp_5_{pair}")],
        [InlineKeyboardButton("10 мин", callback_data=f"exp_10_{pair}")],
        [InlineKeyboardButton("⬅ Назад", callback_data="choose_pair_0")]
    ]

    await update.callback_query.edit_message_text(
        f"Пара: *{escape_md(pair)}*\nВыберите экспирацию:",
        parse_mode="MarkdownV2",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )


# =============== Сигнал ===============================
def generate_signal(pair, timeframe):
    try:
        data = yf.download(pair.replace("/", ""), period="1d", interval="1m")
        if data.empty:
            return None
        data["rsi"] = rsi(data["Close"])
        val = data["rsi"].iloc[-1]
        if val < 30:
            return "⬆ CALL"
        elif val > 70:
            return "⬇ PUT"
        return "❕ Нет сигнала"
    except:
        return None


# =============== Ввод результата ======================
async def ask_result(update, context, pair, exp):
    q = update.callback_query
    uid = q.from_user.id

    signal = generate_signal(pair, exp)
    if not signal:
        await q.edit_message_text("❌ Не удалось получить сигнал.")
        return

    user_state[uid] = {"pair": pair, "exp": exp}

    k = [[
        InlineKeyboardButton("🟢 Плюс", callback_data="result_plus"),
        InlineKeyboardButton("🔴 Минус", callback_data="result_minus")
    ]]

    await q.edit_message_text(
        f"📊 Сигнал: *{escape_md(signal)}*\n"
        f"Пара: *{escape_md(pair)}*\n"
        f"Экспирация: *{exp} мин*",
        parse_mode="MarkdownV2",
        reply_markup=InlineKeyboardMarkup(k)
    )


# =============== Сохранение результата ===================
async def save_result(update, context, result):
    q = update.callback_query
    uid = q.from_user.id

    if uid not in trade_history:
        trade_history[uid] = []

    pair = user_state[uid]["pair"]
    exp = user_state[uid]["exp"]

    trade_history[uid].append(f"{pair} | {exp} мин — {result}")

    k = [
        [InlineKeyboardButton("📈 Новый сигнал", callback_data="choose_pair_0")],
        [InlineKeyboardButton("📜 История", callback_data="history")]
    ]

    await q.edit_message_text(
        f"Записано: *{escape_md(result)}*",
        parse_mode="MarkdownV2",
        reply_markup=InlineKeyboardMarkup(k)
    )


# =============== История =============================
async def history(update, context):
    q = update.callback_query
    uid = q.from_user.id

    if uid not in trade_history or len(trade_history[uid]) == 0:
        await q.edit_message_text("📭 История пустая.")
        return

    text = "📜 *История:*\n\n"
    for t in trade_history[uid]:
        text += f"• {escape_md(t)}\n"

    k = [[InlineKeyboardButton("⬅ Главное меню", callback_data="back_to_menu")]]

    await q.edit_message_text(
        text,
        parse_mode="MarkdownV2",
        reply_markup=InlineKeyboardMarkup(k)
    )


# =====================================================
async def callbacks(update: Update, context: ContextTypes.DEFAULT_TYPE):
    data = update.callback_query.data

    if data.startswith("choose_pair_"):
        await choose_pair(update, context, int(data.split("_")[2]))
    elif data.startswith("pair_"):
        await choose_expiration(update, context, data.split("_")[1])
    elif data.startswith("exp_"):
        _, exp, pair = data.split("_")
        await ask_result(update, context, pair, int(exp))
    elif data == "result_plus":
        await save_result(update, context, "Плюс")
    elif data == "result_minus":
        await save_result(update, context, "Минус")
    elif data == "history":
        await history(update, context)
    elif data == "back_to_menu":
        await start(update, context)


# ====================== FLASK + WEBHOOK ======================

BOT_TOKEN = os.getenv("BOT_TOKEN")     # ← ИСПРАВЛЕНО
WEBHOOK_URL = os.getenv("WEBHOOK_URL")

app = Flask(__name__)

application = ApplicationBuilder().token(BOT_TOKEN).build()
application.add_handler(CommandHandler("start", start))
application.add_handler(CallbackQueryHandler(callbacks))


@app.route("/", methods=["GET"])
def home():
    return "Bot is running"


@app.route(f"/webhook/{BOT_TOKEN}", methods=["POST"])
async def webhook():
    update = Update.de_json(request.get_json(force=True), application.bot)
    await application.process_update(update)
    return "OK", 200


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(
        application.bot.set_webhook(f"{WEBHOOK_URL}/webhook/{BOT_TOKEN}")
    )

    port = int(os.getenv("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
