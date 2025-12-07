import os
import logging
import pandas as pd
import yfinance as yf
import asyncio
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIG
# ============================================================================
EXCHANGE_PAIRS = [
    "EUR/USD","GBP/USD","USD/JPY","AUD/USD","USD/CAD","USD/CHF",
    "EUR/JPY","GBP/JPY","AUD/JPY","EUR/GBP","EUR/AUD","GBP/AUD",
    "CAD/JPY","CHF/JPY","EUR/CAD","GBP/CAD","AUD/CAD","AUD/CHF","CAD/CHF"
]

EXPIRE_LIST = ["1m", "3m", "5m", "10m"]
INTERVAL_MAP = {"3m": "2m", "10m": "5m"}

PAIRS_PER_PAGE = 6
LOOKBACK = 120

user_state = {}
trade_history = {}

# ============================================================================
# INDICATORS
# ============================================================================
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period, min_periods=1).mean()
    avg_loss = loss.rolling(period, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-9)
    return 100 - (100 / (1 + rs))

def SMA(series: pd.Series, period=50):
    return series.rolling(period, min_periods=1).mean()

def EMA(series: pd.Series, period=20):
    return series.ewm(span=period, adjust=False).mean()

def MACD(series: pd.Series):
    fast = series.ewm(span=12, adjust=False).mean()
    slow = series.ewm(span=26, adjust=False).mean()
    macd = fast - slow
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal


# ============================================================================
# HELPERS
# ============================================================================
def escape_md(text: str) -> str:
    import re
    return re.sub(r"([_*\[\]()~`>#+\-=|{}.!])", r"\\\1", str(text))

def get_page_items(page, items):
    start = page * PAIRS_PER_PAGE
    end = start + PAIRS_PER_PAGE
    return items[start:end]

def total_pages(items):
    return (len(items) - 1) // PAIRS_PER_PAGE

def yf_interval(expire: str):
    return INTERVAL_MAP.get(expire, expire)

# ============================================================================
# MENU
# ============================================================================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await show_main_menu(update, context)

async def show_main_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    kb = [
        [InlineKeyboardButton("📈 Биржевой рынок", callback_data="market")],
        [InlineKeyboardButton("📜 История", callback_data="history")],
    ]

    if update.message:
        await update.message.reply_text("Выберите действие:", reply_markup=InlineKeyboardMarkup(kb))
    else:
        q = update.callback_query
        await q.answer()
        await q.edit_message_text("Выберите действие:", reply_markup=InlineKeyboardMarkup(kb))


async def choose_pair(update: Update, context: ContextTypes.DEFAULT_TYPE, page=0):
    q = update.callback_query
    await q.answer()

    pairs = EXCHANGE_PAIRS
    page_items = get_page_items(page, pairs)

    kb = [[InlineKeyboardButton(p, callback_data=f"pair|{p}")] for p in page_items]

    nav = []
    if page > 0:
        nav.append(InlineKeyboardButton("⬅ Назад", callback_data=f"page|{page-1}"))
    if page < total_pages(pairs):
        nav.append(InlineKeyboardButton("Вперед ➡", callback_data=f"page|{page+1}"))

    if nav:
        kb.append(nav)

    kb.append([InlineKeyboardButton("⬅ Главное меню", callback_data="back")])

    await q.edit_message_text("Выберите валютную пару:", reply_markup=InlineKeyboardMarkup(kb))


async def choose_expiry(update: Update, context: ContextTypes.DEFAULT_TYPE, pair):
    q = update.callback_query
    await q.answer()

    kb = [[InlineKeyboardButton(exp, callback_data=f"analyze|{pair}|{exp}")]
          for exp in EXPIRE_LIST]
    kb.append([InlineKeyboardButton("⬅ Назад", callback_data="market")])
    kb.append([InlineKeyboardButton("⬅ Главное меню", callback_data="back")])

    await q.edit_message_text(f"Пара: {pair}\nВыберите таймфрейм:", reply_markup=InlineKeyboardMarkup(kb))

# ============================================================================
# ANALYSIS
# ============================================================================
async def analyze_signal(pair: str, expiry: str):
    ticker = pair.replace("/", "") + "=X"
    interval = yf_interval(expiry)

    # retry загрузки
    for attempt in range(3):
        try:
            df = yf.download(ticker, period="5d", interval=interval, progress=False, threads=False)
            break
        except Exception:
            if attempt < 2:
                await asyncio.sleep(1)
            else:
                return None, ["YFinance error"]

    if df.empty or len(df) < 10:
        return None, ["Недостаточно данных"]

    df = df.tail(LOOKBACK).copy()
    df["rsi"] = rsi(df["Close"])
    df["sma"] = SMA(df["Close"])
    df["ema"] = EMA(df["Close"])
    macd, macd_sig = MACD(df["Close"])
    df["macd"] = macd
    df["signal"] = macd_sig

    last = df.iloc[-1]

    rsi_val = float(last["rsi"])
    close = float(last["Close"])
    sma_val = float(last["sma"])
    ema_val = float(last["ema"])
    macd_val = float(last["macd"])
    signal_val = float(last["signal"])

    notes = []
    buy = 0
    sell = 0

    # RSI
    if rsi_val < 30:
        buy += 1
        notes.append("RSI oversold")
    elif rsi_val > 70:
        sell += 1
        notes.append("RSI overbought")

    # MACD
    if macd_val > signal_val:
        buy += 1
        notes.append("MACD bull")
    else:
        sell += 1
        notes.append("MACD bear")

    # Trend
    if close > sma_val:
        buy += 1
        notes.append("Above SMA")
    else:
        sell += 1
        notes.append("Below SMA")

    if close > ema_val:
        buy += 1
        notes.append("Above EMA")
    else:
        sell += 1
        notes.append("Below EMA")

    if buy > sell:
        direction = "Вверх"
    elif sell > buy:
        direction = "Вниз"
    else:
        direction = "Нет сигнала"

    confidence = min(95, 55 + max(buy, sell) * 10)

    return {
        "direction": direction,
        "confidence": confidence,
        "notes": notes,
    }, notes


async def show_signal(update: Update, context: ContextTypes.DEFAULT_TYPE, pair, expiry):
    q = update.callback_query
    await q.answer()

    msg = await q.edit_message_text("Анализирую...")

    result, notes = await analyze_signal(pair, expiry)
    if not result:
        await msg.edit_text("Ошибка анализа: " + " | ".join(notes))
        return

    user_state[q.from_user.id] = {"pair": pair}

    text = (
        f"📊 Сигнал: *{escape_md(result['direction'])}*\n"
        f"Пара: *{escape_md(pair)}*\n"
        f"Экспирация: *{expiry}*\n"
        f"Уверенность: {result['confidence']}%\n"
        f"Примечания: {escape_md(' | '.join(result['notes']))}"
    )

    kb = [
        [
            InlineKeyboardButton("🟢 Плюс", callback_data="res|plus"),
            InlineKeyboardButton("🔴 Минус", callback_data="res|minus"),
        ],
        [InlineKeyboardButton("⬅ Главное меню", callback_data="back")]
    ]

    await msg.edit_text(text, parse_mode="MarkdownV2", reply_markup=InlineKeyboardMarkup(kb))

# ============================================================================
# SAVE RESULT
# ============================================================================
async def save_result(update: Update, context: ContextTypes.DEFAULT_TYPE, label):
    q = update.callback_query
    await q.answer()

    uid = q.from_user.id
    pair = user_state.get(uid, {}).get("pair", "???")

    trade_history.setdefault(uid, []).append(f"{pair} — {label}")

    kb = [
        [InlineKeyboardButton("📈 Новый сигнал", callback_data="market")],
        [InlineKeyboardButton("⬅ Главное меню", callback_data="back")],
    ]
    await q.edit_message_text(f"Записано: {label}", reply_markup=InlineKeyboardMarkup(kb))

# ============================================================================
# HISTORY
# ============================================================================
async def show_history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()

    uid = q.from_user.id
    hist = trade_history.get(uid, [])

    if not hist:
        await q.edit_message_text("История пустая.")
        return

    text = "📜 История:\n\n" + "\n".join(hist[-50:])
    kb = [[InlineKeyboardButton("⬅ Главное меню", callback_data="back")]]
    await q.edit_message_text(text, reply_markup=InlineKeyboardMarkup(kb))

# ============================================================================
# ROUTER
# ============================================================================
async def router(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    data = q.data

    try:
        if data == "market":
            await choose_pair(update, context)
        elif data.startswith("page|"):
            page = int(data.split("|")[1])
            await choose_pair(update, context, page)
        elif data.startswith("pair|"):
            pair = data.split("|")[1]
            await choose_expiry(update, context, pair)
        elif data.startswith("analyze|"):
            _, pair, exp = data.split("|")
            await show_signal(update, context, pair, exp)
        elif data.startswith("res|"):
            label = "Плюс" if data.endswith("plus") else "Минус"
            await save_result(update, context, label)
        elif data == "history":
            await show_history(update, context)
        elif data == "back":
            await show_main_menu(update, context)
    except Exception as e:
        logger.exception("Callback error")
        await q.edit_message_text("Ошибка. Попробуйте снова.")

# ============================================================================
# APP START (CORRECT WAY FOR RENDER)
# ============================================================================
BOT_TOKEN = os.getenv("BOT_TOKEN")
WEBHOOK_URL = os.getenv("WEBHOOK_URL")

if not BOT_TOKEN or not WEBHOOK_URL:
    raise SystemExit("Please set BOT_TOKEN and WEBHOOK_URL")

app = ApplicationBuilder().token(BOT_TOKEN).build()

app.add_handler(CommandHandler("start", start))
app.add_handler(CallbackQueryHandler(router))

# правильный запуск webhook — без asyncio.run
app.run_webhook(
    listen="0.0.0.0",
    port=int(os.getenv("PORT", 10000)),
    url_path=BOT_TOKEN,
    webhook_url=f"{WEBHOOK_URL.rstrip('/')}/webhook/{BOT_TOKEN}"
)
