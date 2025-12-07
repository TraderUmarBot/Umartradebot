# main.py
import os
import logging
import asyncio
import math
import pandas as pd
import yfinance as yf
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- Globals ----------
user_state = {}
trade_history = {}

EXCHANGE_PAIRS = [
    "EUR/USD","GBP/USD","USD/JPY","AUD/USD","USD/CAD","USD/CHF",
    "EUR/JPY","GBP/JPY","AUD/JPY","EUR/GBP","EUR/AUD","GBP/AUD",
    "CAD/JPY","CHF/JPY","EUR/CAD","GBP/CAD","AUD/CAD","AUD/CHF","CAD/CHF"
]

EXCHANGE_ALLOWED = ["1m", "3m", "5m", "10m"]
INTERVAL_MAP = {"3m": "2m", "10m": "5m"}  # YFinance mapping

PAIRS_PER_PAGE = 6
LOOKBACK = 120

# ---------- Indicators ----------
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period, min_periods=1).mean()
    avg_loss = loss.rolling(period, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-9)
    return 100 - (100 / (1 + rs))

def SMA(series: pd.Series, period: int = 50) -> pd.Series:
    return series.rolling(period, min_periods=1).mean()

def EMA(series: pd.Series, period: int = 20) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def MACD(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def candle_patterns_df(df: pd.DataFrame):
    patterns = []
    if df.empty:
        return patterns
    o, c, h, l = df['Open'].iloc[-1], df['Close'].iloc[-1], df['High'].iloc[-1], df['Low'].iloc[-1]
    body = abs(c - o)
    candle_range = max(h - l, 1e-9)
    upper_shadow = h - max(c, o)
    lower_shadow = min(c, o) - l
    if body / candle_range < 0.25:
        patterns.append("Doji")
    if lower_shadow > 2 * body and body > 0:
        patterns.append("Hammer")
    if upper_shadow > 2 * body and body > 0:
        patterns.append("Inverted Hammer")
    patterns.append("Bullish Candle" if c > o else "Bearish Candle")
    return patterns

def escape_md(text: str) -> str:
    import re
    return re.sub(r'([_*\[\]()~`>#+\-=|{}.!])', r'\\\1', str(text))

# ---------- Helpers ----------
def get_pairs_page(pairs, page):
    start = page * PAIRS_PER_PAGE
    return pairs[start:start + PAIRS_PER_PAGE]

def total_pages(pairs):
    return (len(pairs) - 1) // PAIRS_PER_PAGE

def yfinance_interval_for(requested: str) -> str:
    return INTERVAL_MAP.get(requested, requested)

# ---------- Menu & Handlers ----------
async def show_main_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("📈 Биржевой рынок", callback_data="market|exchange")],
        [InlineKeyboardButton("📜 История сделок", callback_data="history|")]
    ]
    if update.message:
        await update.message.reply_text("👋 Привет! Выберите рынок:", reply_markup=InlineKeyboardMarkup(keyboard))
    else:
        q = update.callback_query
        await q.answer()
        await q.edit_message_text("👋 Привет! Выберите рынок:", reply_markup=InlineKeyboardMarkup(keyboard))

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await show_main_menu(update, context)

async def choose_pair(update: Update, context: ContextTypes.DEFAULT_TYPE, page=0):
    q = update.callback_query
    await q.answer()
    pairs = EXCHANGE_PAIRS
    page_pairs = get_pairs_page(pairs, page)
    keyboard = [[InlineKeyboardButton(p, callback_data=f"pair|{p}")] for p in page_pairs]
    nav = []
    if page > 0:
        nav.append(InlineKeyboardButton("⬅ Назад", callback_data=f"choose|{page-1}"))
    if page < total_pages(pairs):
        nav.append(InlineKeyboardButton("Вперёд ➡", callback_data=f"choose|{page+1}"))
    if nav:
        keyboard.append(nav)
    keyboard.append([InlineKeyboardButton("⬅ Главное меню", callback_data="back|")])
    await q.edit_message_text("⚡ Выберите валютную пару:", reply_markup=InlineKeyboardMarkup(keyboard))

async def choose_expiry(update: Update, context: ContextTypes.DEFAULT_TYPE, pair):
    q = update.callback_query
    await q.answer()
    keyboard = [[InlineKeyboardButton(tf, callback_data=f"analyze|{pair}|{tf}")] for tf in EXCHANGE_ALLOWED]
    keyboard.append([InlineKeyboardButton("⬅ Назад", callback_data="market|exchange")])
    keyboard.append([InlineKeyboardButton("⬅ Главное меню", callback_data="back|")])
    await q.edit_message_text(f"Пара: {pair}\nВыберите таймфрейм:", reply_markup=InlineKeyboardMarkup(keyboard))

# ---------- Analysis ----------
async def analyze_exchange(pair: str, expiry: str):
    ticker = pair.replace("/", "") + "=X"
    yfi_interval = yfinance_interval_for(expiry)
    notes = []
    try:
        df = yf.download(ticker, period="5d", interval=yfi_interval, progress=False, threads=False)
    except Exception as e:
        logger.exception("yfinance download error")
        return None, [f"Ошибка загрузки данных: {e}"]

    if df.empty or len(df) < 5:
        return None, [f"Нет данных для {ticker} (interval={yfi_interval})"]

    df = df.tail(LOOKBACK).copy()
    df["rsi"] = rsi(df["Close"])
    df["sma"] = SMA(df["Close"])
    df["ema"] = EMA(df["Close"])
    macd, macd_signal = MACD(df["Close"])
    df["macd"], df["macd_signal"] = macd, macd_signal
    notes += candle_patterns_df(df)

    last = df.iloc[-1]
    buy = sell = 0
    # RSI
    if last["rsi"] < 30: buy += 1; notes.append("RSI Oversold")
    elif last["rsi"] > 70: sell += 1; notes.append("RSI Overbought")
    # MACD
    if last["macd"] > last["macd_signal"]: buy += 1; notes.append("MACD Bull")
    else: sell += 1; notes.append("MACD Bear")
    # SMA/EMA trend
    if last["Close"] > last["sma"]: buy += 1; notes.append("Above SMA")
    else: sell += 1; notes.append("Below SMA")
    if last["Close"] > last["ema"]: buy += 1; notes.append("Above EMA")
    else: sell += 1; notes.append("Below EMA")

    if buy > sell: signal = "Вверх"
    elif sell > buy: signal = "Вниз"
    else: signal = "Нет сигнала"
    confidence = min(90, 50 + max(buy, sell)*10)
    return {"signal": signal, "conf": confidence, "notes": notes}, notes

async def show_signal(update: Update, context: ContextTypes.DEFAULT_TYPE, pair: str, expiry: str):
    q = update.callback_query
    await q.answer()
    uid = q.from_user.id
    msg = await q.edit_message_text("🔍 Анализирую рынок...")

    user_state[uid] = {"pair": pair, "expiry": expiry}

    result, notes = await analyze_exchange(pair, expiry)
    if not result:
        await msg.edit_text(f"❗ Не удалось собрать сигнал.\nПричины: {' | '.join(notes)}")
        return

    text = (f"📊 Сигнал: *{escape_md(result['signal'])}*\n"
            f"Пара: *{escape_md(pair)}*\n"
            f"Таймфрейм: {escape_md(expiry)}\n"
            f"Уверенность: {result['conf']}%\n"
            f"Notes: {' | '.join(result['notes'])}")
    keyboard = [
        [InlineKeyboardButton("🟢 Плюс", callback_data="result|plus"),
         InlineKeyboardButton("🔴 Минус", callback_data="result|minus")],
        [InlineKeyboardButton("⬅ Главное меню", callback_data="back|")]
    ]
    await msg.edit_text(text, parse_mode="MarkdownV2", reply_markup=InlineKeyboardMarkup(keyboard))

async def save_result(update: Update, context: ContextTypes.DEFAULT_TYPE, result_label: str):
    q = update.callback_query
    await q.answer()
    uid = q.from_user.id
    st = user_state.get(uid, {})
    pair = st.get("pair", "—")

    trade_history.setdefault(uid, []).append(f"{pair} — {result_label}")
    keyboard = [
        [InlineKeyboardButton("📈 Новый сигнал", callback_data="market|exchange")],
        [InlineKeyboardButton("📜 История", callback_data="history|")]
    ]
    await q.edit_message_text(f"✅ Записано: *{escape_md(result_label)}*", parse_mode="MarkdownV2", reply_markup=InlineKeyboardMarkup(keyboard))

async def show_history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    uid = q.from_user.id
    entries = trade_history.get(uid, [])
    if not entries:
        await q.edit_message_text("📭 История пустая.")
        return
    text = "📜 *История:*\n\n" + "\n".join([f"• {escape_md(t)}" for t in entries[-50:]])
    keyboard = [[InlineKeyboardButton("⬅ Главное меню", callback_data="back|")]]
    await q.edit_message_text(text, parse_mode="MarkdownV2", reply_markup=InlineKeyboardMarkup(keyboard))

# ---------- Callback router ----------
async def callbacks(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    data = q.data or ""
    parts = data.split("|")
    cmd = parts[0] if parts else ""
    try:
        if cmd == "market": await choose_pair(update, context)
        elif cmd == "choose": page = int(parts[1]); await choose_pair(update, context, page)
        elif cmd == "pair": pair = parts[1]; await choose_expiry(update, context, pair)
        elif cmd == "analyze": pair, expiry = parts[1], parts[2]; await show_signal(update, context, pair, expiry)
        elif cmd == "result": await save_result(update, context, "Плюс" if parts[1]=="plus" else "Минус")
        elif cmd == "history": await show_history(update, context)
        elif cmd == "back": await show_main_menu(update, context)
    except Exception:
        logger.exception("Ошибка в callback")
        try: await q.edit_message_text("Произошла ошибка, попробуйте ещё раз.")
        except: pass

# ---------- Bot setup ----------
BOT_TOKEN = os.getenv("BOT_TOKEN")
WEBHOOK_URL = os.getenv("WEBHOOK_URL")
if not BOT_TOKEN or not WEBHOOK_URL:
    raise SystemExit("Set BOT_TOKEN and WEBHOOK_URL env vars")

application = ApplicationBuilder().token(BOT_TOKEN).build()
application.add_handler(CommandHandler("start", start))
application.add_handler(CallbackQueryHandler(callbacks))

async def main():
    logger.info("Инициализация Telegram бота...")
    await application.initialize()
    await application.start()
    webhook_url = f"{WEBHOOK_URL.rstrip('/')}/webhook/{BOT_TOKEN}"
    await application.bot.set_webhook(webhook_url)
    logger.info(f"Webhook установлен: {webhook_url}")
    await application.updater.start_webhook(
        listen="0.0.0.0",
        port=int(os.getenv("PORT", 10000)),
        webhook_url_path=BOT_TOKEN
    )
    await application.updater.idle()

if __name__ == "__main__":
    asyncio.run(main())
