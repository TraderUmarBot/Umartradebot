# main.py
# Telegram bot — Pocket Option helper
# Exchange: yfinance + indicators
# OTC: indicators on deterministic user-driven price series (no random)
import os
import logging
import asyncio
import pandas as pd
import yfinance as yf
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes

logging.basicConfig(level=logging.INFO)

# -----------------------
# Globals
# -----------------------
user_state = {}          # per-user ephemeral state
trade_history = {}       # per-user trade results
otc_price_history = {}   # { pair: [close1, close2, ...] } deterministic history for OTC
BASE_OTC_PRICE = 1.0     # default starting price for OTC pairs if none exists
OTC_STEP = 0.1           # chosen variant B: step +/- 0.1

EXCHANGE_PAIRS = [
    "EUR/USD","GBP/USD","USD/JPY","AUD/USD","USD/CAD","USD/CHF",
    "EUR/JPY","GBP/JPY","AUD/JPY","EUR/GBP","EUR/AUD","GBP/AUD",
    "CAD/JPY","CHF/JPY","EUR/CAD","GBP/CAD","AUD/CAD","AUD/CHF","CAD/CHF"
]

OTC_PAIRS = [
    "AUD/CAD OTC","CAD/CHF OTC","CHF/JPY OTC","EUR/GBP OTC","EUR/JPY OTC",
    "GBP/USD OTC","NZD/JPY OTC","NZD/USD OTC","USD/CAD OTC","EUR/RUB OTC",
    "USD/PKR OTC","USD/COP OTC","AUD/USD OTC","EUR/CHF OTC","GBP/JPY OTC",
    "GBP/AUD OTC","USD/JPY OTC","USD/CHF OTC","AUD/JPY OTC","NZD/CAD OTC"
]

PAIRS_PER_PAGE = 6
LOOKBACK = 120

EXCHANGE_FRAMES = ["1m","3m","5m","10m"]   # used for yfinance intervals
OTC_FRAMES = ["1m","3m","5m"]             # logical frames (not used with yfinance)

TF_HIERARCHY = {"exchange": EXCHANGE_FRAMES, "otc": OTC_FRAMES}

# -----------------------
# Indicators
# -----------------------
def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period, min_periods=1).mean()
    avg_loss = loss.rolling(period, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-9)
    return 100 - (100 / (1 + rs))

def SMA(series, period=50):
    return series.rolling(period, min_periods=1).mean()

def EMA(series, period=20):
    return series.ewm(span=period, adjust=False).mean()

def MACD(series, fast=12, slow=26, signal=9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def candle_patterns_df(df):
    patterns = []
    if len(df) == 0:
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

def escape_md(text: str):
    import re
    return re.sub(r'([_*\[\]()~`>#+\-=|{}.!])', r'\\\1', str(text))

# -----------------------
# Helpers
# -----------------------
def get_pairs_page(pairs, page):
    start = page * PAIRS_PER_PAGE
    return pairs[start:start + PAIRS_PER_PAGE]

def total_pages(pairs):
    return (len(pairs) - 1) // PAIRS_PER_PAGE

def ensure_otc_history(pair):
    # initialize deterministic OTC price history if missing
    if pair not in otc_price_history or not otc_price_history[pair]:
        otc_price_history[pair] = [BASE_OTC_PRICE]

def build_otc_dataframe_from_history(pair, lookback=LOOKBACK):
    ensure_otc_history(pair)
    closes = otc_price_history[pair][-lookback:]
    # create OHLC synthetic from closes deterministically (no randomness)
    opens = [closes[i-1] if i>0 else closes[0] for i in range(len(closes))]
    highs = [max(o,c) for o,c in zip(opens, closes)]
    lows = [min(o,c) for o,c in zip(opens, closes)]
    df = pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": closes})
    return df

# -----------------------
# Menu & Handlers
# -----------------------
async def show_main_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("📈 Биржевой рынок", callback_data="market|exchange")],
        [InlineKeyboardButton("📈 OTC рынок", callback_data="market|otc")],
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

async def choose_pair(update: Update, context: ContextTypes.DEFAULT_TYPE, market="exchange", page=0):
    q = update.callback_query
    await q.answer()
    pairs = EXCHANGE_PAIRS if market == "exchange" else OTC_PAIRS
    page_pairs = get_pairs_page(pairs, page)
    keyboard = [[InlineKeyboardButton(p, callback_data=f"pair|{market}|{p}")] for p in page_pairs]
    nav = []
    if page > 0:
        nav.append(InlineKeyboardButton("⬅ Назад", callback_data=f"choose|{market}|{page-1}"))
    if page < total_pages(pairs):
        nav.append(InlineKeyboardButton("Вперёд ➡", callback_data=f"choose|{market}|{page+1}"))
    if nav:
        keyboard.append(nav)
    keyboard.append([InlineKeyboardButton("⬅ Главное меню", callback_data="back|")])
    await q.edit_message_text("⚡ Выберите валютную пару:", reply_markup=InlineKeyboardMarkup(keyboard))

# after pair chosen -> choose expiry/timeframe
async def choose_expiry(update: Update, context: ContextTypes.DEFAULT_TYPE, market, pair):
    q = update.callback_query
    await q.answer()
    # simple expiry options (can be customized)
    keyboard = [
        [InlineKeyboardButton("1 мин", callback_data=f"analyze|{market}|{pair}|1m")],
        [InlineKeyboardButton("5 мин", callback_data=f"analyze|{market}|{pair}|5m")],
        [InlineKeyboardButton("15 мин", callback_data=f"analyze|{market}|{pair}|15m")],
        [InlineKeyboardButton("⬅ Назад", callback_data=f"choose|{market}|0")],
        [InlineKeyboardButton("⬅ Главное меню", callback_data="back|")]
    ]
    await q.edit_message_text(f"Пара: {pair}\nВыберите время экспирации:", reply_markup=InlineKeyboardMarkup(keyboard))

# core analysis
async def analyze_exchange(pair, expiry, q):
    # pair like "EUR/USD" -> ticker for yfinance: "EURUSD=X" (common)
    ticker = pair.replace("/","") + "=X"
    notes = []
    try:
        df = yf.download(ticker, period="5d", interval=expiry, progress=False)
    except Exception as e:
        logging.exception("yfinance error:")
        df = None
    if df is None or df.empty or len(df) < 10:
        notes.append(f"Нет данных yfinance для {ticker} с интервалом {expiry}")
        return None, notes

    df = df.tail(LOOKBACK).copy()
    df["rsi"] = rsi(df["Close"])
    macd, macd_signal = MACD(df["Close"])
    df["macd"], df["macd_signal"] = macd, macd_signal
    notes += candle_patterns_df(df)
    last = df.iloc[-1]

    buy = sell = 0
    if last["rsi"] < 30:
        buy += 1; notes.append("RSI Oversold")
    elif last["rsi"] > 70:
        sell += 1; notes.append("RSI Overbought")
    if last["macd"] > last["macd_signal"]:
        buy += 1; notes.append("MACD Bull")
    else:
        sell += 1; notes.append("MACD Bear")

    if buy > sell:
        signal = "Вверх"
    elif sell > buy:
        signal = "Вниз"
    else:
        signal = "Нет сигнала"

    confidence = min(90, 50 + max(buy, sell) * 10)
    return {"signal": signal, "conf": confidence, "notes": notes}, notes

async def analyze_otc(pair, expiry, q, user_id):
    # Build df from otc_price_history (deterministic)
    df = build_otc_dataframe_from_history(pair, lookback=LOOKBACK)
    if df is None or df.empty or len(df) < 3:
        return None, ["Недостаточно истории OTC для анализа"]
    df["rsi"] = rsi(df["Close"])
    macd, macd_signal = MACD(df["Close"])
    df["macd"], df["macd_signal"] = macd, macd_signal
    notes = candle_patterns_df(df)
    last = df.iloc[-1]

    buy = sell = 0
    if last["rsi"] < 30:
        buy += 1; notes.append("RSI Oversold")
    elif last["rsi"] > 70:
        sell += 1; notes.append("RSI Overbought")
    if last["macd"] > last["macd_signal"]:
        buy += 1; notes.append("MACD Bull")
    else:
        sell += 1; notes.append("MACD Bear")

    if buy > sell:
        signal = "Вверх"
    elif sell > buy:
        signal = "Вниз"
    else:
        signal = "Нет сигнала"

    confidence = min(90, 50 + max(buy, sell) * 10)
    return {"signal": signal, "conf": confidence, "notes": notes}, notes

async def show_signal(update: Update, context: ContextTypes.DEFAULT_TYPE, market, pair, expiry):
    q = update.callback_query
    await q.answer()
    uid = q.from_user.id
    msg = await q.edit_message_text("🔍 Анализирую рынок...")

    # store current selection for user
    user_state[uid] = {"pair": pair, "market": market, "expiry": expiry}

    try:
        if market == "exchange":
            result, notes = await analyze_exchange(pair, expiry, q)
        else:
            result, notes = await analyze_otc(pair, expiry, q, uid)
    except Exception:
        logging.exception("Ошибка в анализе:")
        result = None
        notes = ["Ошибка анализа"]

    if result is None:
        await msg.edit_text(f"❗ Не удалось собрать сигнал для {pair}.\nПричины: {' | '.join(notes)}")
        return

    keyboard = [
        [InlineKeyboardButton("🟢 Плюс", callback_data="result|plus"),
         InlineKeyboardButton("🔴 Минус", callback_data="result|minus")],
        [InlineKeyboardButton("⬅ Главное меню", callback_data="back|")]
    ]

    text = (f"📊 Сигнал: *{escape_md(result['signal'])}*\n"
            f"Пара: *{escape_md(pair)}*\n"
            f"Expiry: {escape_md(expiry)}\n"
            f"Confidence: {result['conf']}%\n"
            f"Notes: {' | '.join(result['notes'])}")
    await msg.edit_text(text, parse_mode="MarkdownV2", reply_markup=InlineKeyboardMarkup(keyboard))

# Save result and update OTC history if needed
async def save_result(update: Update, context: ContextTypes.DEFAULT_TYPE, result_label):
    q = update.callback_query
    await q.answer()
    uid = q.from_user.id
    st = user_state.get(uid, {})
    pair = st.get("pair", "—")
    market = st.get("market", "exchange")

    # record history
    trade_history.setdefault(uid, []).append(f"{pair} — {result_label}")

    # If OTC, update deterministic price history (no randomness)
    if market == "otc":
        ensure_otc_history(pair)
        last_price = otc_price_history[pair][-1]
        if result_label == "Плюс":
            new_price = round(last_price + OTC_STEP, 6)
        else:
            new_price = round(last_price - OTC_STEP, 6)
        otc_price_history[pair].append(new_price)

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

# -----------------------
# Central callback router
# -----------------------
async def callbacks(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()  # quick answer to avoid double-tap UI
    data = q.data or ""
    # split on '|' safely
    parts = data.split("|")
    cmd = parts[0] if parts else ""

    try:
        if cmd == "market":
            market = parts[1]
            await choose_pair(update, context, market, 0)
        elif cmd == "choose":
            market = parts[1]; page = int(parts[2])
            await choose_pair(update, context, market, page)
        elif cmd == "pair":
            market = parts[1]; pair = parts[2]
            await choose_expiry(update, context, market, pair)
        elif cmd == "analyze":
            # analyze|market|pair|expiry
            market = parts[1]; pair = parts[2]; expiry = parts[3]
            await show_signal(update, context, market, pair, expiry)
        elif cmd == "result":
            # result|plus or result|minus
            kind = parts[1]
            if kind == "plus":
                await save_result(update, context, "Плюс")
            else:
                await save_result(update, context, "Минус")
        elif cmd == "history":
            await show_history(update, context)
        elif cmd == "back":
            await show_main_menu(update, context)
        else:
            await q.answer()
    except Exception:
        logging.exception("Ошибка в обработке callback:")
        await q.edit_message_text("Произошла ошибка при обработке. Попробуйте ещё раз.")

# -----------------------
# Bot setup & run
# -----------------------
BOT_TOKEN = os.getenv("BOT_TOKEN")
WEBHOOK_URL = os.getenv("WEBHOOK_URL")

if not BOT_TOKEN or not WEBHOOK_URL:
    logging.error("BOT_TOKEN или WEBHOOK_URL не заданы в окружении.")
    raise SystemExit("Set BOT_TOKEN and WEBHOOK_URL env vars")

application = ApplicationBuilder().token(BOT_TOKEN).build()
application.add_handler(CommandHandler("start", start))
application.add_handler(CallbackQueryHandler(callbacks))

async def main():
    logging.info("Инициализация Telegram бота...")
    await application.initialize()
    await application.start()
    webhook_url = f"{WEBHOOK_URL.rstrip('/')}/webhook/{BOT_TOKEN}"
    await application.bot.set_webhook(webhook_url)
    logging.info(f"Webhook установлен: {webhook_url}")

    await application.run_webhook(
        listen="0.0.0.0",
        port=int(os.getenv("PORT", 10000)),
        url_path=BOT_TOKEN,
        webhook_url=webhook_url
    )

if __name__ == "__main__":
    asyncio.run(main())
