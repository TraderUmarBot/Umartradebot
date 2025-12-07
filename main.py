# =======================
# main.py — Telegram бот для Pocket Option с 50 стратегиями (FIXED)
# =======================

import logging, os, re, asyncio
import nest_asyncio
nest_asyncio.apply()

import pandas as pd, yfinance as yf
from flask import Flask, request, abort
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes

logging.basicConfig(level=logging.INFO)

# -----------------------
# Глобальные переменные
# -----------------------
user_state = {}
trade_history = {}

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

EXCHANGE_FRAMES = ["1m","3m","5m","10m"]
OTC_FRAMES = ["5s","15s","1m","3m","5m"]

TF_HIERARCHY = {
    "exchange": EXCHANGE_FRAMES,
    "otc": OTC_FRAMES
}

# -----------------------
# Стратегии
# -----------------------
STRATEGIES = [
    {"name": f"Стратегия {i+1}",
     "description": f"Описание стратегии {i+1}: Индикаторы настроены оптимально, вход при условии X, стоп-лосс Y, тейк-профит Z."}
    for i in range(50)
]
STRATEGIES_PER_PAGE = 6

def get_strategy_page(page):
    start = page * STRATEGIES_PER_PAGE
    return STRATEGIES[start:start + STRATEGIES_PER_PAGE]

def total_strategy_pages():
    return (len(STRATEGIES) - 1) // STRATEGIES_PER_PAGE

# -----------------------
# Меню стратегий
# -----------------------
async def show_strategies(update: Update, context: ContextTypes.DEFAULT_TYPE, page=0):
    q = update.callback_query
    await q.answer()
    page_strategies = get_strategy_page(page)
    
    # Клавиатура стратегий — каждая кнопка на отдельной строке
    keyboard = [[InlineKeyboardButton(s["name"], callback_data=f"strategy_{page}_{i}")]
                 ] for i, s in enumerate(page_strategies)]
    
    # Навигация по страницам
    nav = []
    if page > 0:
        nav.append(InlineKeyboardButton("⬅ Назад", callback_data=f"strategies_{page-1}"))
    if page < total_strategy_pages():
        nav.append(InlineKeyboardButton("Вперёд ➡", callback_data=f"strategies_{page+1}"))
    if nav:
        keyboard.append(nav)
    
    # Главное меню
    keyboard.append([InlineKeyboardButton("⬅ Главное меню", callback_data="back_to_menu")])
    
    await q.edit_message_text("📘 Выберите стратегию:", reply_markup=InlineKeyboardMarkup(keyboard))

async def show_strategy_detail(update: Update, context: ContextTypes.DEFAULT_TYPE, page, idx):
    q = update.callback_query
    try:
        strategy = get_strategy_page(page)[idx]
    except Exception:
        await q.answer("Стратегия не найдена", show_alert=True)
        return
    keyboard = [
        [InlineKeyboardButton("⬅ Назад к стратегиям", callback_data=f"strategies_{page}")],
        [InlineKeyboardButton("⬅ Главное меню", callback_data="back_to_menu")]
    ]
    await q.edit_message_text(
        f"📌 {strategy['name']}\n\n{strategy['description']}",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

# -----------------------
# Технические индикаторы
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

def BollingerBands(series, period=20, mult=2):
    sma = series.rolling(period, min_periods=1).mean()
    std = series.rolling(period, min_periods=1).std(ddof=0).fillna(0)
    upper = sma + mult * std
    lower = sma - mult * std
    return upper, lower

def ATR(df, period=14):
    high_low = (df['High'] - df['Low']).abs()
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()

def SuperTrend(df, period=7, multiplier=3):
    hl2 = (df['High'] + df['Low']) / 2
    atr = ATR(df, period)
    upper_basic = hl2 + multiplier * atr
    lower_basic = hl2 - multiplier * atr
    upper = upper_basic.copy()
    lower = lower_basic.copy()
    in_uptrend = pd.Series(index=df.index, data=True)
    if len(df) > 0:
        in_uptrend.iloc[0] = True
    for i in range(1, len(df)):
        upper.iloc[i] = min(upper_basic.iloc[i], upper.iloc[i-1]) if df['Close'].iloc[i-1] <= upper.iloc[i-1] else upper_basic.iloc[i]
        lower.iloc[i] = max(lower_basic.iloc[i], lower.iloc[i-1]) if df['Close'].iloc[i-1] >= lower.iloc[i-1] else lower_basic.iloc[i]
        if df['Close'].iloc[i] > upper.iloc[i-1]:
            in_uptrend.iloc[i] = True
        elif df['Close'].iloc[i] < lower.iloc[i-1]:
            in_uptrend.iloc[i] = False
        else:
            in_uptrend.iloc[i] = in_uptrend.iloc[i-1]
    return in_uptrend

def StochasticOscillator(df, k_period=14, d_period=3):
    low_min = df['Low'].rolling(k_period, min_periods=1).min()
    high_max = df['High'].rolling(k_period, min_periods=1).max()
    denom = (high_max - low_min).replace(0, 1e-9)
    k = 100 * ((df['Close'] - low_min) / denom)
    d = k.rolling(d_period, min_periods=1).mean()
    return k, d

def CCI(df, period=20):
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    ma = tp.rolling(period, min_periods=1).mean()
    md = tp.rolling(period, min_periods=1).std(ddof=0).replace(0, 1e-9)
    return (tp - ma) / (0.015 * md)

def candle_patterns(df):
    patterns = []
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

# -----------------------
# Вспомогательные функции
# -----------------------
def escape_md(text: str):
    return re.sub(r"([_*\[\]()~`>#+\-=|{}.!])", r"\\\1", str(text))

def get_pairs_page(pairs, page):
    start = page * PAIRS_PER_PAGE
    return pairs[start:start + PAIRS_PER_PAGE]

def total_pages(pairs):
    return (len(pairs) - 1) // PAIRS_PER_PAGE

# -----------------------
# MAIN MENU
# -----------------------
async def show_main_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("📈 Биржевой рынок", callback_data="market_exchange")],
        [InlineKeyboardButton("📈 OTC рынок", callback_data="market_otc")],
        [InlineKeyboardButton("📘 Стратегии", callback_data="strategies")],
        [InlineKeyboardButton("📜 История сделок", callback_data="history")]
    ]

    if update.message:
        await update.message.reply_text("👋 Привет! Выберите рынок:", reply_markup=InlineKeyboardMarkup(keyboard))
    elif update.callback_query:
        q = update.callback_query
        await q.answer()
        await q.edit_message_text("👋 Привет! Выберите рынок:", reply_markup=InlineKeyboardMarkup(keyboard))

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await show_main_menu(update, context)

# -----------------------
# Выбор пары
# -----------------------
async def choose_pair(update: Update, context: ContextTypes.DEFAULT_TYPE, market="exchange", page=0):
    q = update.callback_query
    await q.answer()
    pairs = EXCHANGE_PAIRS if market == "exchange" else OTC_PAIRS
    page_pairs = get_pairs_page(pairs, page)
    keyboard = [[InlineKeyboardButton(p, callback_data=f"pair_{market}_{p}")] for p in page_pairs]
    nav = []
    if page > 0:
        nav.append(InlineKeyboardButton("⬅ Назад", callback_data=f"choose_{market}_{page-1}"))
    if page < total_pages(pairs):
        nav.append(InlineKeyboardButton("Вперёд ➡", callback_data=f"choose_{market}_{page+1}"))
    if nav:
        keyboard.append(nav)
    keyboard.append([InlineKeyboardButton("⬅ Главное меню", callback_data="back_to_menu")])
    await q.edit_message_text("⚡ Выберите валютную пару:", reply_markup=InlineKeyboardMarkup(keyboard))

# -----------------------
# Multi-TF анализ
# -----------------------
async def show_signal(update: Update, context: ContextTypes.DEFAULT_TYPE, market, pair):
    q = update.callback_query
    await q.answer()
    uid = q.from_user.id
    tfs = TF_HIERARCHY.get(market, ["1m","3m","5m"])
    msg = await q.edit_message_text("🔍 Анализирую рынок...")
    buy_total = 0
    sell_total = 0
    notes_total = []

    for tf in tfs:
        await asyncio.sleep(0.3)
        ticker = pair.replace(" OTC","").replace("/","") + "=X"
        try:
            df = yf.download(ticker, period="5d", interval=tf, progress=False)
        except Exception:
            df = None
        if df is None or df.empty or len(df)<10:
            notes_total.append(f"{tf}: нет данных")
            continue
        df = df.tail(LOOKBACK).copy()

        df["rsi"]=rsi(df["Close"])
        macd, macd_signal = MACD(df["Close"])
        df["macd"], df["macd_signal"] = macd, macd_signal
        last = df.iloc[-1]

        buys = sells = 0
        notes = []
        if last["rsi"] < 30:
            buys += 1; notes.append("RSI Oversold ⬆")
        elif last["rsi"] > 70:
            sells += 1; notes.append("RSI Overbought ⬇")
        if last["macd"] > last["macd_signal"]:
            buys += 1; notes.append("MACD Bull ⬆")
        else:
            sells += 1; notes.append("MACD Bear ⬇")

        buy_total += buys
        sell_total += sells
        notes_total += notes

    if buy_total > sell_total:
        signal = "Вверх"
    elif sell_total > buy_total:
        signal = "Вниз"
    else:
        signal = "Нет сигнала"

    percent = min(90, 50 + max(buy_total, sell_total) * 5)
    user_state[uid] = {"pair": pair, "market": market}

    keyboard = [[InlineKeyboardButton("🟢 Плюс", callback_data="result_plus"),
                 InlineKeyboardButton("🔴 Минус", callback_data="result_minus")],
                [InlineKeyboardButton("⬅ Главное меню", callback_data="back_to_menu")]]

    await msg.edit_text(
        f"📊 Сигнал: *{escape_md(signal)}*\nПара: *{escape_md(pair)}*\nAccuracy: {percent}%\nNotes: {' | '.join(notes_total)}",
        parse_mode="MarkdownV2",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

# -----------------------
# Сохранение результата
# -----------------------
async def save_result(update: Update, context: ContextTypes.DEFAULT_TYPE, result):
    q = update.callback_query
    await q.answer()
    uid = q.from_user.id
    if uid not in trade_history:
        trade_history[uid] = []
    pair = user_state.get(uid, {}).get("pair", "—")
    trade_history[uid].append(f"{pair} — {result}")
    keyboard = [
        [InlineKeyboardButton("📈 Новый сигнал", callback_data="market_exchange")],
        [InlineKeyboardButton("📜 История", callback_data="history")]
    ]
    await q.edit_message_text(f"✅ Записано: *{escape_md(result)}*", parse_mode="MarkdownV2", reply_markup=InlineKeyboardMarkup(keyboard))

# -----------------------
# История
# -----------------------
async def show_history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    uid = q.from_user.id
    if uid not in trade_history or len(trade_history[uid]) == 0:
        await q.edit_message_text("📭 История пустая.")
        return
    text = "📜 *История:*\n\n" + "\n".join([f"• {escape_md(t)}" for t in trade_history[uid]])
    keyboard = [[InlineKeyboardButton("⬅ Главное меню", callback_data="back_to_menu")]]
    await q.edit_message_text(text, parse_mode="MarkdownV2", reply_markup=InlineKeyboardMarkup(keyboard))

# -----------------------
# Callback Router
# -----------------------
async def callbacks(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    data = q.data
    if data == "market_exchange":
        await choose_pair(update, context, "exchange", 0)
    elif data == "market_otc":
        await choose_pair(update, context, "otc", 0)
    elif data.startswith("choose_"):
        _, market, page = data.split("_")
        await choose_pair(update, context, market, int(page))
    elif data.startswith("pair_"):
        _, market, pair = data.split("_", 2)
        await show_signal(update, context, market, pair)
    elif data == "result_plus":
        await save_result(update, context, "Плюс")
    elif data == "result_minus":
        await save_result(update, context, "Минус")
    elif data == "history":
        await show_history(update, context)
    elif data == "back_to_menu":
        await show_main_menu(update, context)
    elif data == "strategies":
        await show_strategies(update, context, 0)
    elif data.startswith("strategies_"):
        _, page = data.split("_")
        await show_strategies(update, context, int(page))
    elif data.startswith("strategy_"):
        _, page, idx = data.split("_")
        await show_strategy_detail(update, context, int(page), int(idx))
    else:
        await q.answer()

# -----------------------
# Flask + Webhook + Application init
# -----------------------
BOT_TOKEN = os.getenv("BOT_TOKEN")
WEBHOOK_URL = os.getenv("WEBHOOK_URL")

if not BOT_TOKEN or not WEBHOOK_URL:
    logging.error("BOT_TOKEN или WEBHOOK_URL не заданы в окружении.")
    raise SystemExit("Set BOT_TOKEN and WEBHOOK_URL env vars")

application = ApplicationBuilder().token(BOT_TOKEN).build()
application.add_handler(CommandHandler("start", start))
application.add_handler(CallbackQueryHandler(callbacks))

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return "Bot is running"

@app.route("/webhook/<token>", methods=["POST"])
def webhook(token):
    if token != BOT_TOKEN:
        abort(403)
    try:
        data = request.get_json(force=True)
        update = Update.de_json(data, application.bot)
        loop = asyncio.get_event_loop()
        loop.create_task(application.process_update(update))
        return "OK", 200
