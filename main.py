# main.py — полностью рабочий бот
import logging
import os
import re
import asyncio
import pandas as pd
import yfinance as yf

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
    "AUD/CAD OTC","CAD/CHF OTC","CHF/JPY OTC","EUR/GBP OTC","EUR/JPY OTC","GBP/USD OTC",
    "NZD/JPY OTC","NZD/USD OTC","USD/CAD OTC","EUR/RUB OTC","USD/PKR OTC","USD/COP OTC",
    "AUD/USD OTC","EUR/CHF OTC","GBP/JPY OTC","GBP/AUD OTC","USD/JPY OTC","USD/CHF OTC",
    "AUD/JPY OTC","NZD/CAD OTC"
]

PAIRS_PER_PAGE = 6
LOOKBACK = 120

# Таймфреймы
EXCHANGE_TIMEFRAMES = ["1m", "3m", "5m", "10m"]
OTC_TIMEFRAMES = ["5s", "30s", "1m", "3m", "5m"]

# -----------------------
# Индикаторы (как в предыдущем коде)
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
    return macd, signal_line, macd - signal_line

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
    if len(df) > 0: in_uptrend.iloc[0] = True
    for i in range(1, len(df)):
        upper.iloc[i] = min(upper_basic.iloc[i], upper.iloc[i-1]) if df['Close'].iloc[i-1] <= upper.iloc[i-1] else upper_basic.iloc[i]
        lower.iloc[i] = max(lower_basic.iloc[i], lower.iloc[i-1]) if df['Close'].iloc[i-1] >= lower.iloc[i-1] else lower_basic.iloc[i]
        if df['Close'].iloc[i] > upper.iloc[i-1]: in_uptrend.iloc[i] = True
        elif df['Close'].iloc[i] < lower.iloc[i-1]: in_uptrend.iloc[i] = False
        else: in_uptrend.iloc[i] = in_uptrend.iloc[i-1]
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
    if body / candle_range < 0.25: patterns.append("Doji")
    if lower_shadow > 2 * body and body > 0: patterns.append("Hammer")
    if upper_shadow > 2 * body and body > 0: patterns.append("Inverted Hammer")
    patterns.append("Bullish Candle" if c > o else "Bearish Candle")
    return patterns

# -----------------------
# Вспомогательные
# -----------------------
def escape_md(text: str):
    return re.sub(r"([_*\[\]()~`>#+\-=|{}.!])", r"\\\1", str(text))

def get_pairs_page(pairs_list, page):
    start = page * PAIRS_PER_PAGE
    return pairs_list[start:start + PAIRS_PER_PAGE]

def total_pages(pairs_list):
    return (len(pairs_list) - 1) // PAIRS_PER_PAGE

# -----------------------
# Сигналы
# -----------------------
def generate_signal(pair, timeframe):
    try:
        ticker = pair.replace("/", "") + "=X"
        df = yf.download(ticker, period="3d", interval=timeframe, progress=False)
        if df.empty or len(df) < 10:
            return "❌ Нет данных"
        df = df.tail(LOOKBACK).copy()

        df["rsi"] = rsi(df["Close"])
        df["sma50"] = SMA(df["Close"], 50)
        df["sma200"] = SMA(df["Close"], 200)
        df["ema20"] = EMA(df["Close"], 20)
        macd, macd_signal, _ = MACD(df["Close"])
        df["macd"], df["macd_signal"] = macd, macd_signal
        df["bb_upper"], df["bb_lower"] = BollingerBands(df["Close"])
        df["atr"] = ATR(df)
        df["supertrend"] = SuperTrend(df)
        k, d = StochasticOscillator(df)
        df["k"], df["d"] = k, d
        df["cci"] = CCI(df)

        last = df.iloc[-1]
        score_up = 0
        score_down = 0
        notes = []

        if pd.notna(last["rsi"]):
            if last["rsi"] < 30: score_up += 1; notes.append("RSI перепродан ⬆")
            elif last["rsi"] > 70: score_down += 1; notes.append("RSI перекуплен ⬇")
        if pd.notna(last["sma50"]) and pd.notna(last["sma200"]) and pd.notna(last["Close"]):
            if last["Close"] > last["sma50"] > last["sma200"]: score_up += 1; notes.append("Uptrend ⬆")
            elif last["Close"] < last["sma50"] < last["sma200"]: score_down += 1; notes.append("Downtrend ⬇")
        if pd.notna(last["macd"]) and pd.notna(last["macd_signal"]):
            if last["macd"] > last["macd_signal"]: score_up += 1; notes.append("MACD Bull ⬆")
            elif last["macd"] < last["macd_signal"]: score_down += 1; notes.append("MACD Bear ⬇")
        if pd.notna(last["bb_upper"]) and pd.notna(last["bb_lower"]) and pd.notna(last["Close"]):
            if last["Close"] < last["bb_lower"]: score_up += 1; notes.append("Price ниже BB ⬆")
            elif last["Close"] > last["bb_upper"]: score_down += 1; notes.append("Price выше BB ⬇")
        if bool(last["supertrend"]): score_up += 1; notes.append("SuperTrend Bull ⬆")
        else: score_down += 1; notes.append("SuperTrend Bear ⬇")
        if pd.notna(last["k"]):
            if last["k"] < 20: score_up += 1; notes.append("Stoch Oversold ⬆")
            elif last["k"] > 80: score_down += 1; notes.append("Stoch Overbought ⬇")
        if pd.notna(last["cci"]):
            if last["cci"] < -100: score_up += 1; notes.append("CCI Oversold ⬆")
            elif last["cci"] > 100: score_down += 1; notes.append("CCI Overbought ⬇")
        for p in candle_patterns(df):
            if p in ("Hammer", "Bullish Candle"): score_up += 1; notes.append(p + " ⬆")
            elif p in ("Inverted Hammer", "Bearish Candle"): score_down += 1; notes.append(p + " ⬇")
            elif p == "Doji": notes.append("Doji ⚖️")

        if score_up > score_down: direction = "Вверх ⬆"; strength_score = score_up
        elif score_down > score_up: direction = "Вниз ⬇"; strength_score = score_down
        else: direction = "Нет сигнала"; strength_score = 0

        max_score = 10
        percent = min(int((strength_score/max_score)*100), 100)
        strength_text = "Высокая" if percent >= 75 else "Средняя" if percent >= 50 else "Низкая"
        details = " | ".join(notes) if notes else "Нет деталей"

        return f"{direction} | {percent}% | Сила: {strength_text} | {details}"
    except Exception as e:
        logging.exception(f"Ошибка генерации сигнала для {pair}: {e}")
        return "❌ Ошибка генерации сигнала"

# -----------------------
# Следующий шаг — Telegram меню, кнопки выбора рынка, пар, таймфрейма и кнопки Плюс/Минус + История
# -----------------------

# -----------------------
# Telegram handlers
# -----------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("📊 Биржевой рынок", callback_data="market_exchange")],
        [InlineKeyboardButton("💹 OTC рынок", callback_data="market_otc")],
        [InlineKeyboardButton("📜 История сделок", callback_data="history")]
    ]
    await update.message.reply_text("👋 Привет! Выберите рынок:", reply_markup=InlineKeyboardMarkup(keyboard))

async def choose_pair(update, context, market, page=0):
    q = update.callback_query
    await q.answer()
    pairs_list = EXCHANGE_PAIRS if market == "exchange" else OTC_PAIRS
    pairs = get_pairs_page(pairs_list, page)
    keyboard = [[InlineKeyboardButton(p, callback_data=f"pair_{market}_{p}")] for p in pairs]
    nav = []
    if page > 0: nav.append(InlineKeyboardButton("⬅ Назад", callback_data=f"choose_pair_{market}_{page-1}"))
    if page < total_pages(pairs_list): nav.append(InlineKeyboardButton("Вперёд ➡", callback_data=f"choose_pair_{market}_{page+1}"))
    if nav: keyboard.append(nav)
    keyboard.append([InlineKeyboardButton("⬅ Главное меню", callback_data="back_to_menu")])
    await q.edit_message_text("⚡ Выберите валютную пару:", reply_markup=InlineKeyboardMarkup(keyboard))

async def choose_timeframe(update, context, market, pair):
    q = update.callback_query
    await q.answer()
    timeframes = EXCHANGE_TIMEFRAMES if market == "exchange" else OTC_TIMEFRAMES
    keyboard = [[InlineKeyboardButton(tf, callback_data=f"exp_{market}_{tf}_{pair}")] for tf in timeframes]
    keyboard.append([InlineKeyboardButton("⬅ Назад", callback_data=f"choose_pair_{market}_0")])
    await q.edit_message_text(f"Пара: *{escape_md(pair)}*\nВыберите таймфрейм:", parse_mode="MarkdownV2",
                              reply_markup=InlineKeyboardMarkup(keyboard))

async def show_signal(update, context, market, pair, tf):
    q = update.callback_query
    uid = q.from_user.id
    signal = generate_signal(pair, tf)
    user_state[uid] = {"market": market, "pair": pair, "tf": tf}
    keyboard = [
        [InlineKeyboardButton("🟢 Плюс", callback_data="result_plus"),
         InlineKeyboardButton("🔴 Минус", callback_data="result_minus")],
        [InlineKeyboardButton("⬅ Главное меню", callback_data="back_to_menu")]
    ]
    await q.edit_message_text(f"📊 Сигнал: *{escape_md(signal)}*\nПара: *{escape_md(pair)}*\nТаймфрейм: *{tf}*",
                              parse_mode="MarkdownV2", reply_markup=InlineKeyboardMarkup(keyboard))

async def save_result(update, context, result):
    q = update.callback_query
    uid = q.from_user.id
    if uid not in trade_history: trade_history[uid] = []
    state = user_state.get(uid, {})
    pair = state.get("pair", "—")
    tf = state.get("tf", "—")
    trade_history[uid].append(f"{pair} | {tf} — {result}")
    keyboard = [
        [InlineKeyboardButton("📈 Новый сигнал", callback_data=f"choose_pair_{state.get('market','exchange')}_0")],
        [InlineKeyboardButton("📜 История", callback_data="history")]
    ]
    await q.edit_message_text(f"✅ Записано: *{escape_md(result)}*", parse_mode="MarkdownV2", reply_markup=InlineKeyboardMarkup(keyboard))

async def history(update, context):
    q = update.callback_query
    uid = q.from_user.id
    if uid not in trade_history or not trade_history[uid]:
        await q.edit_message_text("📭 История пустая.")
        return
    text = "📜 *История сделок:*\n\n" + "\n".join([f"• {escape_md(t)}" for t in trade_history[uid]])
    keyboard = [[InlineKeyboardButton("⬅ Главное меню", callback_data="back_to_menu")]]
    await q.edit_message_text(text, parse_mode="MarkdownV2", reply_markup=InlineKeyboardMarkup(keyboard))

async def callbacks(update: Update, context: ContextTypes.DEFAULT_TYPE):
    data = update.callback_query.data
    if data.startswith("market_"):
        market = data.split("_")[1]
        await choose_pair(update, context, market, 0)
    elif data.startswith("choose_pair_"):
        parts = data.split("_")
        market = parts[2]
        page = int(parts[3])
        await choose_pair(update, context, market, page)
    elif data.startswith("pair_"):
        _, market, pair = data.split("_", 2)
        await choose_timeframe(update, context, market, pair)
    elif data.startswith("exp_"):
        _, market, tf, pair = data.split("_", 3)
        await show_signal(update, context, market, pair, tf)
    elif data == "result_plus": await save_result(update, context, "Плюс")
    elif data == "result_minus": await save_result(update, context, "Минус")
    elif data == "history": await history(update, context)
    elif data == "back_to_menu": await start(update, context)

# -----------------------
# Flask + webhook
# -----------------------
BOT_TOKEN = os.getenv("BOT_TOKEN")
WEBHOOK_URL = os.getenv("WEBHOOK_URL")

if not BOT_TOKEN:
    logging.error("BOT_TOKEN не указан.")

application = ApplicationBuilder().token(BOT_TOKEN).build()
application.add_handler(CommandHandler("start", start))
application.add_handler(CallbackQueryHandler(callbacks))

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return "Bot is running"

@app.route("/webhook/<token>", methods=["POST"])
def webhook(token):
    if not BOT_TOKEN or token != BOT_TOKEN:
        logging.warning("Webhook token mismatch or BOT_TOKEN not set")
        abort(403)
    try:
        data = request.get_json(force=True)
        update = Update.de_json(data, application.bot)
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            loop.run_until_complete(application.process_update(update))
        finally:
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            except Exception: pass
            loop.close()
            asyncio.set_event_loop(None)
        return "OK", 200
    except Exception:
        logging.exception("Ошибка в webhook:")
        return "ERROR", 500

if __name__ == "__main__":
    if BOT_TOKEN and WEBHOOK_URL:
        try:
            url = f"{WEBHOOK_URL.rstrip('/')}/webhook/{BOT_TOKEN}"
            logging.info(f"Setting webhook to: {url}")
            asyncio.run(application.bot.set_webhook(url))
            logging.info("Webhook установлен")
        except Exception:
            logging.exception("Не удалось установить webhook.")
    port = int(os.getenv("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
