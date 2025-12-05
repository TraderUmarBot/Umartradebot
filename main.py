# main.py — рабочая версия для Render (исправлённая)
import logging
import pandas as pd
import yfinance as yf
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes
from flask import Flask, request
import os
import re
import asyncio

logging.basicConfig(level=logging.INFO)

# =====================================================
# Глобальные переменные
# =====================================================
user_state = {}
trade_history = {}

ALL_PAIRS = [
    "EUR/USD","GBP/USD","USD/JPY","AUD/USD","USD/CAD","USD/CHF",
    "EUR/JPY","GBP/JPY","AUD/JPY","EUR/GBP","EUR/AUD","GBP/AUD",
    "CAD/JPY","CHF/JPY","EUR/CAD","GBP/CAD","AUD/CAD","AUD/CHF","CAD/CHF"
]
PAIRS_PER_PAGE = 6
LOOKBACK = 120

# =====================================================
# Индикаторы (оставил твою реализацию, слегка защитил от NaN)
# =====================================================
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
    hist = macd - signal_line
    return macd, signal_line, hist

def BollingerBands(series, period=20, mult=2):
    sma = series.rolling(period, min_periods=1).mean()
    std = series.rolling(period, min_periods=1).std(ddof=0).fillna(0)
    upper = sma + mult*std
    lower = sma - mult*std
    return upper, lower

def ATR(df, period=14):
    high_low = df['High'] - df['Low']
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

    for i in range(1, len(df)):
        # поддержка previous bounds — простая логика
        if df['Close'].iloc[i-1] <= upper.iloc[i-1]:
            upper.iloc[i] = min(upper_basic.iloc[i], upper.iloc[i-1])
        else:
            upper.iloc[i] = upper_basic.iloc[i]

        if df['Close'].iloc[i-1] >= lower.iloc[i-1]:
            lower.iloc[i] = max(lower_basic.iloc[i], lower.iloc[i-1])
        else:
            lower.iloc[i] = lower_basic.iloc[i]

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

# =====================================================
# Вспомогательные
# =====================================================
def escape_md(text: str):
    return re.sub(r"([_*\[\]()~`>#+\-=|{}.!])", r"\\\1", str(text))

def get_pairs_page(page):
    start = page * PAIRS_PER_PAGE
    end = start + PAIRS_PER_PAGE
    return ALL_PAIRS[start:end]

def total_pages():
    return (len(ALL_PAIRS) - 1) // PAIRS_PER_PAGE

# =====================================================
# Генерация сигнала
# =====================================================
def generate_signal(pair, timeframe):
    try:
        # сначала пробуем с =X
        ticker = pair.replace("/", "")
        df = yf.download(ticker, period="3d", interval="1m", progress=False)
        if df.empty:
            df = yf.download(ticker + "=X", period="3d", interval="1m", progress=False)
            if df.empty:
                logging.warning("yfinance returned no data for %s (tried %s and %s)",
                                pair, ticker, ticker+"=X")
                return None

        df = df.tail(LOOKBACK).copy()
        if len(df) < 10:
            return None

        df["rsi"] = rsi(df["Close"])
        df["sma50"] = SMA(df["Close"], 50)
        df["sma200"] = SMA(df["Close"], 200)
        df["ema20"] = EMA(df["Close"], 20)
        macd, macd_signal, macd_hist = MACD(df["Close"])
        df["macd"] = macd
        df["macd_signal"] = macd_signal
        df["bb_upper"], df["bb_lower"] = BollingerBands(df["Close"])
        df["bb_width"] = df["bb_upper"] - df["bb_lower"]
        df["atr"] = ATR(df)
        df["supertrend"] = SuperTrend(df)
        k, d = StochasticOscillator(df)
        df["k"] = k
        df["d"] = d
        df["cci"] = CCI(df)

        last = df.iloc[-1]
        buy_signals = 0
        sell_signals = 0
        notes = []

        if pd.notna(last.get("rsi")):
            if last["rsi"] < 30:
                buy_signals += 1; notes.append("RSI Oversold ⬆")
            elif last["rsi"] > 70:
                sell_signals += 1; notes.append("RSI Overbought ⬇")

        if pd.notna(last.get("sma50")) and pd.notna(last.get("sma200")):
            if last["Close"] > last["sma50"] > last["sma200"]:
                buy_signals += 1; notes.append("Uptrend (SMA50>SMA200) ⬆")
            elif last["Close"] < last["sma50"] < last["sma200"]:
                sell_signals += 1; notes.append("Downtrend (SMA50<SMA200) ⬇")

        if pd.notna(last.get("macd")) and pd.notna(last.get("macd_signal")):
            if last["macd"] > last["macd_signal"]:
                buy_signals += 1; notes.append("MACD Bull ⬆")
            elif last["macd"] < last["macd_signal"]:
                sell_signals += 1; notes.append("MACD Bear ⬇")

        if pd.notna(last.get("bb_upper")) and pd.notna(last.get("bb_lower")):
            if last["Close"] < last["bb_lower"]:
                buy_signals += 1; notes.append("Price below BB ⬆")
            elif last["Close"] > last["bb_upper"]:
                sell_signals += 1; notes.append("Price above BB ⬇")

        if pd.notna(last.get("bb_width")) and pd.notna(last.get("atr")):
            if last["bb_width"] < last["atr"]:
                notes.append("Low volatility — weak signal ⚠️")

        # supertrend — булева серия
        try:
            if bool(df["supertrend"].iloc[-1]):
                buy_signals += 1; notes.append("SuperTrend Bull ⬆")
            else:
                sell_signals += 1; notes.append("SuperTrend Bear ⬇")
        except Exception:
            pass

        if pd.notna(last.get("k")):
            if last["k"] < 20:
                buy_signals += 1; notes.append("Stochastic Oversold ⬆")
            elif last["k"] > 80:
                sell_signals += 1; notes.append("Stochastic Overbought ⬇")

        if pd.notna(last.get("cci")):
            if last["cci"] < -100:
                buy_signals += 1; notes.append("CCI Oversold ⬆")
            elif last["cci"] > 100:
                sell_signals += 1; notes.append("CCI Overbought ⬇")

        for p in candle_patterns(df):
            if p in ["Hammer", "Bullish Candle"]:
                buy_signals += 1; notes.append(f"{p} ⬆")
            elif p in ["Inverted Hammer", "Bearish Candle"]:
                sell_signals += 1; notes.append(f"{p} ⬇")
            elif p == "Doji":
                notes.append("Doji — uncertainty ⚖️")

        final_signal = "❕ Нет явного сигнала"
        strength = "Low"
        if buy_signals >= 5:
            final_signal = "⬆ CALL"
            strength = "High" if buy_signals >= 7 else "Medium"
        elif sell_signals >= 5:
            final_signal = "⬇ PUT"
            strength = "High" if sell_signals >= 7 else "Medium"

        details = " | ".join(notes) if notes else "Нет деталей"
        return f"{final_signal} | Strength: {strength} | {details}"

    except Exception:
        logging.exception("Signal error")
        return None

# =====================================================
# Telegram обработчики
# =====================================================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("📈 Выбрать валютную пару", callback_data="choose_pair_0")],
        [InlineKeyboardButton("📜 История сделок", callback_data="history")]
    ]
    await update.message.reply_text("👋 Привет! Я торговый бот.\n\nВыбери действие:",
                                    reply_markup=InlineKeyboardMarkup(keyboard))

async def choose_pair(update, context, page=0):
    q = update.callback_query
    await q.answer()
    pairs = get_pairs_page(page)
    keyboard = [[InlineKeyboardButton(p, callback_data=f"pair_{p}")] for p in pairs]
    nav = []
    if page > 0: nav.append(InlineKeyboardButton("⬅ Назад", callback_data=f"choose_pair_{page-1}"))
    if page < total_pages(): nav.append(InlineKeyboardButton("Вперёд ➡", callback_data=f"choose_pair_{page+1}"))
    if nav: keyboard.append(nav)
    keyboard.append([InlineKeyboardButton("⬅ Главное меню", callback_data="back_to_menu")])
    await q.edit_message_text("⚡ Выберите валютную пару:", reply_markup=InlineKeyboardMarkup(keyboard))

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

async def ask_result(update, context, pair, exp):
    q = update.callback_query
    uid = q.from_user.id
    signal = generate_signal(pair, exp)
    if not signal:
        await q.edit_message_text("❌ Не удалось получить сигнал (нет данных).")
        return
    user_state[uid] = {"pair": pair, "exp": exp}
    k = [[InlineKeyboardButton("🟢 Плюс", callback_data="result_plus"),
          InlineKeyboardButton("🔴 Минус", callback_data="result_minus")]]
    await q.edit_message_text(f"📊 Сигнал: *{escape_md(signal)}*\nПара: *{escape_md(pair)}*\nЭкспирация: *{exp} мин*",
                              parse_mode="MarkdownV2", reply_markup=InlineKeyboardMarkup(k))

async def save_result(update, context, result):
    q = update.callback_query
    uid = q.from_user.id
    if uid not in trade_history: trade_history[uid] = []
    pair = user_state.get(uid, {}).get("pair", "—")
    exp = user_state.get(uid, {}).get("exp", "—")
    trade_history[uid].append(f"{pair} | {exp} мин — {result}")
    k = [
        [InlineKeyboardButton("📈 Новый сигнал", callback_data="choose_pair_0")],
        [InlineKeyboardButton("📜 История", callback_data="history")]
    ]
    await q.edit_message_text(f"Записано: *{escape_md(result)}*", parse_mode="MarkdownV2", reply_markup=InlineKeyboardMarkup(k))

async def history(update, context):
    q = update.callback_query
    uid = q.from_user.id
    if uid not in trade_history or len(trade_history[uid]) == 0:
        await q.edit_message_text("📭 История пустая.")
        return
    text = "📜 *История:*\n\n" + "\n".join([f"• {escape_md(t)}" for t in trade_history[uid]])
    k = [[InlineKeyboardButton("⬅ Главное меню", callback_data="back_to_menu")]]
    await q.edit_message_text(text, parse_mode="MarkdownV2", reply_markup=InlineKeyboardMarkup(k))

async def callbacks(update: Update, context: ContextTypes.DEFAULT_TYPE):
    data = update.callback_query.data
    if data.startswith("choose_pair_"): await choose_pair(update, context, int(data.split("_")[2]))
    elif data.startswith("pair_"): await choose_expiration(update, context, data.split("_")[1])
    elif data.startswith("exp_"):
        _, exp, pair = data.split("_")
        await ask_result(update, context, pair, int(exp))
    elif data == "result_plus": await save_result(update, context, "Плюс")
    elif data == "result_minus": await save_result(update, context, "Минус")
    elif data == "history": await history(update, context)
    elif data == "back_to_menu": await start(update, context)

# =====================================================
# Flask + webhook (без event-loop ошибок)
# =====================================================
BOT_TOKEN = os.getenv("BOT_TOKEN")
WEBHOOK_URL = os.getenv("WEBHOOK_URL")  # ставь базовый URL: https://your-app.onrender.com (без /webhook/...)

if not BOT_TOKEN:
    logging.error("BOT_TOKEN не указан. Прекращаю запуск.")
    raise SystemExit("BOT_TOKEN is required in environment")

# создаём приложение PTB
application = ApplicationBuilder().token(BOT_TOKEN).build()
application.add_handler(CommandHandler("start", start))
application.add_handler(CallbackQueryHandler(callbacks))

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return "Bot is running"

@app.route(f"/webhook/{BOT_TOKEN}", methods=["POST"])
def webhook():
    """
    Синхронный Flask-обработчик: создаём новый event loop для каждого запроса через asyncio.run.
    Это решает ошибки типа "There is no current event loop in thread".
    """
    try:
        data = request.get_json(force=True)
        logging.debug("Incoming update: %s", data)
        update = Update.de_json(data, application.bot)
        # безопасно запустить асинхронную обработку (создаст fresh loop)
        asyncio.run(application.process_update(update))
        return "OK", 200
    except Exception:
        logging.exception("Ошибка в webhook:")
        return "ERROR", 500

if __name__ == "__main__":
    # Попытка зарегистрировать webhook — поддерживаем два варианта WEBHOOK_URL:
    # - если ты указал базовый домен (https://...): код добавит /webhook/{BOT_TOKEN}
    # - если ты по какой-то причине указал полный путь, он будет использован как есть
    if WEBHOOK_URL:
        if "/webhook/" in WEBHOOK_URL:
            url = WEBHOOK_URL.rstrip("/")
        else:
            url = WEBHOOK_URL.rstrip("/") + f"/webhook/{BOT_TOKEN}"
        try:
            logging.info("Setting webhook to: %s", url)
            asyncio.run(application.bot.set_webhook(url))
            logging.info("Webhook установлен")
        except Exception:
            logging.exception("Не удалось установить webhook. Проверь BOT_TOKEN и WEBHOOK_URL")
    else:
        logging.warning("WEBHOOK_URL не задан — пропущена регистрация webhook (локальный запуск?)")

    port = int(os.getenv("PORT", 10000))
    # Запуск Flask — на проде вместо dev-server лучше gunicorn
    app.run(host="0.0.0.0", port=port)
