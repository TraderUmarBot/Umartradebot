import logging
import pandas as pd
import yfinance as yf
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Application, ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes
from flask import Flask, request
import os
import re
import asyncio

logging.basicConfig(level=logging.INFO)

# =====================================================
# ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ
# =====================================================
user_state = {}
trade_history = {}

ALL_PAIRS = [
    "EUR/USD","GBP/USD","USD/JPY","AUD/USD","USD/CAD","USD/CHF",
    "EUR/JPY","GBP/JPY","AUD/JPY","EUR/GBP","EUR/AUD","GBP/AUD",
    "CAD/JPY","CHF/JPY","EUR/CAD","GBP/CAD","AUD/CAD","AUD/CHF","CAD/CHF"
]
PAIRS_PER_PAGE = 6
LOOKBACK = 30  # Количество свечей для анализа

# =====================================================
# Технические индикаторы
# =====================================================
def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def SMA(series, period=50):
    return series.rolling(period).mean()

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
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = sma + mult*std
    lower = sma - mult*std
    return upper, lower

def ATR(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def SuperTrend(df, period=7, multiplier=3):
    hl2 = (df['High'] + df['Low']) / 2
    atr = ATR(df, period)
    upperband = hl2 + multiplier * atr
    lowerband = hl2 - multiplier * atr
    trend = pd.Series(index=df.index, data=True)
    for i in range(1, len(df)):
        if df['Close'][i] > upperband[i-1]:
            trend[i] = True
        elif df['Close'][i] < lowerband[i-1]:
            trend[i] = False
        else:
            trend[i] = trend[i-1]
    return trend.map({True: df['Close'], False: df['Close']})

def StochasticOscillator(df, k_period=14, d_period=3):
    low_min = df['Low'].rolling(k_period).min()
    high_max = df['High'].rolling(k_period).max()
    k = 100 * ((df['Close'] - low_min) / (high_max - low_min))
    d = k.rolling(d_period).mean()
    return k, d

def CCI(df, period=20):
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    cci = (tp - tp.rolling(period).mean()) / (0.015 * tp.rolling(period).std())
    return cci

# ---------- Свечные паттерны ----------
def candle_patterns(df):
    patterns = []
    o = df['Open'].iloc[-1]
    c = df['Close'].iloc[-1]
    h = df['High'].iloc[-1]
    l = df['Low'].iloc[-1]
    body = abs(c - o)
    candle_range = h - l
    upper_shadow = h - max(c,o)
    lower_shadow = min(c,o) - l
    if body / candle_range < 0.3:
        patterns.append("Doji")
    if lower_shadow > 2*body:
        patterns.append("Hammer")
    if upper_shadow > 2*body:
        patterns.append("Inverted Hammer")
    if c > o:
        patterns.append("Bullish Engulfing")
    else:
        patterns.append("Bearish Engulfing")
    return patterns

# =====================================================
# Вспомогательные функции
# =====================================================
def escape_md(text: str):
    return re.sub(r"([_*\[\]()~`>#+\-=|{}.!])", r"\\\1", text)

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
        df = yf.download(pair.replace("/", ""), period="3d", interval="1m")
        if df.empty: return None
        df = df.tail(LOOKBACK)

        # ====== Индикаторы ======
        df["rsi"] = rsi(df["Close"])
        df["sma50"] = SMA(df["Close"], 50)
        df["sma200"] = SMA(df["Close"], 200)
        df["ema20"] = EMA(df["Close"], 20)
        df["macd"], df["macd_signal"], _ = MACD(df["Close"])
        df["bb_upper"], df["bb_lower"] = BollingerBands(df["Close"])
        df["bb_width"] = df["bb_upper"] - df["bb_lower"]
        df["atr"] = ATR(df)
        df["supertrend"] = SuperTrend(df)
        df["k"], df["d"] = StochasticOscillator(df)
        df["cci"] = CCI(df)

        last = df.iloc[-1]
        buy_signals = 0
        sell_signals = 0
        notes = []

        # ===== Индикаторы =====
        if last["rsi"] < 30: buy_signals +=1; notes.append("RSI Oversold ⬆")
        elif last["rsi"] > 70: sell_signals +=1; notes.append("RSI Overbought ⬇")
        if last["Close"] > last["sma50"] > last["sma200"]: buy_signals +=1; notes.append("Uptrend ⬆")
        elif last["Close"] < last["sma50"] < last["sma200"]: sell_signals +=1; notes.append("Downtrend ⬇")
        if last["macd"] > last["macd_signal"]: buy_signals +=1; notes.append("MACD Bullish ⬆")
        elif last["macd"] < last["macd_signal"]: sell_signals +=1; notes.append("MACD Bearish ⬇")
        if last["Close"] > last["bb_upper"]: sell_signals +=1; notes.append("Price above BB ⬇")
        elif last["Close"] < last["bb_lower"]: buy_signals +=1; notes.append("Price below BB ⬆")
        if last["bb_width"] < last["atr"]: notes.append("Low volatility — сигнал слабый ⚠️")
        if last["Close"] > last["supertrend"]: buy_signals +=1; notes.append("SuperTrend Bull ⬆")
        else: sell_signals +=1; notes.append("SuperTrend Bear ⬇")
        if last["k"] < 20: buy_signals +=1; notes.append("Stochastic Oversold ⬆")
        elif last["k"] > 80: sell_signals +=1; notes.append("Stochastic Overbought ⬇")
        if last["cci"] < -100: buy_signals +=1; notes.append("CCI Oversold ⬆")
        elif last["cci"] > 100: sell_signals +=1; notes.append("CCI Overbought ⬇")

        # ===== Свечные паттерны =====
        patterns = candle_patterns(df)
        for p in patterns:
            if p in ["Hammer", "Bullish Engulfing"]: buy_signals +=1; notes.append(f"{p} ⬆")
            elif p in ["Inverted Hammer", "Bearish Engulfing"]: sell_signals +=1; notes.append(f"{p} ⬇")
            elif p == "Doji": notes.append("Doji — неопределённость ⚖️")

        # ===== Итог =====
        final_signal = "❕ Нет явного сигнала"
        strength = "Low"
        if buy_signals >= 5: final_signal = "⬆ CALL"; strength = "High" if buy_signals >=7 else "Medium"
        elif sell_signals >= 5: final_signal = "⬇ PUT"; strength = "High" if sell_signals >=7 else "Medium"

        return f"{final_signal} | Сила: {strength} | Детали: {' | '.join(notes)}"
    except Exception as e:
        logging.error(f"Signal error: {e}")
        return None

# =====================================================
# Telegram + Flask код
# =====================================================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("📈 Выбрать валютную пару", callback_data="choose_pair_0")],
        [InlineKeyboardButton("📜 История сделок", callback_data="history")]
    ]
    await update.message.reply_text(
        "👋 Привет! Я торговый бот.\n\nВыбери действие:",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

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
        await q.edit_message_text("❌ Не удалось получить сигнал.")
        return
    user_state[uid] = {"pair": pair, "exp": exp}
    k = [[InlineKeyboardButton("🟢 Плюс", callback_data="result_plus"), InlineKeyboardButton("🔴 Минус", callback_data="result_minus")]]
    await q.edit_message_text(f"📊 Сигнал: *{escape_md(signal)}*\nПара: *{escape_md(pair)}*\nЭкспирация: *{exp} мин*",
                              parse_mode="MarkdownV2", reply_markup=InlineKeyboardMarkup(k))

async def save_result(update, context, result):
    q = update.callback_query
    uid = q.from_user.id
    if uid not in trade_history: trade_history[uid] = []
    pair = user_state[uid]["pair"]
    exp = user_state[uid]["exp"]
    trade_history[uid].append(f"{pair} | {exp} мин — {result}")
    k = [[InlineKeyboardButton("📈 Новый сигнал", callback_data="choose_pair_0")],
         [InlineKeyboardButton("📜 История", callback_data="history")]]
    await q.edit_message_text(f"Записано: *{escape_md(result)}*", parse_mode="MarkdownV2", reply_markup=InlineKeyboardMarkup(k))

async def history(update, context):
    q = update.callback_query
    uid = q.from_user.id
    if uid not in trade_history or len(trade_history[uid]) == 0:
        await q.edit_message_text("📭 История пустая.")
        return
    text = "📜 *История:*\n\n"
    for t in trade_history[uid]: text += f"• {escape_md(t)}\n"
    k = [[InlineKeyboardButton("⬅ Главное меню", callback_data="back_to_menu")]]
    await q.edit_message_text(text, parse_mode="MarkdownV2", reply_markup=InlineKeyboardMarkup(k))

async def callbacks(update: Update, context: ContextTypes.DEFAULT_TYPE):
    data = update.callback_query.data
    if data.startswith("choose_pair_"): await choose_pair(update, context, int(data.split("_")[2]))
    elif data.startswith("pair_"): await choose_expiration(update, context, data.split("_")[1])
    elif data.startswith("exp_"): _, exp, pair = data.split("_"); await ask_result(update, context, pair, int(exp))
    elif data == "result_plus": await save_result(update, context, "Плюс")
    elif data == "result_minus": await save_result(update, context, "Минус")
    elif data == "history": await history(update, context)
    elif data == "back_to_menu": await start(update, context)

# ====================== FLASK + WEBHOOK ======================
BOT_TOKEN = os.getenv("BOT_TOKEN")
WEBHOOK_URL = os.getenv("WEBHOOK_URL")
app = Flask(__name__)
application = ApplicationBuilder().token(BOT_TOKEN).build()
application.add_handler(CommandHandler("start", start))
application.add_handler(CallbackQueryHandler(callbacks))

@app.route("/", methods=["GET"])
def home(): return "Bot is running"

@app.route(f"/webhook/{BOT_TOKEN}", methods=["POST"])
async def webhook():
    update = Update.de_json(request.get_json(force=True), application.bot)
    await application.process_update(update)
    return "OK", 200

if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(application.bot.set_webhook(f"{WEBHOOK_URL}/webhook/{BOT_TOKEN}"))
    port = int(os.getenv("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
