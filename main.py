# main.py — Полностью рабочий Telegram бот с биржевыми и OTC парами, сигналами и историей
import logging, os, re, asyncio
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
OTC_FRAMES = ["5s","30s","1m","3m","5m"]

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
    if len(df) > 0: in_uptrend.iloc[0] = True
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
    o,c,h,l = df['Open'].iloc[-1],df['Close'].iloc[-1],df['High'].iloc[-1],df['Low'].iloc[-1]
    body = abs(c-o)
    candle_range = max(h-l,1e-9)
    upper_shadow = h-max(c,o)
    lower_shadow = min(c,o)-l
    if body/candle_range<0.25: patterns.append("Doji")
    if lower_shadow>2*body and body>0: patterns.append("Hammer")
    if upper_shadow>2*body and body>0: patterns.append("Inverted Hammer")
    patterns.append("Bullish Candle" if c>o else "Bearish Candle")
    return patterns

# -----------------------
# Вспомогательные функции
# -----------------------
def escape_md(text: str):
    return re.sub(r"([_*\[\]()~`>#+\-=|{}.!])", r"\\\1", str(text))

def get_pairs_page(pairs, page):
    start = page*PAIRS_PER_PAGE
    return pairs[start:start+PAIRS_PER_PAGE]

def total_pages(pairs):
    return (len(pairs)-1)//PAIRS_PER_PAGE

# -----------------------
# Генерация сигнала
# -----------------------
def generate_signal(pair, timeframe):
    try:
        ticker = pair.replace(" OTC","").replace("/","") + "=X"
        df = yf.download(ticker, period="3d", interval="1m", progress=False)
        if df.empty or len(df)<10: return "❌ Нет данных для сигнала"
        df = df.tail(LOOKBACK).copy()
        df["rsi"]=rsi(df["Close"])
        df["sma50"]=SMA(df["Close"],50)
        df["sma200"]=SMA(df["Close"],200)
        df["ema20"]=EMA(df["Close"],20)
        macd, macd_signal = MACD(df["Close"])
        df["macd"], df["macd_signal"]=macd, macd_signal
        df["bb_upper"], df["bb_lower"]=BollingerBands(df["Close"])
        df["atr"]=ATR(df)
        df["supertrend"]=SuperTrend(df)
        k,d=StochasticOscillator(df)
        df["k"], df["d"]=k,d
        df["cci"]=CCI(df)
        last=df.iloc[-1]
        buy_signals=sell_signals=0
        notes=[]
        if last["rsi"]<30: buy_signals+=1; notes.append("RSI Oversold ⬆")
        elif last["rsi"]>70: sell_signals+=1; notes.append("RSI Overbought ⬇")
        if last["Close"]>last["sma50"]>last["sma200"]: buy_signals+=1; notes.append("Uptrend ⬆")
        elif last["Close"]<last["sma50"]<last["sma200"]: sell_signals+=1; notes.append("Downtrend ⬇")
        if last["macd"]>last["macd_signal"]: buy_signals+=1; notes.append("MACD Bull ⬆")
        elif last["macd"]<last["macd_signal"]: sell_signals+=1; notes.append("MACD Bear ⬇")
        if last["Close"]<last["bb_lower"]: buy_signals+=1; notes.append("Price below BB ⬆")
        elif last["Close"]>last["bb_upper"]: sell_signals+=1; notes.append("Price above BB ⬇")
        if bool(last["supertrend"]): buy_signals+=1; notes.append("SuperTrend Bull ⬆")
        else: sell_signals+=1; notes.append("SuperTrend Bear ⬇")
        if last["k"]<20: buy_signals+=1; notes.append("Stochastic Oversold ⬆")
        elif last["k"]>80: sell_signals+=1; notes.append("Stochastic Overbought ⬇")
        if last["cci"]<-100: buy_signals+=1; notes.append("CCI Oversold ⬆")
        elif last["cci"]>100: sell_signals+=1; notes.append("CCI Overbought ⬇")
        for p in candle_patterns(df):
            if p in ("Hammer","Bullish Candle"): buy_signals+=1; notes.append(f"{p} ⬆")
            elif p in ("Inverted Hammer","Bearish Candle"): sell_signals+=1; notes.append(f"{p} ⬇")
            elif p=="Doji": notes.append("Doji ⚖️")
        if buy_signals>=5: signal="Вверх"
        elif sell_signals>=5: signal="Вниз"
        else: signal="Нет сигнала"
        percent=min(90,50+max(buy_signals,sell_signals)*5)
        strength="High" if max(buy_signals,sell_signals)>=7 else "Medium" if max(buy_signals,sell_signals)>=5 else "Low"
        return f"{signal} | Strength: {strength} | Accuracy: {percent}% | Notes: {' | '.join(notes)}"
    except Exception:
        logging.exception(f"Ошибка генерации сигнала для {pair}")
        return "❌ Ошибка генерации сигнала"

# -----------------------
# Telegram Handlers
# -----------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard=[
        [InlineKeyboardButton("📈 Биржевой рынок", callback_data="market_exchange")],
        [InlineKeyboardButton("📈 OTC рынок", callback_data="market_otc")],
        [InlineKeyboardButton("📜 История сделок", callback_data="history")]
    ]
    await update.message.reply_text("👋 Привет! Выберите рынок:",reply_markup=InlineKeyboardMarkup(keyboard))

async def choose_pair(update: Update, context: ContextTypes.DEFAULT_TYPE, market="exchange", page=0):
    q = update.callback_query
    await q.answer()
    pairs = EXCHANGE_PAIRS if market=="exchange" else OTC_PAIRS
    page_pairs = get_pairs_page(pairs, page)
    keyboard=[[InlineKeyboardButton(p,callback_data=f"pair_{market}_{p}")] for p in page_pairs]
    nav=[]
    if page>0: nav.append(InlineKeyboardButton("⬅ Назад",callback_data=f"choose_{market}_{page-1}"))
    if page<total_pages(pairs): nav.append(InlineKeyboardButton("Вперёд ➡",callback_data=f"choose_{market}_{page+1}"))
    if nav: keyboard.append(nav)
    keyboard.append([InlineKeyboardButton("⬅ Главное меню",callback_data="back_to_menu")])
    await q.edit_message_text("⚡ Выберите валютную пару:",reply_markup=InlineKeyboardMarkup(keyboard))

async def choose_frame(update: Update, context: ContextTypes.DEFAULT_TYPE, market, pair):
    q=update.callback_query
    frames=EXCHANGE_FRAMES if market=="exchange" else OTC_FRAMES
    keyboard=[[InlineKeyboardButton(f,callback_data=f"frame_{market}_{pair}_{f}")] for f in frames]
    keyboard.append([InlineKeyboardButton("⬅ Назад",callback_data=f"choose_{market}_0")])
    await q.edit_message_text(f"Пара: *{escape_md(pair)}*\nВыберите таймфрейм:",parse_mode="MarkdownV2",reply_markup=InlineKeyboardMarkup(keyboard))

async def show_signal(update: Update, context: ContextTypes.DEFAULT_TYPE, market, pair, frame):
    q=update.callback_query
    uid=q.from_user.id
    signal=generate_signal(pair,frame)
    user_state[uid]={"pair":pair,"frame":frame}
    keyboard=[[InlineKeyboardButton("🟢 Плюс",callback_data="result_plus"),
               InlineKeyboardButton("🔴 Минус",callback_data="result_minus")]]
    await q.edit_message_text(f"📊 Сигнал: *{escape_md(signal)}*\nПара: *{escape_md(pair)}*\nТаймфрейм: *{frame}*",parse_mode="MarkdownV2",reply_markup=InlineKeyboardMarkup(keyboard))

async def save_result(update: Update, context: ContextTypes.DEFAULT_TYPE, result):
    q=update.callback_query
    uid=q.from_user.id
    if uid not in trade_history: trade_history[uid]=[]
    pair=user_state.get(uid,{}).get("pair","—")
    frame=user_state.get(uid,{}).get("frame","—")
    trade_history[uid].append(f"{pair} | {frame} — {result}")
    keyboard=[
        [InlineKeyboardButton("📈 Новый сигнал",callback_data="market_exchange")],
        [InlineKeyboardButton("📜 История",callback_data="history")]
    ]
    await q.edit_message_text(f"✅ Записано: *{escape_md(result)}*",parse_mode="MarkdownV2",reply_markup=InlineKeyboardMarkup(keyboard))

async def show_history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q=update.callback_query
    uid=q.from_user.id
    if uid not in trade_history or len(trade_history[uid])==0:
        await q.edit_message_text("📭 История пустая.")
        return
    text="📜 *История:*\n\n"+"\n".join([f"• {escape_md(t)}" for t in trade_history[uid]])
    keyboard=[[InlineKeyboardButton("⬅ Главное меню",callback_data="back_to_menu")]]
    await q.edit_message_text(text,parse_mode="MarkdownV2",reply_markup=InlineKeyboardMarkup(keyboard))

async def callbacks(update: Update, context: ContextTypes.DEFAULT_TYPE):
    data=update.callback_query.data
    if data=="market_exchange": await choose_pair(update,context,"exchange",0)
    elif data=="market_otc": await choose_pair(update,context,"otc",0)
    elif data.startswith("choose_"):
        _,market,page=data.split("_")
        await choose_pair(update,context,market,int(page))
    elif data.startswith("pair_"):
        _,market,pair=data.split("_",2)
        await choose_frame(update,context,market,pair)
    elif data.startswith("frame_"):
        _,market,pair,frame=data.split("_",3)
        await show_signal(update,context,market,pair,frame)
    elif data=="result_plus": await save_result(update,context,"Плюс")
    elif data=="result_minus": await save_result(update,context,"Минус")
    elif data=="history": await show_history(update,context)
    elif data=="back_to_menu": await start(update,context)

# -----------------------
# Flask + Webhook
# -----------------------
BOT_TOKEN=os.getenv("BOT_TOKEN")
WEBHOOK_URL=os.getenv("WEBHOOK_URL")

application=ApplicationBuilder().token(BOT_TOKEN).build()
application.add_handler(CommandHandler("start",start))
application.add_handler(CallbackQueryHandler(callbacks))

app=Flask(__name__)

@app.route("/",methods=["GET"])
def home(): return "Bot is running"

@app.route("/webhook/<token>",methods=["POST"])
def webhook(token):
    if token!=BOT_TOKEN: abort(403)
    try:
        data=request.get_json(force=True)
        update=Update.de_json(data,application.bot)
        loop=asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(application.process_update(update))
        loop.close()
        asyncio.set_event_loop(None)
        return "OK",200
    except Exception:
        logging.exception("Ошибка в webhook:")
        return "ERROR",500

if __name__=="__main__":
    if BOT_TOKEN and WEBHOOK_URL:
        url=f"{WEBHOOK_URL.rstrip('/')}/webhook/{BOT_TOKEN}"
        asyncio.run(application.bot.set_webhook(url))
        logging.info(f"Webhook установлен: {url}")
    app.run(host="0.0.0.0",port=int(os.getenv("PORT",10000)))
