# main.py
import os
import asyncio
import logging
import sqlite3
import time
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, Any, List

import pandas as pd
import numpy as np
import yfinance as yf
import ta  # technical analysis helpers

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
    JobQueue,
)

# ---------------- Config ----------------
BOT_TOKEN = os.getenv("BOT_TOKEN")
WEBHOOK_URL = os.getenv("WEBHOOK_URL")  # e.g. https://your-app.onrender.com
PORT = int(os.getenv("PORT", "10000"))

if not BOT_TOKEN:
    raise SystemExit("Set BOT_TOKEN environment variable")

if not WEBHOOK_URL:
    # Try render hostname fallback (optional)
    render_host = os.getenv("RENDER_EXTERNAL_HOSTNAME")
    if render_host:
        WEBHOOK_URL = f"https://{render_host}"
    else:
        raise SystemExit("Set WEBHOOK_URL environment variable or RENDER_EXTERNAL_HOSTNAME")

WEBHOOK_PATH = f"/webhook/{BOT_TOKEN}"
FULL_WEBHOOK = f"{WEBHOOK_URL.rstrip('/')}{WEBHOOK_PATH}"

# Your trading pairs (user list)
PAIRS = [
    "EURUSD=X","GBPUSD=X","USDJPY=X","AUDUSD=X","USDCAD=X","USDCHF=X",
    "EURJPY=X","GBPJPY=X","AUDJPY=X","EURGBP=X","EURAUD=X","GBPAUD=X",
    "CADJPY=X","CHFJPY=X","EURCAD=X","GBPCAD=X","AUDCAD=X","AUDCHF=X","CADCHF=X"
]

SCAN_INTERVAL_SECONDS = 60  # scan every minute
CACHE_TTL = 30  # seconds for yf cache

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- Storage ----------------
DB_PATH = "signals.db"

def init_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        chat_id INTEGER PRIMARY KEY,
        created_at TEXT
    )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS signals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        pair TEXT,
        direction TEXT,
        expiration INTEGER,
        confidence INTEGER,
        ts TEXT,
        sent_to TEXT
    )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        signal_id INTEGER,
        chat_id INTEGER,
        feedback INTEGER,
        ts TEXT
    )""")
    conn.commit()
    return conn

DB = init_db()

# ---------------- yfinance cache + safe downloader ----------------
YF_CACHE: Dict[Tuple[str, str], Tuple[float, pd.DataFrame]] = {}

async def yf_download_safe(ticker: str, period: str = "2d", interval: str = "1m",
                           retries: int = 3, backoff: float = 1.0) -> pd.DataFrame:
    key = (ticker, interval)
    now = time.time()
    cached = YF_CACHE.get(key)
    if cached and now - cached[0] < CACHE_TTL:
        return cached[1].copy()

    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            df = await asyncio.to_thread(yf.download, ticker, period, interval, progress=False, auto_adjust=True, threads=False)
            if isinstance(df, pd.DataFrame) and not df.empty:
                YF_CACHE[key] = (time.time(), df.copy())
                return df
            last_exc = RuntimeError("Empty DataFrame")
            logger.debug("yfinance empty for %s %s (attempt %d)", ticker, interval, attempt)
        except Exception as e:
            last_exc = e
            logger.warning("yfinance error %s %s attempt %d: %s", ticker, interval, attempt, repr(e))
        await asyncio.sleep(backoff * attempt)
    logger.error("yfinance failed for %s %s: %s", ticker, interval, repr(last_exc))
    return pd.DataFrame()

# ---------------- Indicators & TA helpers ----------------
def ta_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    return ta.momentum.rsi(series, window=period, fillna=True)

def ta_ema(series: pd.Series, period: int = 20) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def ta_macd_vals(series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    macd = series.ewm(span=12, adjust=False).mean() - series.ewm(span=26, adjust=False).mean()
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

def ta_bollinger(series: pd.Series, period: int = 20) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ma = series.rolling(period).mean()
    std = series.rolling(period).std(ddof=0).fillna(0)
    upper = ma + 2 * std
    lower = ma - 2 * std
    return ma, upper, lower

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=period, fillna=True)

# ---------------- Candle patterns (simple) ----------------
def detect_candle_patterns(df: pd.DataFrame) -> List[str]:
    patterns = []
    if df.shape[0] < 2:
        return patterns
    last = df.iloc[-1]
    prev = df.iloc[-2]
    o, c, h, l = last['Open'], last['Close'], last['High'], last['Low']
    body = abs(c - o)
    rng = max(h - l, 1e-9)
    upper = h - max(c, o)
    lower = min(c, o) - l

    # Doji
    if body / rng < 0.15:
        patterns.append("Doji")
    # Hammer / Hanging man
    if lower > 2 * body and body > 0:
        patterns.append("Hammer")
    # Shooting star / Inverted hammer
    if upper > 2 * body and body > 0:
        patterns.append("ShootingStar")
    # Engulfing
    if (prev['Close'] < prev['Open'] and o < c and c > prev['Open'] and o <= prev['Close']):
        patterns.append("BullishEngulfing")
    if (prev['Close'] > prev['Open'] and o > c and c < prev['Open'] and o >= prev['Close']):
        patterns.append("BearishEngulfing")
    return patterns

# ---------------- Support / Resistance (simple) ----------------
def find_support_resistance(df: pd.DataFrame, lookback: int = 60) -> Tuple[List[float], List[float]]:
    # naive: local minima / maxima on Close over last lookback bars
    data = df['Close'].iloc[-lookback:]
    mins = data[(data.shift(1) > data) & (data.shift(-1) > data)].unique().tolist()
    maxs = data[(data.shift(1) < data) & (data.shift(-1) < data)].unique().tolist()
    return sorted(mins), sorted(maxs)

# ---------------- Signal computation ----------------
def score_signal(df: pd.DataFrame) -> Tuple[Optional[str], int, List[str]]:
    """
    Return (direction 'ВВЕРХ'|'ВНИЗ'|None, confidence 0-100, notes)
    """
    notes: List[str] = []
    if df.empty or len(df) < 10:
        return None, 0, ["Недостаточно данных"]

    df = df.tail(200).copy()
    df['rsi'] = ta_rsi(df['Close'])
    df['ema20'] = ta_ema(df['Close'], 20)
    df['ema50'] = ta_ema(df['Close'], 50)
    df['macd'], df['macd_sig'] = ta_macd_vals(df['Close'])
    df['bb_ma'], df['bb_up'], df['bb_low'] = ta_bollinger(df['Close'])
    df['atr'] = atr(df)

    last = df.iloc[-1]
    try:
        rsi_v = float(last['rsi'])
        ema20_v = float(last['ema20'])
        ema50_v = float(last['ema50'])
        macd_v = float(last['macd'])
        macd_sig_v = float(last['macd_sig'])
    except Exception:
        return None, 0, ["Ошибка вычисления индикаторов"]

    buy = 0.0
    sell = 0.0

    # RSI
    if rsi_v < 30:
        buy += 2.0; notes.append("RSI перепродан")
    elif rsi_v > 70:
        sell += 2.0; notes.append("RSI перекуплен")

    # EMA trend
    if ema20_v > ema50_v:
        buy += 1.5; notes.append("EMA20>EMA50")
    else:
        sell += 1.5; notes.append("EMA20<EMA50")

    # MACD
    if macd_v > macd_sig_v:
        buy += 1.5; notes.append("MACD bullish")
    else:
        sell += 1.5; notes.append("MACD bearish")

    # Bollinger
    if last['Close'] < last['bb_low']:
        buy += 1.2; notes.append("Цена ниже нижней BB")
    elif last['Close'] > last['bb_up']:
        sell += 1.2; notes.append("Цена выше верхней BB")

    # Candle patterns
    patterns = detect_candle_patterns(df)
    for p in patterns:
        if p in ("BullishEngulfing", "Hammer"):
            buy += 1.8; notes.append(f"Паттерн {p}")
        if p in ("BearishEngulfing", "ShootingStar"):
            sell += 1.8; notes.append(f"Паттерн {p}")

    # Support/resistance proximity
    mins, maxs = find_support_resistance(df, lookback=60)
    if maxs:
        nearest_res = min(maxs, key=lambda x: abs(x - last['Close']))
        if abs(last['Close'] - nearest_res) / (nearest_res + 1e-9) < 0.0025:  # within 0.25%
            sell += 1.0; notes.append("Близко к сопротивлению")
    if mins:
        nearest_sup = min(mins, key=lambda x: abs(x - last['Close']))
        if abs(last['Close'] - nearest_sup) / (nearest_sup + 1e-9) < 0.0025:
            buy += 1.0; notes.append("Близко к поддержке")

    # ATR-based volatility boost
    atr_v = float(last['atr'] if not pd.isna(last['atr']) else 0)
    if atr_v > df['atr'].median():
        # if volatility elevated, boost whichever side is winning
        if buy > sell:
            buy += 0.8; notes.append("ATR повышен (усиление BUY)")
        elif sell > buy:
            sell += 0.8; notes.append("ATR повышен (усиление SELL)")

    if buy > sell:
        direction = "ВВЕРХ"
        raw = buy - sell
    elif sell > buy:
        direction = "ВНИЗ"
        raw = sell - buy
    else:
        direction = None
        raw = 0.0

    # Normalize to 0-100
    max_possible = 12.0  # approximate
    confidence = int(max(0, min(99, (raw / max_possible) * 100)))
    # minimum confidence floor
    if confidence > 0:
        confidence = max(confidence, 35)

    return direction, confidence, notes

# ---------------- History helpers ----------------
def save_signal(pair: str, direction: str, expiration: int, confidence: int, sent_to: List[int]):
    cur = DB.cursor()
    cur.execute("INSERT INTO signals(pair,direction,expiration,confidence,ts,sent_to) VALUES(?,?,?,?,?,?)",
                (pair, direction, expiration, confidence, datetime.utcnow().isoformat(), ",".join(map(str, sent_to))))
    DB.commit()
    return cur.lastrowid

def save_feedback(signal_id: int, chat_id: int, feedback: int):
    cur = DB.cursor()
    cur.execute("INSERT INTO feedback(signal_id,chat_id,feedback,ts) VALUES(?,?,?,?)",
                (signal_id, chat_id, feedback, datetime.utcnow().isoformat()))
    DB.commit()

def add_user(chat_id: int):
    cur = DB.cursor()
    cur.execute("INSERT OR IGNORE INTO users(chat_id,created_at) VALUES(?,?)", (chat_id, datetime.utcnow().isoformat()))
    DB.commit()

def remove_user(chat_id: int):
    cur = DB.cursor()
    cur.execute("DELETE FROM users WHERE chat_id=?", (chat_id,))
    DB.commit()

def get_subscribers() -> List[int]:
    cur = DB.cursor()
    cur.execute("SELECT chat_id FROM users")
    return [r[0] for r in cur.fetchall()]

# ---------------- Bot UI ----------------
def pair_keyboard() -> InlineKeyboardMarkup:
    keys = []
    for i in range(0, len(PAIRS), 3):
        row = [InlineKeyboardButton(p.replace("=X",""), callback_data=f"pair|{p}") for p in PAIRS[i:i+3]]
        keys.append(row)
    keys.append([InlineKeyboardButton("Отписаться", callback_data="unsub")])
    return InlineKeyboardMarkup(keys)

# ---------------- Handlers ----------------
async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    add_user(chat_id)
    await update.message.reply_text(
        "Привет! Ты подписан на сигналы.\n"
        "Бот будет сканировать рынок и отправлять лучшие сигналы.\n"
        "Команды: /stop - отписаться, /history - твоя история, /last - последний сигнал",
        reply_markup=pair_keyboard()
    )

async def stop_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    remove_user(chat_id)
    await update.message.reply_text("Вы отписаны от автосигналов. Чтобы подписаться снова — /start")

async def history_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    cur = DB.cursor()
    cur.execute("SELECT id,pair,direction,expiration,confidence,ts FROM signals WHERE sent_to LIKE ? ORDER BY id DESC LIMIT 20", (f"%{chat_id}%",))
    rows = cur.fetchall()
    if not rows:
        await update.message.reply_text("Твоя история пуста.")
        return
    text = "Последние сигналы, отправленные тебе:\n"
    for r in rows:
        text += f"{r[5][:19]} | {r[1].replace('=X','')} | {r[2]} | exp {r[3]}m | {r[4]}%\n"
    await update.message.reply_text(text)

async def last_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    cur = DB.cursor()
    cur.execute("SELECT id,pair,direction,expiration,confidence,ts FROM signals ORDER BY id DESC LIMIT 1")
    r = cur.fetchone()
    if not r:
        await update.message.reply_text("Нет сигналов пока.")
        return
    await update.message.reply_text(f"{r[5][:19]} | {r[1].replace('=X','')} | {r[2]} | exp {r[3]}m | {r[4]}%")

async def callback_query_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    data = q.data or ""
    if data.startswith("pair|"):
        pair = data.split("|",1)[1]
        # one-off analyze on demand
        df = await yf_download_safe(pair, period="2d", interval="1m")
        direction, confidence, notes = score_signal(df)
        if not direction:
            await q.edit_message_text(f"Нет сигнала для {pair.replace('=X','')}.\nПричины: {';'.join(notes)}")
            return
        # choose expiration 1-15 by confidence (higher -> longer)
        expiration = max(1, min(15, int(1 + (confidence/100)*14)))
        text = (f"📊 Пара: {pair.replace('=X','')}\n"
                f"Сигнал: {direction}\n"
                f"Экспирация: {expiration} мин\n"
                f"Уверенность: {confidence}%\n"
                f"Причины: {', '.join(notes[:6])}")
        # store temporary signal in DB with sent_to current chat only
        sid = save_signal(pair, direction, expiration, confidence, [q.from_user.id])
        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("✅ +", callback_data=f"fb+|{sid}"), InlineKeyboardButton("❌ -", callback_data=f"fb-|{sid}")],
        ])
        await q.edit_message_text(text, reply_markup=kb)
    elif data.startswith("fb+") or data.startswith("fb-"):
        parts = data.split("|")
        fb = 1 if parts[0] == "fb+" else -1
        sid = int(parts[1])
        save_feedback(sid, q.from_user.id, fb)
        await q.edit_message_text("Спасибо за обратную связь!")

    elif data == "unsub":
        remove_user(q.from_user.id)
        await q.edit_message_text("Вы отписаны от автосигналов.")

# ---------------- Scanner job ----------------
async def scan_and_broadcast(context: ContextTypes.DEFAULT_TYPE):
    logger.info("Сканирую пары...")
    best: Optional[Dict[str, Any]] = None
    for pair in PAIRS:
        df = await yf_download_safe(pair, period="2d", interval="1m")
        if df.empty:
            continue
        direction, confidence, notes = score_signal(df)
        if direction and confidence > 40:
            # choose expiration by confidence
            expiration = max(1, min(15, int(1 + (confidence/100)*14)))
            entry = {"pair": pair, "direction": direction, "confidence": confidence, "expiration": expiration, "notes": notes}
            if not best or confidence > best["confidence"]:
                best = entry
    if not best:
        logger.info("Нет подходящих сигналов в этом цикле.")
        return

    subscribers = get_subscribers()
    if not subscribers:
        logger.info("Нет подписчиков — прекращаю.")
        return

    text = (f"🟢 Умный сигнал\nПара: {best['pair'].replace('=X','')}\nНаправление: {best['direction']}\n"
            f"Экспирация: {best['expiration']} мин\nУверенность: {best['confidence']}%\nПричины: {', '.join(best['notes'][:6])}")
    kb = InlineKeyboardMarkup([[InlineKeyboardButton("✅ +", callback_data=f"fb+|{0}"),
                                InlineKeyboardButton("❌ -", callback_data=f"fb-|{0}")]])
    # Save signal and broadcast to users
    sid = save_signal(best['pair'], best['direction'], best['expiration'], best['confidence'], subscribers)
    # replace 0 in keyboard with real sid using callback_data by creating personalized keyboard per chat
    for chat_id in subscribers:
        kb_user = InlineKeyboardMarkup([[InlineKeyboardButton("✅ +", callback_data=f"fb+|{sid}"),
                                         InlineKeyboardButton("❌ -", callback_data=f"fb-|{sid}")]])
        try:
            await context.bot.send_message(chat_id, text, reply_markup=kb_user)
        except Exception as e:
            logger.warning("Failed to send to %s: %s", chat_id, e)

# ---------------- Main app setup ----------------
def build_application():
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start_handler))
    app.add_handler(CommandHandler("stop", stop_handler))
    app.add_handler(CommandHandler("history", history_handler))
    app.add_handler(CommandHandler("last", last_handler))
    app.add_handler(CallbackQueryHandler(callback_query_handler))
    # schedule scanning job
    app.job_queue.run_repeating(scan_and_broadcast, interval=SCAN_INTERVAL_SECONDS, first=10)
    return app

# ---------------- Run webhook ----------------
if __name__ == "__main__":
    app = build_application()
    logger.info("Setting webhook to %s", FULL_WEBHOOK)
    # run_webhook will manage the asyncio loop correctly for PTB
    app.run_webhook(
        listen="0.0.0.0",
        port=PORT,
        url_path=BOT_TOKEN,
        webhook_url=FULL_WEBHOOK,
    )
