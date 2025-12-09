# main.py — webhook-ready, PTB 20.7, uses env BOT_TOKEN and WEBHOOK_URL
import os
import logging
import asyncio
from datetime import datetime
import time
import sqlite3
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
import yfinance as yf
import ta
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Config from env
BOT_TOKEN = os.getenv("BOT_TOKEN")
WEBHOOK_URL = os.getenv("WEBHOOK_URL") or os.getenv("RENDER_EXTERNAL_HOSTNAME")
PORT = int(os.getenv("PORT", "10000"))

if not BOT_TOKEN:
    raise SystemExit("ERROR: BOT_TOKEN environment variable is not set.")

if not WEBHOOK_URL:
    raise SystemExit("ERROR: WEBHOOK_URL or RENDER_EXTERNAL_HOSTNAME must be set.")

FULL_WEBHOOK = f"https://{WEBHOOK_URL.strip('/')}/webhook/{BOT_TOKEN}"

PAIRS = [
    "EURUSD=X","GBPUSD=X","USDJPY=X","AUDUSD=X","USDCAD=X","USDCHF=X",
    "EURJPY=X","GBPJPY=X","AUDJPY=X","EURGBP=X","EURAUD=X","GBPAUD=X",
    "CADJPY=X","CHFJPY=X","EURCAD=X","GBPCAD=X","AUDCAD=X","AUDCHF=X","CADCHF=X"
]

DB_PATH = "signals.db"

# SQLite init
def init_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (chat_id INTEGER PRIMARY KEY, created_at TEXT)
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pair TEXT,
            direction TEXT,
            expiration INTEGER,
            confidence INTEGER,
            ts TEXT,
            sent_to TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            signal_id INTEGER,
            chat_id INTEGER,
            feedback INTEGER,
            ts TEXT
        )
    """)
    conn.commit()
    return conn

DB = init_db()

# Simple safe yfinance fetch (async)
YF_CACHE: Dict[Tuple[str,str], Tuple[float, pd.DataFrame]] = {}
CACHE_TTL = 30

async def yf_safe(ticker: str, period: str="2d", interval: str="1m") -> pd.DataFrame:
    key = (ticker, interval)
    now = time.time()
    cached = YF_CACHE.get(key)
    if cached and now - cached[0] < CACHE_TTL:
        return cached[1].copy()
    try:
        df = await asyncio.to_thread(yf.download, ticker, period, interval, progress=False, auto_adjust=True, threads=False)
        if isinstance(df, pd.DataFrame) and not df.empty:
            YF_CACHE[key] = (time.time(), df.copy())
            return df
    except Exception as e:
        logger.warning("yfinance error for %s: %s", ticker, e)
    return pd.DataFrame()

# Indicators
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rsi"] = ta.momentum.rsi(df["Close"], window=14, fillna=True)
    df["ema20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["Close"].ewm(span=50, adjust=False).mean()
    macd = df["Close"].ewm(span=12, adjust=False).mean() - df["Close"].ewm(span=26, adjust=False).mean()
    df["macd"], df["macd_sig"] = macd, macd.ewm(span=9, adjust=False).mean()
    return df

def score(df: pd.DataFrame) -> Tuple[Optional[str], int, List[str]]:
    notes = []
    if df.empty or len(df) < 20:
        return None, 0, ["not enough data"]
    df = compute_indicators(df.tail(200))
    last = df.iloc[-1]
    buy = sell = 0.0

    # RSI
    if last["rsi"] < 30:
        buy += 2; notes.append("RSI перепродан")
    elif last["rsi"] > 70:
        sell += 2; notes.append("RSI перекуплен")

    # EMA
    if last["ema20"] > last["ema50"]:
        buy += 1.5; notes.append("EMA20>EMA50")
    else:
        sell += 1.5; notes.append("EMA20<EMA50")

    # MACD
    if last["macd"] > last["macd_sig"]:
        buy += 1.5; notes.append("MACD bullish")
    else:
        sell += 1.5; notes.append("MACD bearish")

    if buy > sell:
        dirc = "ВВЕРХ"
        raw = buy - sell
    elif sell > buy:
        dirc = "ВНИЗ"
        raw = sell - buy
    else:
        dirc = None
        raw = 0
    confidence = int(min(95, max(0, (raw / 8.0) * 100)))
    if confidence and confidence < 35:
        confidence = 35
    return dirc, confidence, notes

# DB helpers
def add_user(chat_id: int):
    cur = DB.cursor()
    cur.execute("INSERT OR IGNORE INTO users(chat_id,created_at) VALUES(?,?)", (chat_id, datetime.utcnow().isoformat()))
    DB.commit()

def remove_user(chat_id: int):
    cur = DB.cursor()
    cur.execute("DELETE FROM users WHERE chat_id=?", (chat_id,))
    DB.commit()

def get_users() -> List[int]:
    cur = DB.cursor()
    cur.execute("SELECT chat_id FROM users")
    return [r[0] for r in cur.fetchall()]

def save_signal(pair: str, direction: str, expiration: int, confidence: int, sent_to: List[int]) -> int:
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

# Bot handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    add_user(update.effective_chat.id)
    keyboard = InlineKeyboardMarkup([[InlineKeyboardButton(p.replace("=X",""), callback_data=f"pair|{p}") for p in PAIRS[:3]]])
    await update.message.reply_text("Подписан. Бот будет присылать сигналы.", reply_markup=keyboard)

async def stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    remove_user(update.effective_chat.id)
    await update.message.reply_text("Отписал тебя от сигналов.")

async def history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    cur = DB.cursor()
    cur.execute("SELECT id,pair,direction,expiration,confidence,ts FROM signals ORDER BY id DESC LIMIT 20")
    rows = cur.fetchall()
    if not rows:
        await update.message.reply_text("История пуста.")
        return
    text = "\n".join([f"{r[5][:19]} | {r[1].replace('=X','')} | {r[2]} | exp {r[3]}m | {r[4]}%" for r in rows])
    await update.message.reply_text(text)

async def callback_q(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    data = q.data or ""
    if data.startswith("pair|"):
        pair = data.split("|",1)[1]
        df = await yf_safe(pair, period="2d", interval="1m")
        direction, confidence, notes = score(df)
        if not direction:
            await q.edit_message_text(f"Нет сигнала для {pair.replace('=X','')}. Причины: {', '.join(notes)}")
            return
        expiration = max(1, min(15, int(1 + (confidence/100)*14)))
        sid = save_signal(pair, direction, expiration, confidence, [q.from_user.id])
        kb = InlineKeyboardMarkup([[InlineKeyboardButton("✅ +", callback_data=f"fb+|{sid}"),
                                    InlineKeyboardButton("❌ -", callback_data=f"fb-|{sid}")]])
        text = (f"Пара: {pair.replace('=X','')}\nСигнал: {direction}\nЭкспирация: {expiration}мин\nУверенность: {confidence}%\nПричины: {', '.join(notes[:6])}")
        await q.edit_message_text(text, reply_markup=kb)
    elif data.startswith("fb+") or data.startswith("fb-"):
        parts = data.split("|")
        fb = 1 if parts[0]=="fb+" else -1
        sid = int(parts[1])
        save_feedback(sid, q.from_user.id, fb)
        await q.edit_message_text("Спасибо за обратную связь!")

# Scanner job
async def scan_and_send(context: ContextTypes.DEFAULT_TYPE):
    logger.info("Scan cycle start")
    best = None
    for p in PAIRS:
        df = await yf_safe(p, period="2d", interval="1m")
        if df.empty: continue
        direction, confidence, notes = score(df)
        if direction and confidence > 40:
            expiration = max(1, min(15, int(1 + (confidence/100)*14)))
            candidate = {"pair": p, "direction": direction, "confidence": confidence, "expiration": expiration, "notes": notes}
            if not best or candidate["confidence"] > best["confidence"]:
                best = candidate
    if not best:
        logger.info("No best signal this cycle")
        return
    subs = get_users()
    if not subs:
        logger.info("No subscribers")
        return
    text = (f"Сигнал: {best['pair'].replace('=X','')}\nНаправление: {best['direction']}\n"
            f"Экспирация: {best['expiration']} мин\nУверенность: {best['confidence']}%\nПричины: {', '.join(best['notes'][:5])}")
    sid = save_signal(best['pair'], best['direction'], best['expiration'], best['confidence'], subs)
    for chat_id in subs:
        kb = InlineKeyboardMarkup([[InlineKeyboardButton("✅ +", callback_data=f"fb+|{sid}"),
                                    InlineKeyboardButton("❌ -", callback_data=f"fb-|{sid}")]])
        try:
            await context.bot.send_message(chat_id, text, reply_markup=kb)
        except Exception as e:
            logger.warning("failed send to %s: %s", chat_id, e)

# Build and run
def build_app():
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("stop", stop))
    app.add_handler(CommandHandler("history", history))
    app.add_handler(CallbackQueryHandler(callback_q))
    # job queue
    app.job_queue.run_repeating(scan_and_send, interval=60, first=10)
    return app

if __name__ == "__main__":
    app = build_app()
    logger.info("Setting webhook to %s", FULL_WEBHOOK)
    app.run_webhook(listen="0.0.0.0", port=PORT, url_path=BOT_TOKEN, webhook_url=FULL_WEBHOOK)
