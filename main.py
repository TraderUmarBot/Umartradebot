# main.py — Переписанный, стабильный, с защитой от rate-limit и фиксами
import os
import logging
import asyncio
import time
from typing import Tuple, Optional, Dict, Any, List

import pandas as pd
import yfinance as yf

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
)

# ---------- Настройка логирования ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- Конфигурация ----------
EXCHANGE_PAIRS = [
    "EUR/USD","GBP/USD","USD/JPY","AUD/USD","USD/CAD","USD/CHF",
    "EUR/JPY","GBP/JPY","AUD/JPY","EUR/GBP","EUR/AUD","GBP/AUD",
    "CAD/JPY","CHF/JPY","EUR/CAD","GBP/CAD","AUD/CAD","AUD/CHF","CAD/CHF"
]

EXCHANGE_ALLOWED = ["1m", "3m", "5m", "10m"]
INTERVAL_MAP = {"3m": "2m", "10m": "5m"}  # YFinance mapping if нужно
PAIRS_PER_PAGE = 6
LOOKBACK = 120

# cache for yfinance results: { (ticker,interval) : (timestamp, dataframe) }
YF_CACHE: Dict[Tuple[str, str], Tuple[float, pd.DataFrame]] = {}
YF_CACHE_TTL = 10.0  # seconds — кеш держим небольшим, чтобы не сильно ждать, но избежать rate limit

# Simple in-memory storage (per-user)
user_state: Dict[int, Dict[str, Any]] = {}
trade_history: Dict[int, List[str]] = {}

# ---------- Technical indicators ----------
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

def candle_patterns_df(df: pd.DataFrame) -> List[str]:
    patterns: List[str] = []
    if df.empty or len(df) < 1:
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

# ---------- Resilient yfinance download (async-friendly) ----------
async def yf_download_cached(ticker: str, interval: str, retries: int = 3, backoff: float = 1.0) -> pd.DataFrame:
    key = (ticker, interval)
    now = time.time()
    # return cache if fresh
    cached = YF_CACHE.get(key)
    if cached and now - cached[0] < YF_CACHE_TTL:
        logger.debug("Using cached yfinance for %s %s", ticker, interval)
        return cached[1].copy()

    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            # yfinance is sync — run in thread to not block loop
            df = await asyncio.to_thread(
                yf.download,
                ticker,
                period="5d",
                interval=interval,
                progress=False,
                threads=False,
            )
            if isinstance(df, pd.DataFrame) and not df.empty:
                # store to cache
                YF_CACHE[key] = (time.time(), df.copy())
                return df
            # if empty, treat as temporary and retry
            last_exc = RuntimeError("Empty DataFrame from yfinance")
            logger.warning("yfinance empty on %s %s attempt %d", ticker, interval, attempt)
        except Exception as e:
            logger.warning("yfinance attempt %d for %s failed: %s", attempt, ticker, repr(e))
            last_exc = e
        await asyncio.sleep(backoff * attempt)
    # final: return empty df and log
    logger.error("yfinance failed for %s %s after %d attempts: %s", ticker, interval, retries, repr(last_exc))
    return pd.DataFrame()

# ---------- Analysis ----------
async def analyze_exchange(pair: str, expiry: str) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    ticker = pair.replace("/", "") + "=X"
    yfi_interval = yfinance_interval_for(expiry)
    notes: List[str] = []

    df = await yf_download_cached(ticker, yfi_interval)
    if df is None or df.empty or len(df) < 5:
        return None, [f"Нет данных для {ticker} (interval={yfi_interval})"]

    df = df.tail(LOOKBACK).copy()
    # compute indicators
    df["rsi"] = rsi(df["Close"])
    df["sma50"] = SMA(df["Close"], period=50)
    df["ema20"] = EMA(df["Close"], period=20)
    macd, macd_signal = MACD(df["Close"])
    df["macd"], df["macd_signal"] = macd, macd_signal

    notes += candle_patterns_df(df)

    last = df.iloc[-1].copy()  # ensure scalar access
    try:
        # get scalars safely
        rsi_v = float(last["rsi"])
        sma_v = float(last["sma50"])
        ema_v = float(last["ema20"])
        macd_v = float(last["macd"])
        macd_sig_v = float(last["macd_signal"])
        close_v = float(last["Close"])
    except Exception as e:
        logger.exception("Ошибка при извлечении значений из df: %s", e)
        return None, ["Ошибка при разборе индикаторов"]

    # Weighted scoring for a stronger signal
    buy_score = 0.0
    sell_score = 0.0

    # RSI
    if rsi_v < 30:
        buy_score += 2.0
        notes.append("RSI Oversold")
    elif rsi_v > 70:
        sell_score += 2.0
        notes.append("RSI Overbought")

    # MACD
    if macd_v > macd_sig_v:
        buy_score += 1.5
        notes.append("MACD Bull")
    else:
        sell_score += 1.5
        notes.append("MACD Bear")

    # Trend vs SMA/EMA
    if close_v > sma_v:
        buy_score += 1.0
        notes.append("Above SMA50")
    else:
        sell_score += 1.0
        notes.append("Below SMA50")

    if close_v > ema_v:
        buy_score += 1.0
        notes.append("Above EMA20")
    else:
        sell_score += 1.0
        notes.append("Below EMA20")

    # Candle bias (small boost)
    last_candle = "Bullish" if last["Close"] > last["Open"] else "Bearish"
    if last_candle == "Bullish":
        buy_score += 0.5
    else:
        sell_score += 0.5

    # Final decision
    if buy_score > sell_score:
        signal = "Вверх"
        score = buy_score
    elif sell_score > buy_score:
        signal = "Вниз"
        score = sell_score
    else:
        signal = "Нет сигнала"
        score = buy_score  # equals sell_score

    # Normalize confidence to 0..100 — tuned to give more weight for stronger score
    # maximum plausible score here ~ (2 + 1.5 + 1 + 1 + 0.5) = 6
    confidence = int(min(99, max(10, (score / 6.0) * 100)))

    result = {
        "signal": signal,
        "conf": confidence,
        "notes": notes,
        "values": {
            "rsi": rsi_v,
            "macd": macd_v,
            "macd_signal": macd_sig_v,
            "sma50": sma_v,
            "ema20": ema_v,
            "close": close_v,
        },
    }
    return result, notes

# ---------- UI / Handlers ----------
async def show_main_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("📈 Биржевой рынок", callback_data="market|exchange")],
        [InlineKeyboardButton("📜 История сделок", callback_data="history|")],
    ]
    if update.message:
        await update.message.reply_text("👋 Привет! Выберите рынок:", reply_markup=InlineKeyboardMarkup(keyboard))
    else:
        q = update.callback_query
        await q.answer()
        await q.edit_message_text("👋 Привет! Выберите рынок:", reply_markup=InlineKeyboardMarkup(keyboard))

async def choose_pair(update: Update, context: ContextTypes.DEFAULT_TYPE, page: int = 0):
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

async def choose_expiry(update: Update, context: ContextTypes.DEFAULT_TYPE, pair: str):
    q = update.callback_query
    await q.answer()
    keyboard = [[InlineKeyboardButton(tf, callback_data=f"analyze|{pair}|{tf}")] for tf in EXCHANGE_ALLOWED]
    keyboard.append([InlineKeyboardButton("⬅ Назад", callback_data="market|exchange")])
    keyboard.append([InlineKeyboardButton("⬅ Главное меню", callback_data="back|")])
    await q.edit_message_text(f"Пара: {escape_md(pair)}\nВыберите таймфрейм:", parse_mode=ParseMode.MARKDOWN_V2, reply_markup=InlineKeyboardMarkup(keyboard))

async def show_signal(update: Update, context: ContextTypes.DEFAULT_TYPE, pair: str, expiry: str):
    q = update.callback_query
    await q.answer()
    uid = q.from_user.id
    # Immediately save to user_state
    user_state[uid] = {"pair": pair, "expiry": expiry, "ts": time.time()}
    # show analyzing message
    try:
        await q.edit_message_text("🔍 Анализирую рынок...")
    except Exception:
        # if edit fails (message changed), just continue
        pass

    result, notes = await analyze_exchange(pair, expiry)
    if not result:
        msg = f"❗ Не удалось собрать сигнал.\nПричины: {' | '.join(notes)}"
        try:
            await q.edit_message_text(msg)
        except Exception:
            await q.answer(text=msg)
        return

    # Build message
    values = result.get("values", {})
    text = (
        f"📊 Сигнал: *{escape_md(result['signal'])}*\n"
        f"Пара: *{escape_md(pair)}*\n"
        f"Таймфрейм: *{escape_md(expiry)}*\n"
        f"Уверенность: *{result['conf']} %*\n\n"
        f"---- Технические значения ----\n"
        f"RSI: {values.get('rsi', '—'):.2f}\n"
        f"MACD: {values.get('macd', 0):.5f}  MACD_sig: {values.get('macd_signal', 0):.5f}\n"
        f"SMA50: {values.get('sma50', 0):.5f}\n"
        f"EMA20: {values.get('ema20', 0):.5f}\n\n"
        f"Notes: {' | '.join(result.get('notes', []))}"
    )
    keyboard = [
        [InlineKeyboardButton("🟢 Плюс (сделка)", callback_data="result|plus"),
         InlineKeyboardButton("🔴 Минус (сделка)", callback_data="result|minus")],
        [InlineKeyboardButton("📈 Новый сигнал", callback_data="market|exchange")],
        [InlineKeyboardButton("⬅ Главное меню", callback_data="back|")]
    ]
    try:
        await q.edit_message_text(text, parse_mode=ParseMode.MARKDOWN_V2, reply_markup=InlineKeyboardMarkup(keyboard))
    except Exception as e:
        logger.exception("Не удалось отправить сообщение с сигналом: %s", e)
        # fallback: send as simple text
        await q.edit_message_text("Сигнал готов. Откройте чат заново, чтобы увидеть детали.")

async def save_result(update: Update, context: ContextTypes.DEFAULT_TYPE, result_label: str):
    q = update.callback_query
    await q.answer()
    uid = q.from_user.id
    st = user_state.get(uid, {})
    pair = st.get("pair", "—")
    expiry = st.get("expiry", "—")
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    trade_history.setdefault(uid, []).append(f"{ts} — {pair} ({expiry}) — {result_label}")
    keyboard = [
        [InlineKeyboardButton("📈 Новый сигнал", callback_data="market|exchange")],
        [InlineKeyboardButton("📜 История", callback_data="history|")]
    ]
    await q.edit_message_text(f"✅ Записано: *{escape_md(result_label)}*", parse_mode=ParseMode.MARKDOWN_V2, reply_markup=InlineKeyboardMarkup(keyboard))

async def show_history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    uid = q.from_user.id
    entries = trade_history.get(uid, [])
    if not entries:
        await q.edit_message_text("📭 История пустая.")
        return
    text = "📜 *История (последние 50):*\n\n" + "\n".join([f"• {escape_md(t)}" for t in entries[-50:]])
    keyboard = [[InlineKeyboardButton("⬅ Главное меню", callback_data="back|")]]
    await q.edit_message_text(text, parse_mode=ParseMode.MARKDOWN_V2, reply_markup=InlineKeyboardMarkup(keyboard))

# ---------- Router ----------
async def callbacks(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    data = q.data or ""
    parts = data.split("|")
    cmd = parts[0] if parts else ""
    try:
        if cmd == "market":
            await choose_pair(update, context)
        elif cmd == "choose":
            page = int(parts[1])
            await choose_pair(update, context, page)
        elif cmd == "pair":
            pair = parts[1]
            await choose_expiry(update, context, pair)
        elif cmd == "analyze":
            pair, expiry = parts[1], parts[2]
            await show_signal(update, context, pair, expiry)
        elif cmd == "result":
            label = "Плюс" if parts[1] == "plus" else "Минус"
            await save_result(update, context, label)
        elif cmd == "history":
            await show_history(update, context)
        elif cmd == "back":
            await show_main_menu(update, context)
        else:
            await q.answer(text="Неизвестная команда")
    except Exception as e:
        logger.exception("Ошибка в callback: %s", e)
        try:
            await q.edit_message_text("Произошла ошибка, попробуйте ещё раз.")
        except Exception:
            pass

# ---------- Bot setup & run ----------
BOT_TOKEN = os.getenv("BOT_TOKEN")
WEBHOOK_URL = os.getenv("WEBHOOK_URL")
PORT = int(os.getenv("PORT", "10000"))

if not BOT_TOKEN or not WEBHOOK_URL:
    logger.error("Set BOT_TOKEN and WEBHOOK_URL env vars")
    raise SystemExit("Set BOT_TOKEN and WEBHOOK_URL env vars")

def build_application() -> "telegram.ext.Application":
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", show_main_menu))
    app.add_handler(CallbackQueryHandler(callbacks))
    return app

def main():
    app = build_application()
    # Using run_webhook (synchronous) avoids messing with multiple event loops
    webhook_path = f"/webhook/{BOT_TOKEN}"
    full_webhook = WEBHOOK_URL.rstrip("/") + webhook_path
    logger.info("Setting webhook to %s", full_webhook)
    # run_webhook will set webhook and start server in the same loop — stable and safe
    app.run_webhook(
        listen="0.0.0.0",
        port=PORT,
        url_path=webhook_path.lstrip("/"),
        webhook_url=full_webhook,
    )

if __name__ == "__main__":
    main()
