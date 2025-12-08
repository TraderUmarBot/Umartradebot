# main.py
# PTB 20+ — Webhook mode (no Flask). Replace BOT_TOKEN and WEBHOOK_URL via env.
import os
import time
import logging
import asyncio
from typing import Tuple, Dict, Any, List, Optional

import pandas as pd
import numpy as np
import yfinance as yf

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
)

# ---------------- CONFIG ----------------
BOT_TOKEN = os.getenv("BOT_TOKEN")
WEBHOOK_URL = os.getenv("WEBHOOK_URL")  # e.g. https://your-app.onrender.com
PORT = int(os.getenv("PORT", "10000"))

if not BOT_TOKEN or not WEBHOOK_URL:
    raise SystemExit("Set BOT_TOKEN and WEBHOOK_URL environment variables")

WEBHOOK_PATH = f"/webhook/{BOT_TOKEN}"
FULL_WEBHOOK = f"{WEBHOOK_URL.rstrip('/')}{WEBHOOK_PATH}"

# your lists (as requested)
EXCHANGE_PAIRS = [
    "EUR/USD","GBP/USD","USD/JPY","AUD/USD","USD/CAD","USD/CHF",
    "EUR/JPY","GBP/JPY","AUD/JPY","EUR/GBP","EUR/AUD","GBP/AUD",
    "CAD/JPY","CHF/JPY","EUR/CAD","GBP/CAD","AUD/CAD","AUD/CHF","CAD/CHF"
]
EXCHANGE_ALLOWED = ["1m", "3m", "5m", "10m"]
INTERVAL_MAP = {"3m": "2m", "10m": "5m"}  # yfinance mapping
PAIRS_PER_PAGE = 6
LOOKBACK = 120

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- In-memory storage ----------------
user_state: Dict[int, Dict[str, Any]] = {}
trade_history: Dict[int, List[str]] = {}

# ---------------- yfinance cache + async download ----------------
YF_CACHE: Dict[Tuple[str, str], Tuple[float, pd.DataFrame]] = {}
YF_CACHE_TTL = 12.0

async def yf_download_cached(ticker: str, interval: str, retries: int = 3, backoff: float = 1.0) -> pd.DataFrame:
    key = (ticker, interval)
    now = time.time()
    cached = YF_CACHE.get(key)
    if cached and now - cached[0] < YF_CACHE_TTL:
        return cached[1].copy()
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            yf_int = INTERVAL_MAP.get(interval, interval)
            df = await asyncio.to_thread(
                yf.download,
                ticker,
                period="7d",
                interval=yf_int,
                progress=False,
                threads=False,
                auto_adjust=True,
            )
            if isinstance(df, pd.DataFrame) and not df.empty:
                YF_CACHE[key] = (time.time(), df.copy())
                return df
            last_exc = RuntimeError("Empty DataFrame from yfinance")
            logger.warning("yfinance empty for %s %s attempt %d", ticker, interval, attempt)
        except Exception as e:
            last_exc = e
            logger.warning("yfinance error for %s %s attempt %d: %s", ticker, interval, attempt, repr(e))
        await asyncio.sleep(backoff * attempt)
    logger.error("yfinance failed for %s %s: %s", ticker, interval, repr(last_exc))
    return pd.DataFrame()

# ---------------- Indicators ----------------
def ta_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period, min_periods=1).mean()
    avg_loss = loss.rolling(period, min_periods=1).mean().replace(0, 1e-9)
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def ta_sma(series: pd.Series, period: int = 50) -> pd.Series:
    return series.rolling(period, min_periods=1).mean()

def ta_ema(series: pd.Series, period: int = 20) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def ta_macd(series: pd.Series, fast=12, slow=26, signal=9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    sig = macd.ewm(span=signal, adjust=False).mean()
    return macd, sig

def ta_bollinger(series: pd.Series, period: int = 20, n_std: float = 2.0):
    ma = series.rolling(period, min_periods=1).mean()
    std = series.rolling(period, min_periods=1).std().fillna(0)
    upper = ma + n_std * std
    lower = ma - n_std * std
    return ma, upper, lower

def ta_stochastic(df: pd.DataFrame, k_period=14, d_period=3):
    low_min = df['Low'].rolling(k_period, min_periods=1).min()
    high_max = df['High'].rolling(k_period, min_periods=1).max()
    k = 100 * (df['Close'] - low_min) / (high_max - low_min).replace(0, 1e-9)
    d = k.rolling(d_period, min_periods=1).mean()
    return k, d

def ta_atr(df: pd.DataFrame, period: int = 14):
    tr1 = df['High'] - df['Low']
    tr2 = (df['High'] - df['Close'].shift()).abs()
    tr3 = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()

def ta_adx(df: pd.DataFrame, period: int = 14):
    up_move = df['High'].diff()
    down_move = -df['Low'].diff()
    plus_dm = ((up_move > down_move) & (up_move > 0)) * up_move
    minus_dm = ((down_move > up_move) & (down_move > 0)) * down_move
    tr1 = df['High'] - df['Low']
    tr2 = (df['High'] - df['Close'].shift()).abs()
    tr3 = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_ = tr.rolling(period, min_periods=1).mean().replace(0, 1e-9)
    plus_di = 100 * (plus_dm.rolling(period, min_periods=1).sum() / atr_)
    minus_di = 100 * (minus_dm.rolling(period, min_periods=1).sum() / atr_)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-9)
    return dx.rolling(period, min_periods=1).mean()

# ---------------- Signal computation ----------------
def compute_signal_with_strength(df: pd.DataFrame) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    notes: List[str] = []
    if df is None or df.empty or len(df) < 10:
        return None, ["Недостаточно данных"]

    df = df.tail(LOOKBACK).copy()
    df['rsi'] = ta_rsi(df['Close'])
    df['macd'], df['macd_sig'] = ta_macd(df['Close'])
    df['sma50'] = ta_sma(df['Close'], 50)
    df['ema20'] = ta_ema(df['Close'], 20)
    df['bb_ma'], df['bb_up'], df['bb_low'] = ta_bollinger(df['Close'])
    df['stoch_k'], df['stoch_d'] = ta_stochastic(df)
    df['atr'] = ta_atr(df)
    df['adx'] = ta_adx(df)

    last = df.iloc[-1].copy()
    try:
        rsi_v = float(last['rsi'])
        close_v = float(last['Close'])
        macd_v = float(last['macd'])
        macd_sig_v = float(last['macd_sig'])
        sma_v = float(last['sma50'])
        ema_v = float(last['ema20'])
        bb_up_v = float(last['bb_up'])
        bb_low_v = float(last['bb_low'])
        st_k = float(last['stoch_k'])
        st_d = float(last['stoch_d'])
        atr_v = float(last['atr'])
        adx_v = float(last['adx'])
    except Exception as e:
        logger.exception("Error extracting scalars: %s", e)
        return None, ["Ошибка извлечения значений"]

    # weights (tuneable)
    W_RSI = 2.5; W_MACD = 1.5; W_SMA = 1.0; W_EMA = 0.8
    W_BB = 1.2; W_STO = 0.9; W_ATR = 0.7; W_ADX = 1.0

    buy = 0.0; sell = 0.0

    if rsi_v < 30:
        buy += W_RSI; notes.append("RSI oversold")
    elif rsi_v > 70:
        sell += W_RSI; notes.append("RSI overbought")

    if macd_v > macd_sig_v:
        buy += W_MACD; notes.append("MACD bull")
    else:
        sell += W_MACD; notes.append("MACD bear")

    if close_v > sma_v:
        buy += W_SMA; notes.append("Above SMA50")
    else:
        sell += W_SMA; notes.append("Below SMA50")

    if close_v > ema_v:
        buy += W_EMA; notes.append("Above EMA20")
    else:
        sell += W_EMA; notes.append("Below EMA20")

    if close_v > bb_up_v:
        sell += W_BB; notes.append("Above BB upper")
    elif close_v < bb_low_v:
        buy += W_BB; notes.append("Below BB lower")

    if st_k < 20 and st_k > st_d:
        buy += W_STO; notes.append("Stochastic bullish")
    elif st_k > 80 and st_k < st_d:
        sell += W_STO; notes.append("Stochastic bearish")

    atr_med = df['atr'].rolling(20, min_periods=1).median().iloc[-1]
    if atr_v > atr_med:
        if buy > sell:
            buy += W_ATR; notes.append("ATR elevated (boost buy)")
        elif sell > buy:
            sell += W_ATR; notes.append("ATR elevated (boost sell)")

    if adx_v > 25:
        if buy > sell:
            buy += W_ADX; notes.append(f"ADX {adx_v:.1f} (trend strengthens buy)")
        elif sell > buy:
            sell += W_ADX; notes.append(f"ADX {adx_v:.1f} (trend strengthens sell)")
    else:
        notes.append(f"ADX low {adx_v:.1f}")

    # small candle bias
    candle_bias = 0.3 if last['Close'] > last['Open'] else -0.3
    if candle_bias > 0:
        buy += candle_bias
    else:
        sell += abs(candle_bias)

    if buy > sell:
        direction = "Вверх"; raw = buy - sell
    elif sell > buy:
        direction = "Вниз"; raw = sell - buy
    else:
        direction = "Нет сигнала"; raw = 0.0

    max_plausible = (W_RSI + W_MACD + W_SMA + W_EMA + W_BB + W_STO + W_ATR + W_ADX + 1.0)
    strength = int(min(99, max(5, (raw / max_plausible) * 100)))

    values = {
        "rsi": rsi_v, "macd": macd_v, "macd_sig": macd_sig_v,
        "sma50": sma_v, "ema20": ema_v, "bb_up": bb_up_v, "bb_low": bb_low_v,
        "stoch_k": st_k, "stoch_d": st_d, "atr": atr_v, "adx": adx_v, "close": close_v
    }

    return {"signal": direction, "strength": strength, "values": values}, notes

# ---------------- UI helpers ----------------
def escape_md(text: str) -> str:
    import re
    return re.sub(r'([_*\[\]()~`>#+\-=|{}.!])', r'\\\1', str(text))

def get_pairs_page(pairs, page):
    start = page * PAIRS_PER_PAGE
    return pairs[start:start + PAIRS_PER_PAGE]

def total_pages(pairs):
    return (len(pairs) - 1) // PAIRS_PER_PAGE

# ---------------- Handlers & Menu ----------------
async def show_main_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    kb = [
        [InlineKeyboardButton("📈 Биржевой рынок", callback_data="market|exchange")],
        [InlineKeyboardButton("📰 Новости (умный сигнал)", callback_data="news|")],
        [InlineKeyboardButton("📜 История сделок", callback_data="history|")]
    ]
    if update.message:
        await update.message.reply_text("👋 Привет! Выберите раздел:", reply_markup=InlineKeyboardMarkup(kb))
    else:
        q = update.callback_query
        await q.answer()
        await q.edit_message_text("👋 Привет! Выберите раздел:", reply_markup=InlineKeyboardMarkup(kb))

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await show_main_menu(update, context)

async def choose_pair_handler(update: Update, context: ContextTypes.DEFAULT_TYPE, page: int = 0):
    q = update.callback_query
    await q.answer()
    page_pairs = get_pairs_page(EXCHANGE_PAIRS, page)
    kb = [[InlineKeyboardButton(p, callback_data=f"pair|{p}")] for p in page_pairs]
    nav = []
    if page > 0:
        nav.append(InlineKeyboardButton("⬅ Назад", callback_data=f"choose|{page-1}"))
    if page < total_pages(EXCHANGE_PAIRS):
        nav.append(InlineKeyboardButton("Вперёд ➡", callback_data=f"choose|{page+1}"))
    if nav:
        kb.append(nav)
    kb.append([InlineKeyboardButton("⬅ Главное меню", callback_data="back|")])
    await q.edit_message_text("⚡ Выберите валютную пару:", reply_markup=InlineKeyboardMarkup(kb))

async def choose_expiry_handler(update: Update, context: ContextTypes.DEFAULT_TYPE, pair: str):
    q = update.callback_query
    await q.answer()
    kb = [[InlineKeyboardButton(tf, callback_data=f"analyze|{pair}|{tf}")] for tf in EXCHANGE_ALLOWED]
    kb.append([InlineKeyboardButton("⬅ Назад", callback_data="market|exchange")])
    kb.append([InlineKeyboardButton("⬅ Главное меню", callback_data="back|")])
    await q.edit_message_text(f"Пара: {pair}\nВыберите экспирацию:", reply_markup=InlineKeyboardMarkup(kb))

async def analyze_and_show(update: Update, context: ContextTypes.DEFAULT_TYPE, pair: str, expiry: str):
    q = update.callback_query
    await q.answer()
    try:
        await q.edit_message_text("🔍 Анализирую, подгружаю данные...")
    except Exception:
        pass
    ticker = pair.replace("/", "") + "=X"
    df = await yf_download_cached(ticker, expiry)
    if df is None or df.empty:
        await q.edit_message_text(f"❗ Нет данных для {pair} ({expiry})")
        return
    result, notes = compute_signal_with_strength(df)
    if not result:
        await q.edit_message_text("❗ Не удалось посчитать сигнал: " + " | ".join(notes))
        return
    context.user_data["last_signal"] = {"pair": pair, "expiry": expiry, "result": result}
    v = result["values"]
    notes_text = "\n".join([f"• {n}" for n in notes[:8]])
    text = (
        f"📊 *Сигнал*: *{escape_md(result['signal'])}*\n"
        f"Пара: *{escape_md(pair)}*\n"
        f"Экспирация: *{escape_md(expiry)}*\n"
        f"Сила сигнала: *{result['strength']} %*\n\n"
        f"— Технические значения —\n"
        f"RSI: {v['rsi']:.2f}\n"
        f"MACD: {v['macd']:.5f}  MACD_sig: {v['macd_sig']:.5f}\n"
        f"SMA50: {v['sma50']:.5f}  EMA20: {v['ema20']:.5f}\n"
        f"BB_up: {v['bb_up']:.5f}  BB_low: {v['bb_low']:.5f}\n"
        f"ATR: {v['atr']:.6f}  ADX: {v['adx']:.2f}\n\n"
        f"Причины:\n{escape_md(notes_text)}"
    )
    kb = [
        [InlineKeyboardButton("🟢 Плюс (сделка)", callback_data="result|plus"),
         InlineKeyboardButton("🔴 Минус (сделка)", callback_data="result|minus")],
        [InlineKeyboardButton("⬅ Главное меню", callback_data="back|")]
    ]
    await q.edit_message_text(text, parse_mode="MarkdownV2", reply_markup=InlineKeyboardMarkup(kb))

async def save_result_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    choice = q.data.split("|")[1]
    last = context.user_data.get("last_signal")
    if not last:
        await q.edit_message_text("Нет сигнала для сохранения.")
        return
    pair = last["pair"]; expiry = last["expiry"]; res = last["result"]
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    entry = f"{ts} — {pair} ({expiry}) — {'Плюс' if choice=='plus' else 'Минус'} — {res['strength']}%"
    history = context.user_data.setdefault("history", [])
    history.append(entry)
    await q.edit_message_text(f"✅ Сохранено: *{escape_md(entry)}*", parse_mode="MarkdownV2")

async def history_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    history = context.user_data.get("history", [])
    if not history:
        await q.edit_message_text("📭 История пустая.")
        return
    text = "📜 *История:*\n\n" + "\n".join([escape_md(x) for x in history[-50:]])
    await q.edit_message_text(text, parse_mode="MarkdownV2")

# news: scan and pick best
async def news_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    await q.edit_message_text("🔎 Сканирую пары для умного сигнала (несколько секунд)...")
    best = None; best_strength = -1; best_pair = None; best_expiry = None; best_notes = None
    for pair in EXCHANGE_PAIRS:
        ticker = pair.replace("/", "") + "=X"
        for expiry in EXCHANGE_ALLOWED:
            df = await yf_download_cached(ticker, expiry)
            if df.empty:
                continue
            res_notes = compute_signal_with_strength(df)
            if res_notes is None:
                continue
            res, notes = res_notes
            if res and res.get("strength", 0) > best_strength:
                best_strength = res["strength"]
                best = res; best_pair = pair; best_expiry = expiry; best_notes = notes
    if not best:
        await q.edit_message_text("Не найден подходящий сигнал.")
        return
    notes_text = "\n".join([f"• {n}" for n in (best_notes or [])[:6]])
    text = (
        f"📰 *Новостной сигнал*\n\n"
        f"Пара: *{escape_md(best_pair)}*\n"
        f"Направление: *{escape_md(best['signal'])}*\n"
        f"Рекомендуемая экспирация: *{escape_md(best_expiry)}*\n"
        f"Сила: *{best['strength']} %*\n\n"
        f"Причины:\n{escape_md(notes_text)}"
    )
    context.user_data["last_news"] = {"pair": best_pair, "expiry": best_expiry, "result": best}
    kb = [
        [InlineKeyboardButton("🟢 Сохранить", callback_data="newsres|plus"),
         InlineKeyboardButton("🔴 Отклонить", callback_data="newsres|minus")],
        [InlineKeyboardButton("⬅ Главное меню", callback_data="back|")]
    ]
    await q.edit_message_text(text, parse_mode="MarkdownV2", reply_markup=InlineKeyboardMarkup(kb))

async def save_news_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    choice = q.data.split("|")[1]
    last = context.user_data.get("last_news")
    if not last:
        await q.edit_message_text("Нет новостного сигнала для сохранения.")
        return
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    entry = f"{ts} — {last['pair']} ({last['expiry']}) — {'Плюс' if choice=='plus' else 'Минус'} — {last['result']['strength']}%"
    history = context.user_data.setdefault("history", [])
    history.append(entry)
    await q.edit_message_text(f"✅ Сохранено: *{escape_md(entry)}*", parse_mode="MarkdownV2")

# router
async def callbacks(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    data = q.data or ""
    try:
        parts = data.split("|")
        cmd = parts[0]
        if cmd == "market":
            await choose_pair_handler(update, context, page=0)
        elif cmd == "choose":
            page = int(parts[1]); await choose_pair_handler(update, context, page=page)
        elif cmd == "pair":
            pair = parts[1]; await choose_expiry_handler(update, context, pair)
        elif cmd == "analyze":
            pair, expiry = parts[1], parts[2]; await analyze_and_show(update, context, pair, expiry)
        elif cmd == "result":
            await save_result_handler(update, context)
        elif cmd == "history":
            await history_handler(update, context)
        elif cmd == "back":
            await show_main_menu(update, context)
        elif cmd == "news":
            await news_handler(update, context)
        elif cmd == "newsres":
            await save_news_handler(update, context)
        else:
            await q.answer("Неизвестная команда")
    except Exception as e:
        logger.exception("Callback error: %s", e)
        try:
            await q.edit_message_text("Произошла ошибка, попробуйте ещё раз.")
        except Exception:
            pass

# ---------------- App bootstrap ----------------
app = ApplicationBuilder().token(BOT_TOKEN).concurrent_updates(True).build()
app.add_handler(CommandHandler("start", start_cmd))
app.add_handler(CallbackQueryHandler(callbacks))

# ---------------- Run webhook (single loop) ----------------
if __name__ == "__main__":
    logger.info("Setting webhook to %s", FULL_WEBHOOK)
    # run_webhook blocks the process, manages event loop internally
    app.run_webhook(
        listen="0.0.0.0",
        port=PORT,
        url_path=BOT_TOKEN,
        webhook_url=FULL_WEBHOOK,
    )
