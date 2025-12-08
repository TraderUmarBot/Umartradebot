# main.py — PTB 20+ (Flask webhook) with your EXCHANGE_PAIRS, expiries, TA and News
import os
import time
import logging
import asyncio
from typing import Tuple, Dict, Any, List, Optional

from flask import Flask, request
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler

import yfinance as yf
import pandas as pd
import numpy as np

# ------------------------
# CONFIG (your lists)
# ------------------------
EXCHANGE_PAIRS = [
    "EUR/USD","GBP/USD","USD/JPY","AUD/USD","USD/CAD","USD/CHF",
    "EUR/JPY","GBP/JPY","AUD/JPY","EUR/GBP","EUR/AUD","GBP/AUD",
    "CAD/JPY","CHF/JPY","EUR/CAD","GBP/CAD","AUD/CAD","AUD/CHF","CAD/CHF"
]

EXCHANGE_ALLOWED = ["1m", "3m", "5m", "10m"]
INTERVAL_MAP = {"3m": "2m", "10m": "5m"}  # yfinance mapping
PAIRS_PER_PAGE = 6
LOOKBACK = 120

BOT_TOKEN = os.getenv("BOT_TOKEN")
WEBHOOK_HOST = os.getenv("WEBHOOK_URL")  # e.g. https://your-app.onrender.com
PORT = int(os.getenv("PORT", "10000"))
if not BOT_TOKEN or not WEBHOOK_HOST:
    raise SystemExit("Set BOT_TOKEN and WEBHOOK_URL env vars")

WEBHOOK_PATH = f"/webhook/{BOT_TOKEN}"
WEBHOOK_URL = f"{WEBHOOK_HOST.rstrip('/')}{WEBHOOK_PATH}"

# ------------------------
# Logging
# ------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------
# yfinance cache + helper (async-friendly)
# ------------------------
YF_CACHE: Dict[Tuple[str, str], Tuple[float, pd.DataFrame]] = {}
YF_CACHE_TTL = 12.0  # seconds

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
            # run blocking yfinance in thread
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
            last_exc = RuntimeError("Empty DataFrame")
        except Exception as e:
            last_exc = e
            logger.warning("yfinance error %s %s attempt %d: %s", ticker, interval, attempt, repr(e))
        await asyncio.sleep(backoff * attempt)
    logger.error("yfinance failed for %s %s: %s", ticker, interval, repr(last_exc))
    return pd.DataFrame()

# ------------------------
# Indicators
# ------------------------
def ta_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period, min_periods=1).mean()
    avg_loss = loss.rolling(period, min_periods=1).mean().replace(0, 1e-9)
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def ta_macd(series: pd.Series, fast=12, slow=26, signal=9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    sig = macd.ewm(span=signal, adjust=False).mean()
    return macd, sig

def ta_sma(series: pd.Series, period=50):
    return series.rolling(period, min_periods=1).mean()

def ta_ema(series: pd.Series, period=20):
    return series.ewm(span=period, adjust=False).mean()

def ta_bollinger(series: pd.Series, period=20, n_std=2.0):
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

def ta_atr(df: pd.DataFrame, period=14):
    tr1 = df['High'] - df['Low']
    tr2 = (df['High'] - df['Close'].shift()).abs()
    tr3 = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()

def ta_adx(df: pd.DataFrame, period=14):
    up_move = df['High'].diff()
    down_move = -df['Low'].diff()
    plus_dm = ((up_move > down_move) & (up_move > 0)) * up_move
    minus_dm = ((down_move > up_move) & (down_move > 0)) * down_move
    tr1 = df['High'] - df['Low']
    tr2 = (df['High'] - df['Close'].shift()).abs()
    tr3 = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_val = tr.rolling(period, min_periods=1).mean().replace(0, 1e-9)
    plus_di = 100 * (plus_dm.rolling(period, min_periods=1).sum() / atr_val)
    minus_di = 100 * (minus_dm.rolling(period, min_periods=1).sum() / atr_val)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-9)
    return dx.rolling(period, min_periods=1).mean()

# ------------------------
# Signal with strength (weights tuned)
# ------------------------
def compute_signal_with_strength(df: pd.DataFrame) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    notes: List[str] = []
    if df is None or df.empty or len(df) < 10:
        return None, ["Недостаточно данных"]

    df = df.tail(LOOKBACK).copy()
    df['rsi'] = ta_rsi(df['Close'])
    df['macd'], df['macd_sig'] = ta_macd(df['Close'])
    df['sma50'] = ta_sma(df['Close'], 50)
    df['ema20'] = ta_ema(df['Close'], 20)
    df['bb_ma'], df['bb_up'], df['bb_low'] = ta_bollinger(df['Close'], 20)
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
        logger.exception("Error scalars: %s", e)
        return None, ["Ошибка извлечения значений"]

    # weights
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

    cand_bias = 0.3 if last['Close'] > last['Open'] else -0.3
    if cand_bias > 0: buy += cand_bias
    else: sell += abs(cand_bias)

    if buy > sell:
        direction = "Вверх"; raw = buy - sell
    elif sell > buy:
        direction = "Вниз"; raw = sell - buy
    else:
        direction = "Нет сигнала"; raw = 0.0

    max_plausible = (W_RSI + W_MACD + W_SMA + W_EMA + W_BB + W_STO + W_ATR + W_ADX + 1.0)
    strength = int(min(99, max(5, (raw / max_plausible) * 100)))

    result = {
        "signal": direction,
        "strength": strength,
        "values": {
            "rsi": rsi_v, "macd": macd_v, "macd_sig": macd_sig_v,
            "sma50": sma_v, "ema20": ema_v, "bb_up": bb_up_v, "bb_low": bb_low_v,
            "stoch_k": st_k, "stoch_d": st_d, "atr": atr_v, "adx": adx_v, "close": close_v
        }
    }
    return result, notes

# ------------------------
# UI helpers
# ------------------------
def escape_md(text: str) -> str:
    import re
    return re.sub(r'([_*\[\]()~`>#+\-=|{}.!])', r'\\\1', str(text))

def get_pairs_page(pairs: List[str], page: int) -> List[str]:
    start = page * PAIRS_PER_PAGE
    return pairs[start:start + PAIRS_PER_PAGE]

def total_pages(pairs: List[str]) -> int:
    return (len(pairs) - 1) // PAIRS_PER_PAGE

# ------------------------
# Telegram handlers
# ------------------------
app = ApplicationBuilder().token(BOT_TOKEN).concurrent_updates(True).build()

async def start_cmd(update: Update, context):
    kb = [
        [InlineKeyboardButton("📈 Биржевой рынок", callback_data="market|exchange")],
        [InlineKeyboardButton("📰 Новости (умный сигнал)", callback_data="news|")],
        [InlineKeyboardButton("📜 История", callback_data="history|")]
    ]
    if update.message:
        await update.message.reply_text("👋 Привет! Выберите:", reply_markup=InlineKeyboardMarkup(kb))
    else:
        q = update.callback_query
        await q.answer()
        await q.edit_message_text("👋 Привет! Выберите:", reply_markup=InlineKeyboardMarkup(kb))

async def choose_pair(update: Update, context, page: int = 0):
    q = update.callback_query
    await q.answer()
    page_pairs = get_pairs_page(EXCHANGE_PAIRS, page)
    kb = [[InlineKeyboardButton(p, callback_data=f"pair|{p}")] for p in page_pairs]
    nav = []
    if page > 0: nav.append(InlineKeyboardButton("⬅ Назад", callback_data=f"choose|{page-1}"))
    if page < total_pages(EXCHANGE_PAIRS): nav.append(InlineKeyboardButton("Вперёд ➡", callback_data=f"choose|{page+1}"))
    if nav: kb.append(nav)
    kb.append([InlineKeyboardButton("⬅ Главное меню", callback_data="back|")])
    await q.edit_message_text("⚡ Выберите валютную пару:", reply_markup=InlineKeyboardMarkup(kb))

async def choose_expiry(update: Update, context, pair: str):
    q = update.callback_query
    await q.answer()
    kb = [[InlineKeyboardButton(tf, callback_data=f"analyze|{pair}|{tf}")] for tf in EXCHANGE_ALLOWED]
    kb.append([InlineKeyboardButton("⬅ Назад", callback_data="market|exchange")])
    kb.append([InlineKeyboardButton("⬅ Главное меню", callback_data="back|")])
    await q.edit_message_text(f"Пара: {pair}\nВыберите экспирацию:", reply_markup=InlineKeyboardMarkup(kb))

async def analyze_and_show(update: Update, context, pair: str, expiry: str):
    q = update.callback_query
    await q.answer()
    uid = q.from_user.id
    try:
        await q.edit_message_text("🔍 Анализирую, подгружаю данные...")
    except Exception:
        pass

    ticker = pair.replace("/", "") + "=X"
    df = await yf_download_cached(ticker, expiry)
    if df.empty:
        await q.edit_message_text(f"❗ Нет данных для {pair} ({expiry})")
        return

    result, notes = compute_signal_with_strength(df)
    if not result:
        await q.edit_message_text("❗ Не удалось посчитать сигнал: " + " | ".join(notes))
        return

    # save last signal to user_data
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

async def save_result(update: Update, context):
    q = update.callback_query
    await q.answer()
    uid = q.from_user.id
    choice = q.data.split("|")[1]
    last = context.user_data.get("last_signal")
    if not last:
        await q.edit_message_text("Нет сигнала для сохранения.")
        return
    pair = last["pair"]; expiry = last["expiry"]; res = last["result"]
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    entry = f"{ts} — {pair} ({expiry}) — {'Плюс' if choice=='plus' else 'Минус'} — {res['strength']}%"
    # save to in-memory history per user
    history = context.user_data.setdefault("history", [])
    history.append(entry)
    await q.edit_message_text(f"✅ Сохранено: *{escape_md(entry)}*", parse_mode="MarkdownV2")

async def show_history(update: Update, context):
    q = update.callback_query
    await q.answer()
    uid = q.from_user.id
    history = context.user_data.get("history", [])
    if not history:
        await q.edit_message_text("📭 История пустая.")
        return
    text = "📜 *История:*\n\n" + "\n".join([escape_md(x) for x in history[-50:]])
    await q.edit_message_text(text, parse_mode="MarkdownV2")

# news: scan EXCHANGE_PAIRS and pick best by strength
async def news_handler(update: Update, context):
    q = update.callback_query
    await q.answer()
    await q.edit_message_text("🔎 Ищу лучший сигнал по всем парам (это может занять пару секунд)...")
    best = None; best_strength = -1; best_pair = None; best_expiry = None; best_notes = None
    # analyze with expiry candidates to pick recommended expiry too
    for pair in EXCHANGE_PAIRS:
        ticker = pair.replace("/", "") + "=X"
        for expiry in EXCHANGE_ALLOWED:
            df = await yf_download_cached(ticker, expiry)
            if df.empty: continue
            res, notes = compute_signal_with_strength(df)
            if not res: continue
            if res['strength'] > best_strength:
                best_strength = res['strength']
                best = res
                best_pair = pair
                best_expiry = expiry
                best_notes = notes
    if not best:
        await q.edit_message_text("Не найден подходящий сигнал для новостей сейчас.")
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
    # save last news to user_data
    context.user_data["last_news"] = {"pair": best_pair, "expiry": best_expiry, "result": best}
    kb = [
        [InlineKeyboardButton("🟢 Сохранить", callback_data="newsres|plus"),
         InlineKeyboardButton("🔴 Отклонить", callback_data="newsres|minus")],
        [InlineKeyboardButton("⬅ Главное меню", callback_data="back|")]
    ]
    await q.edit_message_text(text, parse_mode="MarkdownV2", reply_markup=InlineKeyboardMarkup(kb))

async def save_news_result(update: Update, context):
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
async def callbacks(update: Update, context):
    q = update.callback_query
    await q.answer()
    data = q.data or ""
    try:
        parts = data.split("|")
        cmd = parts[0]
        if cmd == "market":
            await choose_pair(update, context, page=0)
        elif cmd == "choose":
            page = int(parts[1]); await choose_pair(update, context, page=page)
        elif cmd == "pair":
            pair = parts[1]; await choose_expiry(update, context, pair)
        elif cmd == "analyze":
            pair, expiry = parts[1], parts[2]; await analyze_and_show(update, context, pair, expiry)
        elif cmd == "result":
            await save_result(update, context)
        elif cmd == "history":
            await show_history(update, context)
        elif cmd == "back":
            await start_cmd(update, context)
        elif cmd == "news":
            await news_handler(update, context)
        elif cmd == "newsres":
            await save_news_result(update, context)
        else:
            await q.answer("Неизвестная команда")
    except Exception as e:
        logger.exception("Callback error: %s", e)
        try:
            await q.edit_message_text("Произошла ошибка, попробуйте ещё раз.")
        except Exception:
            pass

# register handlers
app.add_handler(CommandHandler("start", start_cmd))
app.add_handler(CallbackQueryHandler(callbacks))

# ------------------------
# Flask webhook
# ------------------------
flask_app = Flask(__name__)

@flask_app.route("/", methods=["GET"])
def home():
    return "Bot is running", 200

@flask_app.route(WEBHOOK_PATH, methods=["POST"])
def webhook_handler():
    data = request.get_json(force=True)
    update = Update.de_json(data, app.bot)
    app.update_queue.put_nowait(update)
    return "OK", 200

# set webhook task
async def set_webhook():
    logger.info("Setting webhook to %s", WEBHOOK_URL)
    try:
        await app.bot.delete_webhook()
    except Exception:
        pass
    await app.bot.set_webhook(WEBHOOK_URL)
    logger.info("Webhook set")

# run
def main():
    logger.info("Starting application...")
    app.run_async()
    app.create_task(set_webhook())
    flask_app.run(host="0.0.0.0", port=PORT)

if __name__ == "__main__":
    main()
