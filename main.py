#!/usr/bin/env python3
# main.py
"""
Trading Signals Telegram Bot — single-file version using TwelveData API
Features:
 - Pairs menu (user-provided pairs)
 - Timeframes: 1min, 5min, 15min
 - Indicators: RSI, MACD, EMA(9,21), SMA(50), Bollinger Bands, Volume SMA, EMA slope, simple engulfing candles
 - Signal logic: weighted voting, outputs BUY/SELL/NO TRADE with human-readable reason
 - Stores signals in SQLite and collects user feedback (plus/minus)
 - After user marks result, returns to main menu
 - Ready to run: set TELEGRAM_TOKEN and TWELVEDATA_API_KEY environment variables
"""
import os
import time
import logging
import requests
import sqlite3
from datetime import datetime
from typing import Tuple, Dict, Any, List

import pandas as pd
import numpy as np
import pandas_ta as ta

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, ParseMode
from telegram.ext import Updater, CommandHandler, CallbackQueryHandler, CallbackContext
from telegram.update import Update

# ---------------------------
# Configuration
# ---------------------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY", "")
DB_FILE = os.getenv("DB_FILE", "signals.db")

# Pairs list provided by you
PAIRS = [
    "EUR/USD","GBP/USD","USD/JPY","AUD/USD","USD/CAD","USD/CHF","EUR/JPY","GBP/JPY",
    "AUD/JPY","EUR/GBP","EUR/AUD","GBP/AUD","CAD/JPY","CHF/JPY","EUR/CAD","GBP/CAD",
    "AUD/CAD","AUD/CHF","CAD/CHF"
]

# Timeframe labels shown to user mapping to TwelveData intervals
TF_MAP = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min"
}

# How many candles to fetch (safety margin)
CANDLES = 300

# Indicator thresholds (tweakable)
RSI_BUY = 55
RSI_SELL = 45

# TwelveData endpoints
TD_BASE = "https://api.twelvedata.com/time_series"

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("trading_bot")

# ---------------------------
# DB (sqlite) storage
# ---------------------------
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT,
    symbol TEXT,
    timeframe TEXT,
    signal TEXT,
    reason TEXT,
    result TEXT
);
"""

class Storage:
    def __init__(self, db_file=DB_FILE):
        self.db_file = db_file
        self._init_db()

    def _get_conn(self):
        return sqlite3.connect(self.db_file, check_same_thread=False)

    def _init_db(self):
        conn = self._get_conn()
        c = conn.cursor()
        c.execute(CREATE_TABLE_SQL)
        conn.commit()
        conn.close()

    def add_signal(self, symbol: str, timeframe: str, signal: str, reason: str) -> int:
        conn = self._get_conn()
        c = conn.cursor()
        ts = datetime.utcnow().isoformat()
        c.execute("INSERT INTO signals (ts, symbol, timeframe, signal, reason, result) VALUES (?, ?, ?, ?, ?, ?)",
                  (ts, symbol, timeframe, signal, reason, None))
        conn.commit()
        sid = c.lastrowid
        conn.close()
        return sid

    def set_result(self, signal_id: int, result: str):
        conn = self._get_conn()
        c = conn.cursor()
        c.execute("UPDATE signals SET result = ? WHERE id = ?", (result, signal_id))
        conn.commit()
        conn.close()

    def get_stats(self) -> Dict[str, Any]:
        conn = self._get_conn()
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM signals")
        total = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM signals WHERE result = 'plus'")
        plus = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM signals WHERE result = 'minus'")
        minus = c.fetchone()[0]
        accuracy = round((plus / total * 100), 2) if total > 0 else 0.0
        c.execute("SELECT id, ts, symbol, timeframe, signal, reason, result FROM signals ORDER BY id DESC LIMIT 10")
        rows = c.fetchall()
        last10 = [dict(zip(["id","ts","symbol","timeframe","signal","reason","result"], row)) for row in rows]
        conn.close()
        return {"total": total, "plus": plus, "minus": minus, "accuracy": accuracy, "last10": last10}

storage = Storage()

# ---------------------------
# UI helpers (keyboards, ascii logo)
# ---------------------------
ASCII_LOGO = r"""
██████╗ ██████╗  ██████╗██╗  ██╗
██╔══██╗██╔══██╗██╔════╝██║ ██╔╝
██████╔╝██████╔╝██║     █████╔╝ 
██╔═══╝ ██╔══██╗██║     ██╔═██╗ 
██║     ██║  ██║╚██████╗██║  ██╗
╚═╝     ╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝
Trading Signals Bot — v1.0
"""

def main_menu_keyboard(pairs: List[str]) -> InlineKeyboardMarkup:
    keyboard = []
    # Make 2-column layout for compactness
    row = []
    for idx, p in enumerate(pairs, start=1):
        row.append(InlineKeyboardButton(p, callback_data=f"pair|{p}"))
        if len(row) == 2:
            keyboard.append(row)
            row = []
    if row:
        keyboard.append(row)
    keyboard.append([InlineKeyboardButton("Статистика", callback_data="cmd|stats"),
                     InlineKeyboardButton("Помощь", callback_data="cmd|help")])
    return InlineKeyboardMarkup(keyboard)

def timeframe_keyboard() -> InlineKeyboardMarkup:
    kb = [
        [InlineKeyboardButton("1 минута", callback_data="tf|1m"),
         InlineKeyboardButton("5 минут", callback_data="tf|5m")],
        [InlineKeyboardButton("15 минут", callback_data="tf|15m")],
        [InlineKeyboardButton("Назад", callback_data="cmd|back")]
    ]
    return InlineKeyboardMarkup(kb)

def result_keyboard(signal_id: int) -> InlineKeyboardMarkup:
    kb = [
        [InlineKeyboardButton("Плюс ✅", callback_data=f"result|{signal_id}|plus"),
         InlineKeyboardButton("Минус ❌", callback_data=f"result|{signal_id}|minus")],
        [InlineKeyboardButton("Главное меню", callback_data="cmd|back")]
    ]
    return InlineKeyboardMarkup(kb)

# ---------------------------
# TwelveData fetcher
# ---------------------------
def fetch_ohlcv_twelvedata(symbol: str, interval: str, outputsize: int = CANDLES) -> pd.DataFrame:
    """
    Fetch OHLCV from TwelveData time_series endpoint.
    symbol: like 'EUR/USD' or 'GBP/USD' etc.
    interval: '1min', '5min', '15min'...
    Returns DataFrame with columns ['timestamp','open','high','low','close','volume'] (timestamp UTC)
    """
    if not TWELVEDATA_API_KEY:
        raise RuntimeError("TWELVEDATA_API_KEY not set in environment variables")
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "format": "JSON",
        "apikey": TWELVEDATA_API_KEY
    }
    resp = requests.get(TD_BASE, params=params, timeout=15)
    if resp.status_code != 200:
        raise RuntimeError(f"TwelveData HTTP {resp.status_code}: {resp.text}")
    data = resp.json()
    if "status" in data and data.get("status") == "error":
        # API returns JSON with status:error sometimes
        raise RuntimeError(f"TwelveData API error: {data.get('message') or data}")
    if "values" not in data:
        raise RuntimeError(f"TwelveData returned unexpected payload: {data}")
    vals = data["values"]
    # values are in descending order (most recent first) — convert to df and sort ascending
    df = pd.DataFrame(vals)
    # expected columns: datetime, open, high, low, close, volume
    # Convert and reorder
    if 'datetime' in df.columns:
        df = df.rename(columns={'datetime':'timestamp'})
    df = df[['timestamp','open','high','low','close','volume']]
    # Convert types
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df[['open','high','low','close','volume']] = df[['open','high','low','close','volume']].astype(float)
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df

# ---------------------------
# Indicator computation and signal generation
# ---------------------------
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Given OHLCV df, compute indicators and return augmented df"""
    df2 = df.copy().reset_index(drop=True)
    # Basic indicators using pandas_ta
    df2['rsi'] = ta.rsi(df2['close'], length=14)
    macd = ta.macd(df2['close'])
    # macd may return columns with names, merge
    for col in macd.columns:
        df2[col] = macd[col]
    # EMAs and SMA
    df2['ema9'] = ta.ema(df2['close'], length=9)
    df2['ema21'] = ta.ema(df2['close'], length=21)
    df2['sma50'] = ta.sma(df2['close'], length=50)
    # Bollinger Bands
    bb = ta.bbands(df2['close'], length=20, std=2)
    for col in bb.columns:
        df2[col] = bb[col]
    # Volume SMA
    df2['vol_sma20'] = ta.sma(df2['volume'], length=20)
    # EMA slope (simple diff)
    df2['ema21_slope'] = df2['ema21'].diff()
    # Simple engulfing detection
    df2['bull_engulf'] = ((df2['close'] > df2['open']) &
                         (df2['open'].shift(1) > df2['close'].shift(1)) &
                         (df2['close'] > df2['open'].shift(1)) &
                         (df2['open'] < df2['close'].shift(1))).astype(int)
    df2['bear_engulf'] = ((df2['close'] < df2['open']) &
                         (df2['open'].shift(1) < df2['close'].shift(1)) &
                         (df2['close'] < df2['open'].shift(1)) &
                         (df2['open'] > df2['close'].shift(1))).astype(int)
    return df2

def generate_signal_from_indicators(df: pd.DataFrame) -> Tuple[str, str]:
    """
    Combine indicators via simple weighted voting to produce final signal.
    Returns (signal, reason).
    """
    d = df.copy().reset_index(drop=True).tail(30)
    latest = d.iloc[-1]
    reasons = []
    # RSI
    rsi_vote = 0
    if pd.notna(latest.get('rsi')):
        if latest['rsi'] >= RSI_BUY:
            rsi_vote = 1
            reasons.append(f"RSI выше {RSI_BUY}")
        elif latest['rsi'] <= RSI_SELL:
            rsi_vote = -1
            reasons.append(f"RSI ниже {RSI_SELL}")
    # EMA/SMA trend
    ema_vote = 0
    if pd.notna(latest.get('ema9')) and pd.notna(latest.get('ema21')) and pd.notna(latest.get('sma50')):
        if latest['ema9'] > latest['ema21'] and latest['ema21'] > latest['sma50']:
            ema_vote = 1
            reasons.append("EMA9 > EMA21 > SMA50 (восходящий тренд)")
        elif latest['ema9'] < latest['ema21'] and latest['ema21'] < latest['sma50']:
            ema_vote = -1
            reasons.append("EMA9 < EMA21 < SMA50 (нисходящий тренд)")
    # MACD
    macd_vote = 0
    if 'MACD_12_26_9' in d.columns and 'MACDs_12_26_9' in d.columns and 'MACDh_12_26_9' in d.columns:
        m = latest['MACD_12_26_9']
        ms = latest['MACDs_12_26_9']
        mh = latest['MACDh_12_26_9']
        if m > ms and mh > 0:
            macd_vote = 1
            reasons.append("MACD растущий")
        elif m < ms and mh < 0:
            macd_vote = -1
            reasons.append("MACD падающий")
    # Bollinger Bands
    bb_vote = 0
    if 'BBU_20_2.0' in latest and 'BBL_20_2.0' in latest:
        if latest['close'] >= latest['BBU_20_2.0']:
            bb_vote = -1
            reasons.append("Цена у верхней линии Боллинджера")
        elif latest['close'] <= latest['BBL_20_2.0']:
            bb_vote = 1
            reasons.append("Цена у нижней линии Боллинджера")
    # Volume spike
    vol_vote = 0
    if pd.notna(latest.get('vol_sma20')) and latest['volume'] > 1.5 * latest['vol_sma20']:
        if latest['close'] > latest['open']:
            vol_vote = 1
            reasons.append("Объём высокий и свеча бычья")
        else:
            vol_vote = -1
            reasons.append("Объём высокий и свеча медвежья")
    # Candlestick
    cs_vote = 0
    if latest.get('bull_engulf', 0) == 1:
        cs_vote = 1
        reasons.append("Бычье engulfing")
    if latest.get('bear_engulf', 0) == 1:
        cs_vote = -1
        reasons.append("Медвежье engulfing")
    # Trend slope
    trend_vote = 0
    if pd.notna(latest.get('ema21_slope')):
        if latest['ema21_slope'] > 0:
            trend_vote = 1
        elif latest['ema21_slope'] < 0:
            trend_vote = -1

    votes = {'rsi': rsi_vote, 'ema': ema_vote, 'macd': macd_vote, 'bb': bb_vote,
             'vol': vol_vote, 'cs': cs_vote, 'trend': trend_vote}
    score = sum(votes.values())

    if score >= 2:
        signal = "BUY"
    elif score <= -2:
        signal = "SELL"
    else:
        signal = "NO TRADE"

    reason = "; ".join(reasons) if reasons else "Индикаторы не дали явного сигнала"
    # short numeric summary
    rsi_val = latest['rsi'] if pd.notna(latest.get('rsi')) else None
    ema9 = latest['ema9'] if pd.notna(latest.get('ema9')) else None
    ema21 = latest['ema21'] if pd.notna(latest.get('ema21')) else None
    macdh = latest.get('MACDh_12_26_9') if latest.get('MACDh_12_26_9') is not None else None
    summary_parts = []
    if rsi_val is not None:
        summary_parts.append(f"RSI={rsi_val:.1f}")
    if ema9 is not None and ema21 is not None:
        summary_parts.append(f"EMA9={ema9:.5g},EMA21={ema21:.5g}")
    if macdh is not None:
        summary_parts.append(f"MACDh={macdh:.5g}")
    summary = ", ".join(summary_parts)
    full_reason = f"{reason}. Кратко: {summary}."
    return signal, full_reason

def analyze_pair(symbol: str, timeframe_label: str) -> Tuple[str, str]:
    """
    symbol: e.g., 'EUR/USD'
    timeframe_label: one of keys of TF_MAP like '1m','5m','15m'
    Returns (signal, reason)
    """
    interval = TF_MAP.get(timeframe_label)
    if not interval:
        raise ValueError("Unsupported timeframe")
    df = fetch_ohlcv_twelvedata(symbol, interval, outputsize=CANDLES)
    if df.empty:
        raise RuntimeError("No data received from TwelveData")
    df_ind = compute_indicators(df)
    signal, reason = generate_signal_from_indicators(df_ind)
    return signal, reason

# ---------------------------
# Bot state & handlers
# ---------------------------
CHAT_STATE: Dict[int, Dict[str, str]] = {}  # chat_id -> {"pair":..., ...}

def start(update: Update, context: CallbackContext):
    chat_id = update.effective_chat.id
    text = f"<pre>{ASCII_LOGO}</pre>\nПривет! Выбери валютную пару для анализа:"
    context.bot.send_message(chat_id=chat_id, text=text, parse_mode=ParseMode.HTML, reply_markup=main_menu_keyboard(PAIRS))

def help_cmd(update: Update, context: CallbackContext):
    text = ("Я бот сигнальный. Рабочий процесс:\n"
            "1) /start — выбрать пару\n"
            "2) выбрать таймфрейм (1m/5m/15m)\n"
            "3) бот генерирует сигнал и причину\n"
            "4) после сделки нажми Плюс/Минус — бот обновит статистику и вернёт в главное меню\n\n"
            "Команды:\n/start\n/stats\n/help")
    update.message.reply_text(text)

def stats_cmd(update: Update, context: CallbackContext):
    s = storage.get_stats()
    text = (f"📊 <b>Статистика</b>\n\nВсего сигналов: {s['total']}\n"
            f"Плюсы: {s['plus']}\nМинусы: {s['minus']}\nТочность: {s['accuracy']}%\n\nПоследние 10 сигналов:\n")
    for item in s['last10']:
        text += f"#{item['id']} {item['ts'][:19]} {item['symbol']} {item['timeframe']} — {item['signal']} — результат: {item['result']}\n"
    update.message.reply_text(text, parse_mode=ParseMode.HTML)

def callback_handler(update: Update, context: CallbackContext):
    query = update.callback_query
    data = query.data
    chat_id = query.message.chat.id
    query.answer()
    try:
        if data.startswith("pair|"):
            pair = data.split("|",1)[1]
            CHAT_STATE[chat_id] = {"pair": pair}
            context.bot.send_message(chat_id=chat_id, text=f"Пара выбрана: {pair}\nВыберите таймфрейм:", reply_markup=timeframe_keyboard())
            return
        if data.startswith("tf|"):
            tf = data.split("|",1)[1]
            state = CHAT_STATE.get(chat_id)
            if not state or "pair" not in state:
                context.bot.send_message(chat_id=chat_id, text="Сначала выберите пару.", reply_markup=main_menu_keyboard(PAIRS))
                return
            pair = state["pair"]
            # inform
            context.bot.send_message(chat_id=chat_id, text=f"Генерирую сигнал для {pair} на {tf}...\n(обращаюсь к TwelveData — это может занять 1-4 секунды)")
            try:
                signal, reason = analyze_pair(pair, tf)
            except Exception as e:
                logger.exception("Анализ завершился ошибкой")
                context.bot.send_message(chat_id=chat_id, text=f"Ошибка при получении данных/анализе: {e}\nПопробуйте другой таймфрейм или пару.", reply_markup=main_menu_keyboard(PAIRS))
                return
            sid = storage.add_signal(pair, tf, signal, reason)
            msg = f"Сигнал на {tf} по {pair}: <b>{signal}</b>\nПричина: {reason}\n\nID сигнала: {sid}\n\nОтметьте результат сделки:"
            context.bot.send_message(chat_id=chat_id, text=msg, parse_mode=ParseMode.HTML, reply_markup=result_keyboard(sid))
            return
        if data.startswith("result|"):
            # result|<id>|plus
            parts = data.split("|")
            if len(parts) != 3:
                return
            sid = int(parts[1])
            res = parts[2]
            if res == "plus":
                storage.set_result(sid, "plus")
                context.bot.send_message(chat_id=chat_id, text=f"Отмечено: ПЛЮС для сигнала #{sid}")
            else:
                storage.set_result(sid, "minus")
                context.bot.send_message(chat_id=chat_id, text=f"Отмечено: МИНУС для сигнала #{sid}")
            # After marking result, return to main menu as requested
            time.sleep(0.2)
            context.bot.send_message(chat_id=chat_id, text="Возврат в главное меню. Выберите пару для торговли:", reply_markup=main_menu_keyboard(PAIRS))
            return
        if data.startswith("cmd|"):
            cmd = data.split("|",1)[1]
            if cmd == "stats":
                s = storage.get_stats()
                text = (f"📊 <b>Статистика</b>\n\nВсего сигналов: {s['total']}\n"
                        f"Плюсы: {s['plus']}\nМинусы: {s['minus']}\nТочность: {s['accuracy']}%\n\nПоследние 10 сигналов:\n")
                for item in s['last10']:
                    text += f"#{item['id']} {item['ts'][:19]} {item['symbol']} {item['timeframe']} — {item['signal']} — результат: {item['result']}\n"
                context.bot.send_message(chat_id=chat_id, text=text, parse_mode=ParseMode.HTML)
            elif cmd == "help":
                context.bot.send_message(chat_id=chat_id, text="Инструкция: выберите пару, затем таймфрейм. После сигнала отметьте результат сделки (Плюс/Минус).")
            elif cmd == "back":
                context.bot.send_message(chat_id=chat_id, text="Главное меню: выберите пару.", reply_markup=main_menu_keyboard(PAIRS))
            return
    except Exception as e:
        logger.exception("Ошибка в callback_handler")
        context.bot.send_message(chat_id=chat_id, text=f"Произошла внутренняя ошибка: {e}")

def error_handler(update: Update, context: CallbackContext):
    logger.exception("Update caused error: %s", context.error)

# ---------------------------
# Main
# ---------------------------
def main():
    if not TELEGRAM_TOKEN:
        print("ERROR: TELEGRAM_TOKEN not set. Set TELEGRAM_TOKEN in environment variables.")
        return
    if not TWELVEDATA_API_KEY:
        print("ERROR: TWELVEDATA_API_KEY not set. Set TWELVEDATA_API_KEY in environment variables.")
        return

    updater = Updater(token=TELEGRAM_TOKEN, use_context=True)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("help", help_cmd))
    dp.add_handler(CommandHandler("stats", stats_cmd))
    dp.add_handler(CallbackQueryHandler(callback_handler))
    dp.add_error_handler(error_handler)

    logger.info("Bot starting...")
    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()
