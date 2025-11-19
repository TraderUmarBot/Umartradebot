#!/usr/bin/env python3
# main.py
"""
OTC Signals Telegram Bot — single-file
- Categories: Forex OTC, Crypto OTC, Stocks OTC
- Data source: TwelveData API (set TWELVEDATA_API_KEY)
- Telegram token: TELEGRAM_TOKEN
- Expiries: 30s (proxy 1min), 1m, 3m (resampled), 5m
- "Подождите..." ~10s before sending signal
- After result (Плюс/Минус) -> save in SQLite and return to main menu
- ~15 indicators combined by weighted voting
"""
import os
import time
import logging
import requests
import sqlite3
from datetime import datetime
from typing import List, Dict, Any, Tuple

import pandas as pd
import numpy as np
import pandas_ta as ta

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, ParseMode
from telegram.ext import Updater, CommandHandler, CallbackQueryHandler, CallbackContext
from telegram.update import Update

# -------------------------
# CONFIG
# -------------------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY", "")
DB_FILE = os.getenv("DB_FILE", "otc_signals.db")

# Provided lists from you (exact as given). We'll strip the trailing " OTC" when mapping.
FOREX_OTC_RAW = [
"EUR/USD OTC","AUD/CAD OTC","AUD/CHF OTC","AUD/JPY OTC","AUD/NZD OTC","AUD/USD OTC","CAD/CHF OTC","CAD/JPY OTC",
"CHF/JPY OTC","EUR/CHF OTC","EUR/GBP OTC","EUR/JPY OTC","EUR/NZD OTC","GBP/AUD OTC","GBP/JPY OTC","GBP/USD OTC",
"NZD/JPY OTC","NZD/USD OTC","USD/CAD OTC","USD/CHF OTC","USD/JPY OTC","USD/RUB OTC","EUR/RUB OTC","CHF/NOK OTC",
"EUR/HUF OTC","USD/CNH OTC","EUR/TRY OTC","USD/INR OTC","USD/SGD OTC","USD/CLP OTC","USD/MYR OTC","USD/THB OTC",
"USD/VND OTC","USD/PKR OTC","USD/EGP OTC","USD/PHP OTC","USD/MXN OTC","USD/DZD OTC","USD/ARS OTC","YER/USD OTC",
"LBP/USD OTC","TND/USD OTC","MAD/USD OTC","BHD/CNY OTC","AED/CNY OTC","SAR/CNY OTC","QAR/CNY OTC","OMR/CNY OTC",
"JOD/CNY OTC","NGN/USD OTC","KES/USD OTC","ZAR/USD OTC","UAH/USD OTC"
]

CRYPTO_OTC_RAW = [
"Bitcoin OTC","Toncoin OTC","Cardano OTC","Dogecoin OTC","Litecoin OTC","Avalanche OTC","Ethereum OTC","Solana OTC",
"TRON OTC","Polygon OTC","Chainlink OTC","BNB OTC","Polkadot OTC","Bitcoin ETF OTC"
]

STOCKS_OTC_RAW = [
"Apple OTC","McDonald’s OTC","Tesla OTC","Pfizer Inc OTC","American Express OTC","FedEx OTC","VISA OTC","Alibaba OTC",
"Netflix OTC","Marathon Digital Holdings OTC","Citigroup Inc OTC","Johnson & Johnson OTC","Microsoft OTC","Amazon OTC",
"VIX OTC","Boeing Company OTC","GameStop Corp OTC","Palantir Technologies OTC","Advanced Micro Devices (AMD) OTC",
"Coinbase Global OTC"
]

# Normalize lists: strip " OTC" and common punctuation
def normalize_name(n: str) -> str:
    return n.replace(" OTC","").replace("’","'").replace(" (AMD)"," AMD").strip()

FOREX_OTC = [normalize_name(x) for x in FOREX_OTC_RAW]
CRYPTO_OTC = [normalize_name(x) for x in CRYPTO_OTC_RAW]
STOCKS_OTC = [normalize_name(x) for x in STOCKS_OTC_RAW]

# Expiry options
EXPIRIES = ["30s","1m","3m","5m"]
# Map expiry to TwelveData interval (approx)
TF_MAP = {"30s":"1min","1m":"1min","3m":"1min","5m":"5min"}

# TwelveData time_series endpoint
TD_URL = "https://api.twelvedata.com/time_series"

# candles to request
CANDLES = 500

# thresholds
RSI_BUY = 55
RSI_SELL = 45

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("otc_bot")

# -------------------------
# Symbol mapping dictionaries (best-effort)
# For Crypto and Stocks we map friendly names -> symbols used by TwelveData
# Many TwelveData tickers are like "BTC/USD", "AAPL" etc.
# We'll attempt to map the common ones; unknown names will be used as-is.
# -------------------------
CRYPTO_MAP = {
    "Bitcoin":"BTC/USD",
    "Toncoin":"TON/USD",       # TON is common symbol
    "Cardano":"ADA/USD",
    "Dogecoin":"DOGE/USD",
    "Litecoin":"LTC/USD",
    "Avalanche":"AVAX/USD",
    "Ethereum":"ETH/USD",
    "Solana":"SOL/USD",
    "TRON":"TRX/USD",
    "Polygon":"MATIC/USD",
    "Chainlink":"LINK/USD",
    "BNB":"BNB/USD",
    "Polkadot":"DOT/USD",
    "Bitcoin ETF":"BITO"       # example ETF ticker (may vary); TwelveData might not have ETF name — attempt
}

STOCK_MAP = {
    "Apple":"AAPL",
    "McDonald's":"MCD",
    "Tesla":"TSLA",
    "Pfizer Inc":"PFE",
    "American Express":"AXP",
    "FedEx":"FDX",
    "VISA":"V",
    "Alibaba":"BABA",
    "Netflix":"NFLX",
    "Marathon Digital Holdings":"MARA",
    "Citigroup Inc":"C",
    "Johnson & Johnson":"JNJ",
    "Microsoft":"MSFT",
    "Amazon":"AMZN",
    "VIX":"^VIX",  # note: volatility indices sometimes not available
    "Boeing Company":"BA",
    "GameStop Corp":"GME",
    "Palantir Technologies":"PLTR",
    "Advanced Micro Devices AMD":"AMD",
    "Coinbase Global":"COIN"
}

# For Forex pairs we will remove " / " and keep as e.g. "EUR/USD" which TwelveData understands
# Some exotic tickers like USD/RUB likely exist in TwelveData as "USD/RUB"

# -------------------------
# DB
# -------------------------
CREATE_SQL = """
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT,
    user_id INTEGER,
    category TEXT,
    symbol TEXT,
    expiry TEXT,
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
        c.execute(CREATE_SQL)
        conn.commit()
        conn.close()

    def add_signal(self, user_id:int, category:str, symbol:str, expiry:str, signal:str, reason:str) -> int:
        conn = self._get_conn()
        c = conn.cursor()
        ts = datetime.utcnow().isoformat()
        c.execute("INSERT INTO signals (ts, user_id, category, symbol, expiry, signal, reason, result) VALUES (?,?,?,?,?,?,?,?)",
                  (ts, user_id, category, symbol, expiry, signal, reason, None))
        conn.commit()
        sid = c.lastrowid
        conn.close()
        return sid

    def set_result(self, sid:int, result:str):
        conn = self._get_conn()
        c = conn.cursor()
        c.execute("UPDATE signals SET result = ? WHERE id = ?", (result, sid))
        conn.commit()
        conn.close()

    def get_stats(self) -> Dict[str,Any]:
        conn = self._get_conn()
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM signals")
        total = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM signals WHERE result='plus'")
        plus = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM signals WHERE result='minus'")
        minus = c.fetchone()[0]
        accuracy = round((plus/total*100),2) if total>0 else 0.0
        c.execute("SELECT id, ts, user_id, category, symbol, expiry, signal, reason, result FROM signals ORDER BY id DESC LIMIT 10")
        rows = c.fetchall()
        last10 = [dict(zip(["id","ts","user_id","category","symbol","expiry","signal","reason","result"], r)) for r in rows]
        conn.close()
        return {"total":total,"plus":plus,"minus":minus,"accuracy":accuracy,"last10":last10}

storage = Storage()

# -------------------------
# UI (keyboards)
# -------------------------
ASCII_LOGO = r"""
 ____  _______  _____ 
|  _ \|__   __|/ ____|
| |_) |  | |  | (___  
|  _ <   | |   \___ \ 
| |_) |  | |   ____) |
|____/   |_|  |_____/ 
OTC Signals Bot — v1.0
"""

def chunk_buttons(items: List[str], cols:int=2):
    kb=[]
    row=[]
    for item in items:
        row.append(InlineKeyboardButton(item, callback_data=f"sel|{item}"))
        if len(row)==cols:
            kb.append(row)
            row=[]
    if row:
        kb.append(row)
    return kb

def main_menu_kb():
    kb = []
    kb.append([InlineKeyboardButton("Forex OTC", callback_data="cat|forex"),
               InlineKeyboardButton("Crypto OTC", callback_data="cat|crypto")])
    kb.append([InlineKeyboardButton("Stocks OTC", callback_data="cat|stocks"),
               InlineKeyboardButton("Статистика", callback_data="cmd|stats")])
    kb.append([InlineKeyboardButton("Помощь", callback_data="cmd|help")])
    return InlineKeyboardMarkup(kb)

def list_menu_kb(items: List[str]):
    kb = chunk_buttons(items, cols=2)
    kb.append([InlineKeyboardButton("Назад", callback_data="cmd|back")])
    return InlineKeyboardMarkup(kb)

def expiry_kb():
    kb = [
        [InlineKeyboardButton("30s", callback_data="exp|30s"),
         InlineKeyboardButton("1m", callback_data="exp|1m")],
        [InlineKeyboardButton("3m", callback_data="exp|3m"),
         InlineKeyboardButton("5m", callback_data="exp|5m")],
        [InlineKeyboardButton("Отмена", callback_data="cmd|back")]
    ]
    return InlineKeyboardMarkup(kb)

def result_kb(sid:int):
    kb = [
        [InlineKeyboardButton("Плюс ✅", callback_data=f"res|{sid}|plus"),
         InlineKeyboardButton("Минус ❌", callback_data=f"res|{sid}|minus")]
    ]
    return InlineKeyboardMarkup(kb)

# -------------------------
# TwelveData fetcher
# -------------------------
def fetch_td(symbol: str, interval: str, outputsize:int=CANDLES) -> pd.DataFrame:
    """
    Fetch from TwelveData. symbol should be like 'EUR/USD' or 'BTC/USD' or 'AAPL'
    interval: '1min','5min', etc.
    """
    if not TWELVEDATA_API_KEY:
        raise RuntimeError("TWELVEDATA_API_KEY not set")
    params = {"symbol": symbol, "interval": interval, "outputsize": outputsize, "format":"JSON", "apikey":TWELVEDATA_API_KEY}
    r = requests.get(TD_URL, params=params, timeout=15)
    if r.status_code != 200:
        raise RuntimeError(f"TwelveData HTTP {r.status_code}: {r.text}")
    j = r.json()
    if j.get("status") == "error":
        raise RuntimeError(f"TwelveData error: {j.get('message')}")
    vals = j.get("values")
    if not vals:
        raise RuntimeError("TwelveData returned empty values")
    df = pd.DataFrame(vals)
    if 'datetime' in df.columns:
        df = df.rename(columns={'datetime':'timestamp'})
    df = df[['timestamp','open','high','low','close','volume']]
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df[['open','high','low','close','volume']] = df[['open','high','low','close','volume']].astype(float)
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df

# -------------------------
# Indicators & signal logic (~15 indicators)
# -------------------------
def compute_indicators(df:pd.DataFrame) -> pd.DataFrame:
    d = df.copy().reset_index(drop=True)
    # momentum
    d['rsi'] = ta.rsi(d['close'], length=14)
    st = ta.stoch(d['high'], d['low'], d['close'])
    if 'STOCHk_14_3_3' in st.columns:
        d['stoch_k'] = st['STOCHk_14_3_3']
        d['stoch_d'] = st['STOCHd_14_3_3']
    # MACD
    macd = ta.macd(d['close'])
    for col in macd.columns:
        d[col] = macd[col]
    # EMAs/SMA
    d['ema9'] = ta.ema(d['close'], length=9)
    d['ema21'] = ta.ema(d['close'], length=21)
    d['ema50'] = ta.ema(d['close'], length=50)
    d['sma50'] = ta.sma(d['close'], length=50)
    d['sma200'] = ta.sma(d['close'], length=200)
    # Bollinger
    bb = ta.bbands(d['close'], length=20, std=2)
    for col in bb.columns:
        d[col] = bb[col]
    # Volatility and trend
    d['atr'] = ta.atr(d['high'], d['low'], d['close'], length=14)
    d['adx'] = ta.adx(d['high'], d['low'], d['close'])['ADX_14']
    # OBV / MFI / CCI / Williams / ROC / PSAR
    d['obv'] = ta.obv(d['close'], d['volume'])
    d['mfi'] = ta.mfi(d['high'], d['low'], d['close'], d['volume'], length=14)
    d['cci'] = ta.cci(d['high'], d['low'], d['close'], length=20)
    d['williams'] = ta.willr(d['high'], d['low'], d['close'], length=14)
    d['roc'] = ta.roc(d['close'], length=12)
    try:
        psar = ta.psar(d['high'], d['low'], d['close'])
        # pick PSAR long column if exists
        if 'PSARl_0.02_0.2' in psar.columns:
            d['psar'] = psar['PSARl_0.02_0.2']
        else:
            d['psar'] = np.nan
    except Exception:
        d['psar'] = np.nan
    # candlestick simple patterns
    d['bull_engulf'] = ((d['close'] > d['open']) & (d['open'].shift(1) > d['close'].shift(1)) & (d['close'] > d['open'].shift(1))).astype(int)
    d['bear_engulf'] = ((d['close'] < d['open']) & (d['open'].shift(1) < d['close'].shift(1)) & (d['close'] < d['open'].shift(1))).astype(int)
    d['ema21_slope'] = d['ema21'].diff()
    return d

def vote(df:pd.DataFrame) -> Tuple[str,str]:
    d = df.copy().reset_index(drop=True).tail(60)
    latest = d.iloc[-1]
    votes = []
    reasons = []

    # 1 RSI
    if pd.notna(latest.get('rsi')):
        if latest['rsi'] > RSI_BUY:
            votes.append(1); reasons.append("RSI>55")
        elif latest['rsi'] < RSI_SELL:
            votes.append(-1); reasons.append("RSI<45")
    # 2 MACD
    if 'MACD_12_26_9' in d.columns and 'MACDs_12_26_9' in d.columns and 'MACDh_12_26_9' in d.columns:
        if latest['MACD_12_26_9'] > latest['MACDs_12_26_9'] and latest['MACDh_12_26_9'] > 0:
            votes.append(1); reasons.append("MACD растёт")
        elif latest['MACD_12_26_9'] < latest['MACDs_12_26_9'] and latest['MACDh_12_26_9'] < 0:
            votes.append(-1); reasons.append("MACD падает")
    # 3 EMA alignment
    if pd.notna(latest.get('ema9')) and pd.notna(latest.get('ema21')) and pd.notna(latest.get('ema50')):
        if latest['ema9'] > latest['ema21'] > latest['ema50']:
            votes.append(1); reasons.append("EMA9>EMA21>EMA50")
        elif latest['ema9'] < latest['ema21'] < latest['ema50']:
            votes.append(-1); reasons.append("EMA9<EMA21<EMA50")
    # 4 SMA200 context
    if pd.notna(latest.get('sma200')):
        if latest['close'] > latest['sma200']:
            votes.append(1); reasons.append("Цена выше SMA200")
        else:
            votes.append(-1); reasons.append("Цена ниже SMA200")
    # 5 Bollinger
    if pd.notna(latest.get('BBU_20_2.0')) and pd.notna(latest.get('BBL_20_2.0')):
        if latest['close'] >= latest['BBU_20_2.0']:
            votes.append(-1); reasons.append("Цена у верхней полосы Боллинджера")
        elif latest['close'] <= latest['BBL_20_2.0']:
            votes.append(1); reasons.append("Цена у нижней полосы Боллинджера")
    # 6 ATR breakout
    if pd.notna(latest.get('atr')):
        ma20 = d['close'].rolling(20).mean().iloc[-1]
        if pd.notna(ma20):
            if latest['close'] > ma20 + 1.5*latest['atr']:
                votes.append(1); reasons.append("Прорыв выше (ATR)")
            elif latest['close'] < ma20 - 1.5*latest['atr']:
                votes.append(-1); reasons.append("Прорыв ниже (ATR)")
    # 7 ADX and slope
    if pd.notna(latest.get('adx')) and pd.notna(latest.get('ema21_slope')):
        if latest['adx'] > 25:
            if latest['ema21_slope'] > 0:
                votes.append(1); reasons.append("ADX>25, тренд вверх")
            else:
                votes.append(-1); reasons.append("ADX>25, тренд вниз")
    # 8 Volume spike
    vol_sma = d['volume'].rolling(20).mean().iloc[-1]
    if pd.notna(vol_sma) and latest['volume'] > 1.5*vol_sma:
        if latest['close'] > latest['open']:
            votes.append(1); reasons.append("Объём высокий и свеча бычья")
        else:
            votes.append(-1); reasons.append("Объём высокий и свеча медвежья")
    # 9 MFI
    if pd.notna(latest.get('mfi')):
        if latest['mfi'] > 80: votes.append(-1); reasons.append("MFI перекуплен")
        elif latest['mfi'] < 20: votes.append(1); reasons.append("MFI перепродан")
    # 10 Stochastic
    if pd.notna(latest.get('stoch_k')) and pd.notna(latest.get('stoch_d')):
        if latest['stoch_k'] > latest['stoch_d'] and latest['stoch_k'] < 20:
            votes.append(1); reasons.append("Stoch бычье в перепрод")
        elif latest['stoch_k'] < latest['stoch_d'] and latest['stoch_k'] > 80:
            votes.append(-1); reasons.append("Stoch медв в перекуп")
    # 11 CCI
    if pd.notna(latest.get('cci')):
        if latest['cci'] > 100: votes.append(1); reasons.append("CCI>100")
        elif latest['cci'] < -100: votes.append(-1); reasons.append("CCI<-100")
    # 12 Williams
    if pd.notna(latest.get('williams')):
        if latest['williams'] < -80: votes.append(1); reasons.append("Williams перепрод")
        elif latest['williams'] > -20: votes.append(-1); reasons.append("Williams перекуп")
    # 13 ROC
    if pd.notna(latest.get('roc')):
        if latest['roc'] > 0: votes.append(1); reasons.append("ROC положительный")
        else: votes.append(-1); reasons.append("ROC отрицательный")
    # 14 PSAR
    if pd.notna(latest.get('psar')):
        if latest['close'] > latest['psar']: votes.append(1); reasons.append("Цена выше PSAR")
        else: votes.append(-1); reasons.append("Цена ниже PSAR")
    # 15 Candles
    if latest.get('bull_engulf',0)==1: votes.append(1); reasons.append("Bullish engulf")
    if latest.get('bear_engulf',0)==1: votes.append(-1); reasons.append("Bearish engulf")

    score = sum(votes)
    if score >= 3:
        sig = "BUY"
    elif score <= -3:
        sig = "SELL"
    else:
        sig = "NO TRADE"
    reason = "; ".join(reasons[:8]) if reasons else "Индикаторы не дали явного сигнала"
    nums = []
    if pd.notna(latest.get('rsi')): nums.append(f"RSI={latest['rsi']:.1f}")
    if pd.notna(latest.get('MACDh_12_26_9')): nums.append(f"MACDh={latest['MACDh_12_26_9']:.4g}")
    if pd.notna(latest.get('ema9')) and pd.notna(latest.get('ema21')): nums.append(f"EMA9/21={latest['ema9']:.5g}/{latest['ema21']:.5g}")
    summary = ", ".join(nums)
    full_reason = f"{reason}. Кратко: {summary}."
    return sig, full_reason

# -------------------------
# Chat state
# -------------------------
CHAT_STATE: Dict[int, Dict[str,str]] = {}

# -------------------------
# Handlers
# -------------------------
def start(update:Update, context:CallbackContext):
    chat = update.effective_chat.id
    text = f"<pre>{ASCII_LOGO}</pre>\nПривет! Выберите раздел для OTC-сигналов:"
    context.bot.send_message(chat_id=chat, text=text, parse_mode=ParseMode.HTML, reply_markup=main_menu_kb())

def help_cmd(update:Update, context:CallbackContext):
    txt = ("Инструкция:\n1) /start — главное меню\n2) Выбери раздел → пару → время экспирации\n3) Подожди ~10 секунд — бот анализирует и выдаёт сигнал\n4) После завершения сделки нажми Плюс/Минус — результат сохранится\nПосле нажатия результата бот вернёт тебя в главное меню.")
    update.message.reply_text(txt)

def stats_cmd(update:Update, context:CallbackContext):
    s = storage.get_stats()
    txt = f"📊 Статистика\nВсего: {s['total']}\nПлюсы: {s['plus']}\nМинусы: {s['minus']}\nТочность: {s['accuracy']}%\n\nПоследние 10:"
    for it in s['last10']:
        txt += f"\n#{it['id']} {it['ts'][:19]} user:{it['user_id']} {it['category']} {it['symbol']} {it['expiry']} → {it['signal']} ({it['result']})"
    update.message.reply_text(txt)

# Utility to map display name -> fetch symbol
def get_td_symbol(category:str, display_name:str) -> str:
    # Forex: display_name like "EUR/USD" — TwelveData accepts same
    if category == "forex":
        # ensure format "EUR/USD"
        name = display_name.strip()
        return name
    if category == "crypto":
        return CRYPTO_MAP.get(display_name, display_name.replace(" ","/") + "/USD" if "/" not in display_name else display_name)
    if category == "stocks":
        # try mapping
        # some stock names in list include punctuation; handle special cases
        key = display_name.replace(" (AMD)"," AMD").replace("’","'").strip()
        mapped = STOCK_MAP.get(key)
        if mapped:
            return mapped
        # fallback: use uppercase first token
        token = "".join(ch for ch in key if ch.isalnum() or ch==" ").split()[0].upper()
        return token
    return display_name

def callback_handler(update:Update, context:CallbackContext):
    query = update.callback_query
    data = query.data
    chat = query.message.chat.id
    query.answer()
    try:
        if data.startswith("cat|"):
            cat = data.split("|",1)[1]
            CHAT_STATE[chat] = {"category":cat}
            if cat == "forex":
                context.bot.send_message(chat_id=chat, text="Выберите Forex OTC пару:", reply_markup=list_menu_kb(FOREX_OTC))
            elif cat == "crypto":
                context.bot.send_message(chat_id=chat, text="Выберите Crypto OTC пару:", reply_markup=list_menu_kb(CRYPTO_OTC))
            elif cat == "stocks":
                context.bot.send_message(chat_id=chat, text="Выберите Stocks OTC:", reply_markup=list_menu_kb(STOCKS_OTC))
            return

        if data.startswith("sel|"):
            sel = data.split("|",1)[1]
            state = CHAT_STATE.get(chat, {})
            if not state or "category" not in state:
                context.bot.send_message(chat_id=chat, text="Сначала выберите раздел.", reply_markup=main_menu_kb())
                return
            state["symbol_display"] = sel
            CHAT_STATE[chat] = state
            context.bot.send_message(chat_id=chat, text=f"Пара/актив: {sel}\nВыберите время экспирации:", reply_markup=expiry_kb())
            return

        if data.startswith("exp|"):
            exp = data.split("|",1)[1]
            state = CHAT_STATE.get(chat, {})
            if not state or "symbol_display" not in state:
                context.bot.send_message(chat_id=chat, text="Сначала выберите пару/актив.", reply_markup=main_menu_kb())
                return
            display = state["symbol_display"]
            category = state["category"]
            # show analyzing message
            analyzing = context.bot.send_message(chat_id=chat, text=f"Подождите, идёт анализ {display} для экспирации {exp}...\n(примерно 10 секунд)")
            # perform analysis
            try:
                td_interval = TF_MAP.get(exp, "1min")
                td_symbol = get_td_symbol(category, display)
                df = fetch_td(td_symbol, td_interval, outputsize=CANDLES)
                # resample if 3m
                if exp == "3m":
                    df = df.set_index('timestamp').resample("3T").agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna().reset_index()
                df_ind = compute_indicators(df)
                # wait ~10 seconds (simulate/process)
                time.sleep(10)
                sig, reason = vote(df_ind)
            except Exception as e:
                logger.exception("Analysis error")
                context.bot.send_message(chat_id=chat, text=f"Ошибка при анализе {display}: {e}\nВернулся в главное меню.", reply_markup=main_menu_kb())
                return
            user_id = update.effective_user.id
            sid = storage.add_signal(user_id, category, display, exp, sig, reason)
            out = f"Сигнал ({exp}) по {display}: <b>{sig}</b>\nПричина: {reason}\n\nID: {sid}\n\nКогда сделка завершится — отметьте результат:"
            context.bot.send_message(chat_id=chat, text=out, parse_mode=ParseMode.HTML, reply_markup=result_kb(sid))
            return

        if data.startswith("res|"):
            parts = data.split("|")
            if len(parts) != 3:
                return
            sid = int(parts[1]); res = parts[2]
            if res == "plus":
                storage.set_result(sid, "plus")
                context.bot.send_message(chat_id=chat, text=f"Отмечено: ПЛЮС для сигнала #{sid}")
            else:
                storage.set_result(sid, "minus")
                context.bot.send_message(chat_id=chat, text=f"Отмечено: МИНУС для сигнала #{sid}")
            # return to main menu
            time.sleep(0.2)
            context.bot.send_message(chat_id=chat, text="Возврат в главное меню:", reply_markup=main_menu_kb())
            return

        if data.startswith("cmd|"):
            cmd = data.split("|",1)[1]
            if cmd == "stats":
                s = storage.get_stats()
                txt = f"📊 Статистика\nВсего: {s['total']}\nПлюсы: {s['plus']}\nМинусы: {s['minus']}\nТочность: {s['accuracy']}%\n\nПоследние 10:"
                for it in s['last10']:
                    txt += f"\n#{it['id']} {it['ts'][:19]} user:{it['user_id']} {it['category']} {it['symbol']} {it['expiry']} → {it['signal']} ({it['result']})"
                context.bot.send_message(chat_id=chat, text=txt)
            elif cmd == "help":
                context.bot.send_message(chat_id=chat, text="Инструкция: выберите раздел → актив → экспирация → дождитесь сигнала → после сделки отметьте результат. Бот вернёт вас в главное меню.")
            elif cmd == "back":
                context.bot.send_message(chat_id=chat, text="Главное меню:", reply_markup=main_menu_kb())
            return

    except Exception as e:
        logger.exception("Callback handler error")
        context.bot.send_message(chat_id=chat, text=f"Внутренняя ошибка: {e}")

def error_handler(update:Update, context:CallbackContext):
    logger.exception("Update caused error: %s", context.error)

# -------------------------
# Main
# -------------------------
def main():
    if not TELEGRAM_TOKEN:
        print("ERROR: TELEGRAM_TOKEN env var not set")
        return
    if not TWELVEDATA_API_KEY:
        print("ERROR: TWELVEDATA_API_KEY env var not set")
        return
    updater = Updater(token=TELEGRAM_TOKEN, use_context=True)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("help", help_cmd))
    dp.add_handler(CommandHandler("stats", stats_cmd))
    dp.add_handler(CallbackQueryHandler(callback_handler))
    dp.add_error_handler(error_handler)
    logger.info("Starting OTC Signals Bot")
    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()
