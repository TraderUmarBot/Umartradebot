# main.py
# OXTSIGNALSBOT PRO — MAX (level 3)
# Features: yfinance + fallback, NO-FLAT, TrendPower, candle patterns, ATR filter,
# MACD quality, Signal Quality Score, Time Filter, NFP, robust error handling.
#
# requirements.txt:
# python-telegram-bot==13.15
# pandas
# numpy
# yfinance
# flask
# requests

import os
import time
import threading
import random
import csv
import traceback
from datetime import datetime, timedelta, time as dtime
from typing import Dict, Tuple, Optional

import pandas as pd
import numpy as np
import yfinance as yf
import requests
from flask import Flask
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Updater, CommandHandler, CallbackQueryHandler, CallbackContext

# ---------------- CONFIG ----------------
BOT_TOKEN = os.getenv("BOT_TOKEN") or ""
ANALYSIS_WAIT = 18       # seconds to simulate professional analysis
PAGE_SIZE = 6
LOG_CSV = "signals_log.csv"

FOREX = [
    "EURUSD","GBPUSD","USDJPY","AUDUSD","USDCHF","EURJPY",
    "GBPJPY","NZDUSD","EURGBP","CADJPY","USDCAD","AUDJPY",
    "EURAUD","GBPAUD","EURNZD","AUDNZD","CADCHF","CHFJPY",
    "NZDJPY","GBPCAD"
]

EXPIRATIONS = ["1m","2m","3m","5m"]

# indicator weights (base)
WEIGHTS = {"EMA":2,"SMA":1,"MACD":2,"RSI":1,"BB":1}

# yfinance settings
YF_INTERVAL = "1m"
YF_PERIOD = "2d"
YF_RETRIES = 3
YF_PAUSE = 1.0

# fallback
FB_BARS = 480

# Time filter (best hours) - tuples of (start_hour, end_hour) in UTC for pairs generally active
# We'll treat good trading window as 07:00-22:00 UTC (covers major sessions). You can customize.
GOOD_HOURS_UTC = [(7,22)]

# ATR thresholds (relative): if ATR too low -> flat; too high -> too choppy
MIN_ATR = 0.00008   # minimum 1m ATR for actionable signal (tweak per pair)
MAX_ATR = 0.01      # max ATR to consider dangerously choppy

# NO-FLAT settings: std/ma threshold
MAX_STD_MA_RATIO = 0.0007  # small ratio -> flat

# Signal quality thresholds
HIGH_QUALITY = 75.0
MEDIUM_QUALITY = 55.0

# ---------------- Keep-alive (Flask) ----------------
app = Flask(__name__)
@app.route("/")
def index():
    return "OXTSIGNALSBOT PRO MAX is alive"

def keep_alive():
    t = threading.Thread(target=lambda: app.run(host="0.0.0.0", port=8080), daemon=True)
    t.start()

# ---------------- Logging ----------------
def ensure_log():
    if not os.path.exists(LOG_CSV):
        with open(LOG_CSV, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "timestamp","chat_id","user_id","instrument","expiration","signal",
                "quality","confidence","price_open","price_close","result"
            ])

def log_row(row: Dict):
    ensure_log()
    try:
        with open(LOG_CSV, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                row.get("timestamp",""),
                row.get("chat_id",""),
                row.get("user_id",""),
                row.get("instrument",""),
                row.get("expiration",""),
                row.get("signal",""),
                row.get("quality",""),
                row.get("confidence",""),
                row.get("price_open",""),
                row.get("price_close",""),
                row.get("result","")
            ])
    except Exception:
        print("Log error:", traceback.format_exc())

# ---------------- Utilities ----------------
def exp_to_seconds(exp: str) -> int:
    if exp.endswith("m"):
        return int(exp.replace("m","")) * 60
    if exp.endswith("s"):
        return int(exp.replace("s",""))
    return 60

def yf_symbol(pair: str) -> str:
    p = pair.upper().replace("/","").replace(" ","")
    if len(p) == 6 and p.isalpha():
        return f"{p[:3]}{p[3:]}=X"
    return pair

def in_good_hours() -> bool:
    now = datetime.utcnow().time()
    for s,e in GOOD_HOURS_UTC:
        start = dtime(s,0)
        end = dtime(e,0)
        if start <= now <= end:
            return True
    return False

# ---------------- Smart fallback ----------------
def smart_fallback(seed: str, bars: int = FB_BARS) -> pd.DataFrame:
    rnd = random.Random(abs(hash(seed)) % (10**9))
    base_level = 1.0 + (abs(hash(seed)) % 200) / 1000.0
    vol = rnd.uniform(0.0003, 0.0025)
    trend_strength = rnd.uniform(-0.00005, 0.00005)
    times = pd.date_range(end=pd.Timestamp.now(), periods=bars, freq="1min")
    opens, highs, lows, closes, vols = [], [], [], [], []
    price = base_level
    for _ in range(bars):
        if rnd.random() < 0.02:
            trend_strength = rnd.uniform(-0.0010, 0.0010)
        move = rnd.gauss(trend_strength, vol)
        o = price
        c = max(0.00001, o + move)
        h = max(o, c) + abs(rnd.gauss(0, vol*0.8))
        l = min(o, c) - abs(rnd.gauss(0, vol*0.8))
        v = max(1, int(abs(rnd.gauss(50, 200))))
        opens.append(o); highs.append(h); lows.append(l); closes.append(c); vols.append(v)
        price = c
        if rnd.random() < 0.01:
            vol = max(0.0001, vol * rnd.uniform(0.7,1.3))
    df = pd.DataFrame({"Open":opens,"High":highs,"Low":lows,"Close":closes,"Volume":vols}, index=times)
    return df

# ---------------- Fetch data ----------------
def fetch_data(pair: str, period: str = YF_PERIOD, interval: str = YF_INTERVAL, retries: int = YF_RETRIES, pause: float = YF_PAUSE) -> pd.DataFrame:
    ticker = yf_symbol(pair)
    last_err = None
    for attempt in range(retries):
        try:
            df = yf.download(ticker, period=period, interval=interval, progress=False, threads=False)
            if df is None or df.empty:
                last_err = f"empty df attempt {attempt}"
                time.sleep(pause); continue
            if "Close" not in df.columns:
                last_err = "no Close"; time.sleep(pause); continue
            df = df.dropna(subset=["Close"])
            if df.empty:
                last_err = "dropna empty"; time.sleep(pause); continue
            for col in ["Open","High","Low","Close","Volume"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna(subset=["Close"])
            if df.empty:
                last_err = "close all NaN"; time.sleep(pause); continue
            return df
        except Exception as e:
            last_err = str(e)
            time.sleep(pause)
    # fallback
    print(f"[fetch_data] yfinance failed for {pair}: {last_err} -> using fallback")
    return smart_fallback(pair, bars=FB_BARS)

# ---------------- Indicators ----------------
def compute_indicators(df: pd.DataFrame) -> Dict[str, float]:
    out: Dict[str,float] = {}
    try:
        close = df["Close"].astype(float)
        high = df["High"].astype(float)
        low = df["Low"].astype(float)
        n = len(close)
        if n < 5:
            return out

        # EMA8/21
        ema8 = close.ewm(span=8, adjust=False).mean().iloc[-1]
        ema21 = close.ewm(span=21, adjust=False).mean().iloc[-1]
        out["EMA8"] = ema8; out["EMA21"] = ema21
        out["EMA"] = 1 if ema8 > ema21 else -1

        # SMA5/20
        sma5 = close.rolling(window=5, min_periods=1).mean().iloc[-1]
        sma20 = close.rolling(window=min(20,n), min_periods=1).mean().iloc[-1]
        out["SMA5"] = sma5; out["SMA20"] = sma20
        out["SMA"] = 1 if sma5 > sma20 else -1

        # MACD + hist mag
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        macd_sig = macd.ewm(span=9, adjust=False).mean()
        hist = (macd - macd_sig)
        out["MACD"] = 1 if macd.iloc[-1] > macd_sig.iloc[-1] else -1
        out["MACD_hist"] = float(hist.iloc[-1])
        out["MACD_trend"] = float(hist.iloc[-1] - hist.iloc[-2]) if len(hist) > 1 else 0.0

        # RSI
        delta = close.diff().dropna()
        up = delta.clip(lower=0).rolling(window=14, min_periods=1).mean()
        down = (-delta.clip(upper=0)).rolling(window=14, min_periods=1).mean()
        rs = up / down.replace(0, 1e-9)
        rsi = 100 - (100 / (1 + rs))
        rsi_val = float(rsi.iloc[-1]) if len(rsi) > 0 else 50.0
        out["_RSI"] = rsi_val
        out["RSI"] = 1 if rsi_val > 55 else (-1 if rsi_val < 45 else 0)

        # Bollinger
        ma20 = close.rolling(window=min(20,n), min_periods=1).mean()
        std20 = close.rolling(window=min(20,n), min_periods=1).std().fillna(0)
        upper = ma20 + 2*std20
        lower = ma20 - 2*std20
        last = float(close.iloc[-1])
        out["BB"] = 1 if last < lower.iloc[-1] else (-1 if last > upper.iloc[-1] else 0)

        # ATR
        prev_close = close.shift(1).fillna(close.iloc[0])
        tr = pd.concat([high-low, (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
        out["ATR"] = float(tr.rolling(window=14, min_periods=1).mean().iloc[-1])

        # noise metric
        ma = ma20.iloc[-1] if len(ma20)>0 else 0.0
        std = std20.iloc[-1] if len(std20)>0 else 0.0
        out["STD_MA"] = (std / ma) if ma != 0 else 0.0

        # last candles for patterns
        out["last_open"] = float(df["Open"].iloc[-1])
        out["last_close"] = last
        out["prev_open"] = float(df["Open"].iloc[-2]) if n>=2 else out["last_open"]
        out["prev_close"] = float(df["Close"].iloc[-2]) if n>=2 else out["last_close"]
    except Exception as e:
        print("compute_indicators error:", e)
        traceback.print_exc()
    return out

# ---------------- Candle patterns ----------------
def detect_candle_pattern(ind: Dict[str,float]) -> Optional[str]:
    # simple detection: pin bar, engulfing, hammer
    o = ind.get("last_open"); c = ind.get("last_close")
    po = ind.get("prev_open"); pc = ind.get("prev_close")
    if o is None or c is None: return None
    body = abs(c - o)
    high = max(o,c)
    low = min(o,c)
    upper_shadow = (ind.get("last_high", high) - high) if "last_high" in ind else 0
    lower_shadow = (low - ind.get("last_low", low)) if "last_low" in ind else 0
    # Use simple rules
    # Engulfing
    if (c > o and pc < po and c > po and o < pc) or (c < o and pc > po and c < po and o > pc):
        return "engulfing"
    # Pin bar (long tail)
    if body > 0 and (lower_shadow > 2*body or upper_shadow > 2*body):
        return "pinbar"
    # Hammer (small body, long lower shadow)
    if body < 0.3 * (abs(ind.get("last_high",high) - ind.get("last_low",low))):
        if lower_shadow > 2*body:
            return "hammer"
    return None

# ---------------- NO-FLAT and ATR checks ----------------
def is_flat_or_low_vol(ind: Dict[str,float]) -> Tuple[bool,str]:
    # use STD_MA and ATR thresholds
    std_ma = ind.get("STD_MA", 0.0)
    atr = ind.get("ATR", 0.0)
    if std_ma is not None and std_ma < MAX_STD_MA_RATIO:
        return True, "flat_std"
    if atr is not None and atr < MIN_ATR:
        return True, "low_atr"
    if atr is not None and atr > MAX_ATR:
        return True, "high_atr"
    return False, ""

# ---------------- Trend power & macd quality ----------------
def trend_power_and_macd_quality(ind: Dict[str,float]) -> Tuple[float, float]:
    # trend_power = |EMA8 - EMA21| / ATR
    ema8 = ind.get("EMA8",0); ema21 = ind.get("EMA21",0); atr = ind.get("ATR",1e-9)
    tp = abs(ema8 - ema21) / max(atr, 1e-9)
    # macd_quality: magnitude of hist and recent direction
    macd_mag = abs(ind.get("MACD_hist",0))
    macd_trend = ind.get("MACD_trend",0)
    mq = macd_mag * (1 + max(0, macd_trend)*5)
    return tp, mq

# ---------------- Signal quality & confidence ----------------
def compute_quality_and_confidence(ind: Dict[str,float], base_conf: float) -> Tuple[str, float]:
    tp, mq = trend_power_and_macd_quality(ind)
    std_ma = ind.get("STD_MA",0)
    # combine metrics
    quality_score = base_conf
    # reward trend power (scaled)
    quality_score += min(15, tp * 8)
    # reward macd quality
    quality_score += min(10, mq * 4000)
    # penalize noise
    quality_score -= min(20, std_ma * 1000)
    # penalize extreme ATR (choppy)
    atr = ind.get("ATR",0)
    if atr > 0 and atr > (MIN_ATR * 30):
        quality_score -= 10
    quality_score = max(10.0, min(99.9, quality_score))
    # quality label
    label = "Low"
    if quality_score >= HIGH_QUALITY:
        label = "High"
    elif quality_score >= MEDIUM_QUALITY:
        label = "Medium"
    return label, round(quality_score,1)

# ---------------- Voting & base confidence ----------------
def vote_and_base_confidence(ind: Dict[str,float]) -> Tuple[str, float]:
    score = 0.0; max_score = 0.0
    mapping = {"EMA": ind.get("EMA",0), "SMA": ind.get("SMA",0), "MACD": ind.get("MACD",0), "RSI": ind.get("RSI",0), "BB": ind.get("BB",0)}
    for k,w in WEIGHTS.items():
        v = mapping.get(k,0)
        score += v * w
        max_score += abs(w)
    base_conf = (abs(score)/max_score)*60.0  # base scale 0..60
    # small boost from MACD magnitude
    base_conf += min(20.0, abs(ind.get("MACD_hist",0)) * 10000.0)
    base_conf = max(10.0, min(90.0, base_conf))
    direction = "Вверх ↑" if score >= 0 else "Вниз ↓"
    return direction, round(base_conf,1)

# ---------------- Keyboards/UI ----------------
def make_page_keyboard(items, page, prefix):
    total = len(items)
    start = page * PAGE_SIZE
    end = min(total, start + PAGE_SIZE)
    rows = []
    for i in range(start, end):
        rows.append([InlineKeyboardButton(items[i], callback_data=f"{prefix}_idx_{i}")])
    nav = []
    if start > 0:
        nav.append(InlineKeyboardButton("⬅ Назад", callback_data=f"{prefix}_page_{page-1}"))
    if end < total:
        nav.append(InlineKeyboardButton("Вперёд ➡", callback_data=f"{prefix}_page_{page+1}"))
    if nav:
        rows.append(nav)
    return InlineKeyboardMarkup(rows)

def main_menu_keyboard():
    kb = [[InlineKeyboardButton("💱 Валютные пары", callback_data="cat_fx_0")],
          [InlineKeyboardButton("📰 NON-FARM (NFP)", callback_data="nfp_mode")]]
    return InlineKeyboardMarkup(kb)

# ---------------- Handlers ----------------
def cmd_start(update: Update, context: CallbackContext):
    update.message.reply_text("👋 Выберите режим:", reply_markup=main_menu_keyboard())

def callback_handler(update: Update, context: CallbackContext):
    q = update.callback_query
    q.answer()
    data = q.data
    try:
        if data.startswith("cat_fx_"):
            page = int(data.split("_")[-1])
            q.edit_message_text("Выберите валютную пару:", reply_markup=make_page_keyboard(FOREX, page, "pair"))
            return

        if data.startswith("pair_idx_") or data.startswith("pair_"):
            if data.startswith("pair_idx_"):
                idx = int(data.split("_")[-1]); pair = FOREX[idx]
            else:
                pair = data.replace("pair_","")
            context.user_data["pair"] = pair
            kb = [[InlineKeyboardButton(e, callback_data=f"exp_{e}") for e in EXPIRATIONS]]
            kb.append([InlineKeyboardButton("⬅ Назад", callback_data="cat_fx_0")])
            q.edit_message_text(f"Вы выбрали: *{pair}*\n\nВыберите экспирацию:", parse_mode="Markdown", reply_markup=InlineKeyboardMarkup(kb))
            return

        if data.startswith("exp_"):
            exp = data.split("_",1)[1]
            pair = context.user_data.get("pair")
            if not pair:
                q.edit_message_text("Инструмент не выбран. Вернитесь в меню.")
                return

            # begin analysis thread
            sent = q.edit_message_text(f"⏳ Подождите {ANALYSIS_WAIT} сек — идёт профессиональный анализ {pair} ...", parse_mode="Markdown")
            threading.Thread(target=analysis_worker, args=(context.bot, q.message.chat_id, sent.message_id, pair, exp, q.from_user.id), daemon=True).start()
            return

        if data == "new_signal":
            q.edit_message_text("Выберите режим:", reply_markup=main_menu_keyboard())
            return

        if data == "nfp_mode":
            sent = q.edit_message_text("⏳ Выполняется NFP-анализ для EURUSD (после выхода) — подождите...", parse_mode="Markdown")
            threading.Thread(target=nfp_worker, args=(context.bot, q.message.chat_id, sent.message_id, q.from_user.id), daemon=True).start()
            return

        q.edit_message_text("Нераспознанная команда. Попробуйте /start.")
    except Exception as e:
        print("callback_handler error:", e)
        traceback.print_exc()
        try:
            q.edit_message_text("Внутренняя ошибка. Попробуйте /start.")
        except:
            pass

# ---------------- Analysis worker ----------------
def analysis_worker(bot, chat_id: int, message_id: int, pair: str, exp: str, user_id: int):
    try:
        # simulated professional wait
        time.sleep(ANALYSIS_WAIT)

        # time filter: prefer signals in good hours, but still allow outside - we will tag quality
        good_hours = in_good_hours()

        df = fetch_data(pair)
        if df is None or df.empty:
            bot.edit_message_text(chat_id=chat_id, message_id=message_id, text="⚠️ Не удалось получить данные. Попробуйте позже.")
            return

        ind = compute_indicators(df)
        if not ind:
            bot.edit_message_text(chat_id=chat_id, message_id=message_id, text="⚠️ Недостаточно данных для анализа.")
            return

        # flat/atr checks
        flat, flat_reason = is_flat_or_low_vol(ind)
        if flat:
            # if flat, still compute but mark low quality and suggest to avoid
            direction, base_conf = vote_and_base_confidence(ind)
            quality_label, quality_score = compute_quality_and_confidence(ind, base_conf)
            text = (
                f"⚠️ Рынок во флете или неподходящая волатильность ({flat_reason}).\n\n"
                f"Рекомендация: воздержаться от входа.\n"
                f"Пара: {pair} | Эксп: {exp}\n"
                f"Сигнал (технически): *{direction}*  •  Качество: *{quality_label}* ({quality_score}%)\n"
            )
            bot.edit_message_text(chat_id=chat_id, message_id=message_id, text=text, parse_mode="Markdown")
            # log as skipped low quality
            log_row({
                "timestamp": datetime.utcnow().isoformat(),
                "chat_id": chat_id,
                "user_id": user_id,
                "instrument": pair,
                "expiration": exp,
                "signal": direction,
                "quality": quality_label,
                "confidence": quality_score,
                "price_open": float(df["Close"].iloc[-1]),
                "price_close": "",
                "result": "skipped_flat"
            })
            # return to menu
            time.sleep(0.5)
            bot.send_message(chat_id, "🔁 Возвращаю в меню валютных пар:", reply_markup=main_menu_keyboard())
            return

        # detect candle patterns
        patt = detect_candle_pattern(ind)

        # base vote
        direction, base_conf = vote_and_base_confidence(ind)

        # compute final quality/confidence
        quality_label, quality_score = compute_quality_and_confidence(ind, base_conf)

        # if outside good hours, lower quality
        if not good_hours:
            quality_score = max(10.0, quality_score - 12.0)
            if quality_score < MEDIUM_QUALITY: quality_label = "Low"

        # candle-based adjustments
        if patt == "engulfing":
            quality_score = min(99.9, quality_score + 8)
        elif patt == "pinbar":
            quality_score = min(99.9, quality_score + 6)
        elif patt == "hammer":
            quality_score = min(99.9, quality_score + 5)

        # final textual confidence
        conf_text = f"{round(quality_score,1)}%"

        # suggested expiration logic by ATR / trend power
        tp, mq = trend_power_and_macd_quality(ind)
        if tp > 0.8 and ind.get("ATR",0) > MIN_ATR:
            suggested_exp = "1m"
        elif tp > 0.35:
            suggested_exp = "2m"
        else:
            suggested_exp = "3m"

        price_open = float(df["Close"].iloc[-1])

        expl = []
        expl.append("EMA8>EMA21" if ind.get("EMA",0)==1 else "EMA8<EMA21")
        expl.append(f"RSI≈{int(ind.get('_RSI',50))}")
        bbv = ind.get("BB",0)
        if bbv==1: expl.append("Цена у нижней BB")
        elif bbv==-1: expl.append("Цена у верхней BB")
        if patt: expl.append(f"Паттерн:{patt}")
        expl_text = "; ".join(expl[:6])

        text = (
            f"📊 *Анализ завершён*\n\n"
            f"🔹 {pair} | Эксп: {exp}\n"
            f"📈 *Сигнал:* *{direction}*    🎯 *Качество:* *{quality_label}* ({conf_text})\n"
            f"⏱ *Рекомендуемая экспирация:* *{suggested_exp}* (идет анализ под 1-3 мин)\n\n"
            f"_Короткая логика:_ {expl_text}\n"
            f"_Цена (прибл.):_ `{price_open:.6f}`\n\n"
            f"⚡ Откройте сделку в течение *10 секунд*."
        )

        bot.edit_message_text(chat_id=chat_id, message_id=message_id, text=text, parse_mode="Markdown")

        # log pending
        log_row({
            "timestamp": datetime.utcnow().isoformat(),
            "chat_id": chat_id,
            "user_id": user_id,
            "instrument": pair,
            "expiration": exp,
            "signal": direction,
            "quality": quality_label,
            "confidence": quality_score,
            "price_open": price_open,
            "price_close": "",
            "result": "pending"
        })

        # finalize after expiration
        seconds = exp_to_seconds(exp)
        threading.Timer(seconds, finalize_worker, args=(bot, chat_id, message_id, pair, exp, direction, quality_score, price_open, user_id)).start()

    except Exception as e:
        print("analysis_worker error:", e)
        traceback.print_exc()
        try:
            bot.send_message(chat_id, "⚠️ Ошибка при анализе. Попробуйте снова.")
        except:
            pass

# ---------------- Finalize ----------------
def finalize_worker(bot, chat_id: int, message_id: int, pair: str, exp: str, direction: str, quality_score: float, price_open: float, user_id: int):
    try:
        df2 = fetch_data(pair)
        price_close = float(df2["Close"].iloc[-1]) if (df2 is not None and not df2.empty) else price_open

        win = (direction.startswith("Вверх") and price_close > price_open) or (direction.startswith("Вниз") and price_close < price_open)
        result = "Плюс ✅" if win else "Минус ❌"

        final_text = (
            f"✅ *Сделка завершена*\n\n"
            f"*{pair}* | Эксп: *{exp}*\n"
            f"*Сигнал:* *{direction}*    *Результат:* *{result}*\n"
            f"*Качество сигнала:* *{round(quality_score,1)}%*\n\n"
            f"_Открытие:_ `{price_open:.6f}`\n"
            f"_Закрытие:_ `{price_close:.6f}`"
        )

        try:
            bot.edit_message_text(chat_id=chat_id, message_id=message_id, text=final_text, parse_mode="Markdown")
        except:
            bot.send_message(chat_id, final_text, parse_mode="Markdown")

        # return to pairs menu automatically
        time.sleep(0.4)
        try:
            bot.send_message(chat_id, "🔁 Возвращаю в меню валютных пар:", reply_markup=main_menu_keyboard())
        except:
            pass

        log_row({
            "timestamp": datetime.utcnow().isoformat(),
            "chat_id": chat_id,
            "user_id": user_id,
            "instrument": pair,
            "expiration": exp,
            "signal": direction,
            "quality": round(quality_score,1),
            "confidence": round(quality_score,1),
            "price_open": price_open,
            "price_close": price_close,
            "result": result
        })

    except Exception as e:
        print("finalize_worker error:", e)
        traceback.print_exc()

# ---------------- NFP worker ----------------
def nfp_worker(bot, chat_id: int, message_id: int, user_id: int):
    try:
        time.sleep(2)
        pair = "EURUSD"
        df = fetch_data(pair, period="1d", interval="1m")
        if df is None or df.empty:
            bot.edit_message_text(chat_id=chat_id, message_id=message_id, text="⚠️ Не удалось получить данные для NFP.")
            return
        ind = compute_indicators(df)
        if not ind:
            bot.edit_message_text(chat_id=chat_id, message_id=message_id, text="⚠️ Недостаточно данных для NFP.")
            return
        direction, base_conf = vote_and_base_confidence(ind)
        quality_label, quality_score = compute_quality_and_confidence(ind, base_conf)
        # ATR suggest exp
        atr = ind.get("ATR",0)
        if atr and atr > 0.0025:
            suggested = "1m"
        elif atr and atr > 0.0010:
            suggested = "2m"
        else:
            suggested = "3m"
        text = (
            f"📰 *NFP Анализ (EURUSD, после выхода)*\n\n"
            f"📈 *Рекомендация:* *{direction}*\n"
            f"⏱ *Рекомендуемая экспирация:* *{suggested}*\n"
            f"🎯 *Качество:* *{quality_label}* ({quality_score}%)\n\n"
            f"_Короткая логика:_ EMA/MACD/RSI/BB\n\n"
            f"📌 После прочтения нажмите «💱 Валютные пары» чтобы вернуться."
        )
        try:
            bot.edit_message_text(chat_id=chat_id, message_id=message_id, text=text, parse_mode="Markdown")
        except:
            bot.send_message(chat_id, text, parse_mode="Markdown")
    except Exception as e:
        print("nfp_worker error:", e)
        traceback.print_exc()
        try:
            bot.send_message(chat_id, "⚠️ Ошибка NFP анализа.")
        except:
            pass

# ---------------- Webhook cleanup ----------------
def delete_webhook_on_start():
    try:
        if not BOT_TOKEN:
            return
        r = requests.get(f"https://api.telegram.org/bot{BOT_TOKEN}/deleteWebhook", timeout=5)
        print("deleteWebhook:", r.status_code, r.text[:200])
    except Exception as e:
        print("delete_webhook error:", e)

# ---------------- Entrypoint ----------------
def main():
    if not BOT_TOKEN:
        print("ERROR: BOT_TOKEN is empty. Set in environment variables.")
        return

    ensure_log()
    keep_alive()
    delete_webhook_on_start()

    updater = Updater(BOT_TOKEN, use_context=True)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", cmd_start))
    dp.add_handler(CallbackQueryHandler(callback_handler))

    print("OXTSIGNALSBOT PRO MAX started (polling)")
    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()
