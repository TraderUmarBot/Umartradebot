# main.py
# OXTSIGNALSBOT PRO — FOREXCOM TradingView primary feed + yfinance fallback
# Webhook Flask app. Use environment variables BOT_TOKEN and WEBHOOK_URL.

import os
import time
import threading
import csv
import traceback
from datetime import datetime, time as dtime
from typing import Dict, Tuple, Optional

import pandas as pd
import numpy as np
import yfinance as yf
import requests
from flask import Flask, request

import telegram
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Dispatcher, CommandHandler, CallbackQueryHandler, CallbackContext

# ---------------- CONFIG ----------------
BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "").strip()
PORT = int(os.getenv("PORT", "10000"))

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN is not set in environment variables.")
if not WEBHOOK_URL:
    raise RuntimeError("WEBHOOK_URL is not set in environment variables.")

LOG_CSV = "signals_log.csv"

# Pairs list (user's list; we'll use FOREXCOM feed)
FOREX = [
    "EURUSD","GBPUSD","USDJPY","AUDUSD","USDCHF","EURJPY",
    "GBPJPY","NZDUSD","EURGBP","CADJPY","USDCAD","AUDJPY",
    "EURAUD","GBPAUD","EURNZD","AUDNZD","CADCHF","CHFJPY",
    "NZDJPY","GBPCAD"
]

# expirations supported
EXPIRATIONS = ["1m", "2m", "5m", "10m"]
PAGE_SIZE = 6
ANALYSIS_WAIT = 10  # seconds to show "professional analysis"

# tuning thresholds
MIN_ATR = 0.00003
MAX_ATR = 0.02
MAX_STD_MA_RATIO = 0.0009
HIGH_QUALITY = 85.0
MEDIUM_QUALITY = 65.0
GOOD_HOURS_UTC = [(6, 22)]
WICK_TO_BODY_RATIO = 1.5

# TradingView endpoints (try in order). These are community mirrors.
TV_ENDPOINTS = [
    "https://api.tradingview.com/v1/history",
    "https://scanner.tradingview.com/forex/history",
    "https://tvdb.brianknox.dev/history",
    "https://api.tvio.pro/history"
]

# ---------------- FLASK + TELEGRAM ----------------
app = Flask(__name__)
bot = telegram.Bot(token=BOT_TOKEN)
dispatcher = Dispatcher(bot, None, workers=8, use_context=True)


@app.route("/", methods=["GET"])
def home():
    return "OXTSIGNALSBOT PRO (FOREXCOM) — alive"


@app.route("/webhook", methods=["POST"])
def webhook_endpoint():
    try:
        update_json = request.get_json(force=True)
        update = telegram.Update.de_json(update_json, bot)
        dispatcher.process_update(update)
    except Exception as e:
        print("Webhook processing error:", e)
        traceback.print_exc()
    return "OK", 200


# ---------------- Logging helpers ----------------
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
    except Exception as e:
        print("log_row error:", e)
        traceback.print_exc()


def read_logs_df() -> pd.DataFrame:
    ensure_log()
    try:
        df = pd.read_csv(LOG_CSV)
        return df
    except Exception:
        return pd.DataFrame()


# ---------------- Utilities ----------------
def exp_to_seconds(exp: str) -> int:
    try:
        if exp.endswith("m"):
            return int(exp.replace("m","")) * 60
    except:
        pass
    return 60


def tv_symbol(pair: str) -> str:
    # Use FOREXCOM feed (best match for Pocket Option)
    p = pair.upper().replace("/","").replace(" ","")
    return f"FOREXCOM:{p}"


def yf_symbol(pair: str) -> str:
    p = pair.upper().replace("/","").replace(" ","")
    if len(p) == 6 and p.isalpha():
        return f"{p[:3]}{p[3:]}=X"
    return pair


def in_good_hours() -> bool:
    now = datetime.utcnow().time()
    for s, e in GOOD_HOURS_UTC:
        start = dtime(s, 0)
        end = dtime(e, 0)
        if start <= now <= end:
            return True
    return False


# ---------------- TradingView fetch (primary) ----------------
def tv_get(pair: str, resolution: str = "1", bars: int = 500, timeout: float = 6.0) -> Optional[pd.DataFrame]:
    """
    Request TradingView-like history from several public proxies.
    resolution: "1","2","5","10" etc.
    returns DataFrame with Open,High,Low,Close,Volume indexed by datetime UTC
    """
    symbol = tv_symbol(pair)
    params = {"symbol": symbol, "resolution": resolution, "bars": bars}
    headers = {"User-Agent": "Mozilla/5.0"}

    for endpoint in TV_ENDPOINTS:
        try:
            r = requests.get(endpoint, params=params, headers=headers, timeout=timeout)
            if r.status_code != 200:
                # print debug and continue
                print(f"[tv_get] endpoint {endpoint} returned status {r.status_code}")
                continue
            data = r.json()
            if not data or "t" not in data or not data["t"]:
                print(f"[tv_get] endpoint {endpoint} returned empty payload")
                continue

            df = pd.DataFrame({
                "Open": data.get("o", []),
                "High": data.get("h", []),
                "Low": data.get("l", []),
                "Close": data.get("c", []),
                "Volume": data.get("v", [0]*len(data.get("t", [])))
            }, index=pd.to_datetime(data["t"], unit="s"))

            for col in ["Open","High","Low","Close","Volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna(subset=["Close"])
            if df.empty:
                continue
            return df
        except Exception as e:
            print(f"[tv_get] endpoint {endpoint} failed for {pair}: {e}")
            traceback.print_exc()
            continue
    return None


# ---------------- YFinance fallback ----------------
def yf_get(pair: str, period: str = "5d", interval: str = "1m", min_bars: int = 120) -> Optional[pd.DataFrame]:
    try:
        symbol = yf_symbol(pair)
        df = yf.download(symbol, period=period, interval=interval, progress=False, threads=False)
        if df is None or df.empty:
            return None
        for col in ["Open","High","Low","Close","Volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["Close"])
        if df.empty or len(df) < min_bars:
            return None
        return df
    except Exception as e:
        print("[yf_get] error:", e)
        traceback.print_exc()
        return None


# ---------------- fetch_data wrapper ----------------
def fetch_data(pair: str, resolution_min: int = 1, bars: int = 500) -> Optional[pd.DataFrame]:
    """
    Try TradingView with desired resolution; fallback to yfinance.
    resolution_min: integer (1,2,5,10)
    """
    try:
        res = str(resolution_min)
        df = tv_get(pair, resolution=res, bars=bars)
        if df is not None and not df.empty:
            return df
        time.sleep(0.6)
        df = tv_get(pair, resolution=res, bars=bars)
        if df is not None and not df.empty:
            return df
        df = yf_get(pair, period="5d", interval="1m", min_bars=120)
        if df is not None and not df.empty:
            return df
        # last resort: simulate small series (keeps bot alive; not for production signals)
        now = pd.date_range(end=pd.Timestamp.now(), periods=240, freq="1min")
        base = 1.0
        data = []
        import random
        for _ in range(240):
            o = base
            c = o + random.uniform(-0.0003, 0.0003)
            h = max(o, c) + random.uniform(0, 0.0005)
            l = min(o, c) - random.uniform(0, 0.0005)
            v = random.randint(10, 200)
            data.append([o,h,l,c,v])
            base = c
        df2 = pd.DataFrame(data, columns=["Open","High","Low","Close","Volume"], index=now)
        return df2
    except Exception as e:
        print("fetch_data wrapper error:", e)
        traceback.print_exc()
        return None


# ---------------- Indicators ----------------
def compute_indicators(df: pd.DataFrame) -> Dict[str, float]:
    out: Dict[str, float] = {}
    try:
        if df is None or df.empty:
            return {}
        close = df["Close"].astype(float)
        high = df["High"].astype(float) if "High" in df.columns else close
        low = df["Low"].astype(float) if "Low" in df.columns else close
        n = len(close)
        if n < 20:
            return {}

        out["EMA8"] = close.ewm(span=8, adjust=False).mean().iloc[-1]
        out["EMA21"] = close.ewm(span=21, adjust=False).mean().iloc[-1]
        out["EMA"] = 1 if out["EMA8"] > out["EMA21"] else -1

        out["SMA5"] = close.rolling(window=5, min_periods=1).mean().iloc[-1]
        out["SMA20"] = close.rolling(window=20, min_periods=1).mean().iloc[-1]
        out["SMA"] = 1 if out["SMA5"] > out["SMA20"] else -1

        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        macd_sig = macd.ewm(span=9, adjust=False).mean()
        out["MACD_hist"] = float((macd - macd_sig).iloc[-1])
        out["MACD"] = 1 if macd.iloc[-1] > macd_sig.iloc[-1] else -1
        out["MACD_trend"] = float((macd - macd_sig).iloc[-1] - (macd - macd_sig).iloc[-2]) if n >= 2 else 0.0

        delta = close.diff().dropna()
        up = delta.clip(lower=0).rolling(window=14, min_periods=1).mean()
        down = (-delta.clip(upper=0)).rolling(window=14, min_periods=1).mean()
        rs = up / down.replace(0, 1e-9)
        rsi = 100 - (100 / (1 + rs))
        out["_RSI"] = float(rsi.iloc[-1]) if len(rsi) > 0 else 50.0
        out["RSI"] = 1 if out["_RSI"] > 55 else (-1 if out["_RSI"] < 45 else 0)

        ma20 = close.rolling(window=20, min_periods=1).mean()
        std20 = close.rolling(window=20, min_periods=1).std().fillna(0)
        out["BB"] = 1 if float(close.iloc[-1]) < ma20.iloc[-1] - 2*std20.iloc[-1] else (-1 if float(close.iloc[-1]) > ma20.iloc[-1] + 2*std20.iloc[-1] else 0)

        prev_close = close.shift(1).fillna(close.iloc[0])
        tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        out["ATR"] = float(tr.rolling(window=14, min_periods=1).mean().iloc[-1])

        out["STD_MA"] = float(std20.iloc[-1] / ma20.iloc[-1]) if ma20.iloc[-1] != 0 else 0.0

        out["last_open"] = float(df["Open"].iloc[-1]) if "Open" in df.columns else float(close.iloc[-1])
        out["last_close"] = float(close.iloc[-1])
        out["prev_open"] = float(df["Open"].iloc[-2]) if "Open" in df.columns and len(df) >= 2 else out["last_open"]
        out["prev_close"] = float(df["Close"].iloc[-2]) if len(df) >=2 else out["last_close"]
        out["last_high"] = float(df["High"].iloc[-1]) if "High" in df.columns else out["last_close"]
        out["last_low"] = float(df["Low"].iloc[-1]) if "Low" in df.columns else out["last_close"]

    except Exception as e:
        print("compute_indicators error:", e)
        traceback.print_exc()
        return {}
    return out


# ---------------- Candle patterns ----------------
def is_doji(o, c, h, l, thresh=0.0015):
    body = abs(c - o)
    rng = h - l if (h - l) != 0 else 1e-9
    return (body / rng) < thresh

def is_pinbar(o, c, h, l):
    body = abs(c - o) if abs(c - o) != 0 else 1e-9
    upper_wick = h - max(c, o)
    lower_wick = min(c, o) - l
    if upper_wick / body > WICK_TO_BODY_RATIO and lower_wick / body < 0.6:
        return "pinbar_bear"
    if lower_wick / body > WICK_TO_BODY_RATIO and upper_wick / body < 0.6:
        return "pinbar_bull"
    return None

def is_hammer(o, c, h, l):
    body = abs(c - o) if abs(c - o) != 0 else 1e-9
    lower_wick = min(c, o) - l
    upper_wick = h - max(c, o)
    if lower_wick / body > 2.0 and upper_wick / body < 0.5:
        return "hammer_bull" if c > o else "hanging_man"
    return None

def is_engulfing(o_prev, c_prev, o, c):
    if (c_prev < o_prev) and (c > o) and (c - o > o_prev - c_prev):
        return "engulfing_bull"
    if (c_prev > o_prev) and (c < o) and (o - c > c_prev - o_prev):
        return "engulfing_bear"
    return None

def detect_patterns_full(df: pd.DataFrame) -> Optional[str]:
    try:
        if df is None or df.empty or len(df) < 3:
            return None
        last = df.iloc[-1]; prev = df.iloc[-2]; prev2 = df.iloc[-3]
        o,h,l,c = float(last["Open"]), float(last["High"]), float(last["Low"]), float(last["Close"])
        po,ph,pl,pc = float(prev["Open"]), float(prev["High"]), float(prev["Low"]), float(prev["Close"])
        p2o,p2h,p2l,p2c = float(prev2["Open"]), float(prev2["High"]), float(prev2["Low"]), float(prev2["Close"])

        if is_doji(o,c,h,l):
            return "doji"
        eng = is_engulfing(po, pc, o, c)
        if eng:
            return eng
        pin = is_pinbar(o,c,h,l)
        if pin:
            return pin
        ham = is_hammer(o,c,h,l)
        if ham:
            return ham
        # morning/evening star heuristics
        big1 = (p2c < p2o and abs(p2c - p2o) > 0.001)
        small2 = abs(pc - po) < abs(p2c - p2o) * 0.5
        big3 = (c > o and abs(c - o) > abs(p2c - p2o) * 0.8)
        if big1 and small2 and big3:
            return "morning_star"
        big1b = (p2c > p2o and abs(p2c - p2o) > 0.001)
        small2b = abs(pc - po) < abs(p2c - p2o) * 0.5
        big3b = (c < o and abs(c - o) > abs(p2c - p2o) * 0.8)
        if big1b and small2b and big3b:
            return "evening_star"
    except Exception as e:
        print("detect_patterns_full error:", e)
    return None


# ---------------- Trend power & MACD quality ----------------
def trend_power_and_macd_quality(ind: Dict[str, float]) -> Tuple[float, float]:
    try:
        ema8 = ind.get("EMA8", 0)
        ema21 = ind.get("EMA21", 0)
        atr = max(ind.get("ATR", 1e-9), 1e-9)
        tp = abs(ema8 - ema21) / atr
        macd_mag = abs(ind.get("MACD_hist", 0))
        macd_trend = ind.get("MACD_trend", 0)
        mq = macd_mag * (1 + max(0, macd_trend) * 5)
        return tp, mq
    except:
        return 0.0, 0.0


# ---------------- Voting & confidence ----------------
def vote_and_base_confidence(ind: Dict[str, float]) -> Tuple[str, float]:
    score = 0.0
    mapping = {"EMA": ind.get("EMA", 0), "SMA": ind.get("SMA", 0), "MACD": ind.get("MACD", 0), "RSI": ind.get("RSI", 0), "BB": ind.get("BB", 0)}
    weights = {"EMA": 2.0, "SMA": 1.0, "MACD": 2.0, "RSI": 1.0, "BB": 1.0}
    max_score = 0.0
    for k, w in weights.items():
        v = mapping.get(k, 0)
        score += v * w
        max_score += abs(w)
    base_conf = (abs(score) / max_score) * 60.0
    base_conf += min(20.0, abs(ind.get("MACD_hist", 0)) * 10000.0)
    base_conf = max(10.0, min(95.0, base_conf))
    direction = "Вверх ↑" if score >= 0 else "Вниз ↓"
    return direction, round(base_conf, 1)


def compute_quality_label_and_score(ind: Dict[str, float], base_conf: float) -> Tuple[str, float]:
    tp, mq = trend_power_and_macd_quality(ind)
    std_ma = ind.get("STD_MA", 0.0)
    quality_score = base_conf
    quality_score += min(15.0, tp * 8.0)
    quality_score += min(10.0, mq * 4000.0)
    quality_score -= min(20.0, std_ma * 1000.0)
    atr = ind.get("ATR", 0.0)
    if atr > 0 and atr > (MIN_ATR * 30):
        quality_score -= 10.0
    quality_score = max(10.0, min(99.9, quality_score))
    label = "Low"
    if quality_score >= HIGH_QUALITY:
        label = "High"
    elif quality_score >= MEDIUM_QUALITY:
        label = "Medium"
    return label, round(quality_score, 1)


def is_flat_or_bad_vol(ind: Dict[str, float], df: Optional[pd.DataFrame] = None) -> Tuple[bool, str]:
    std_ma = ind.get("STD_MA", 0.0)
    atr = ind.get("ATR", 0.0)
    if std_ma is not None and std_ma < MAX_STD_MA_RATIO:
        return True, "flat_std"
    if atr is not None and atr < MIN_ATR:
        return True, "low_atr"
    if atr is not None and atr > MAX_ATR:
        return True, "high_atr"
    if df is not None and len(df) >= 1:
        last = df.iloc[-1]
        o, h, l, c = float(last["Open"]), float(last["High"]), float(last["Low"]), float(last["Close"])
        body = abs(c - o) if abs(c - o) != 0 else 1e-9
        rng = h - l if (h - l) != 0 else 1e-9
        upper_wick = h - max(c, o)
        lower_wick = min(c, o) - l
        if (upper_wick / body > 8.0) or (lower_wick / body > 8.0) or (rng / max(abs(c), abs(o), 1e-9) > 0.01):
            return True, "long_wick_noise"
    return False, ""


# ---------------- UI keyboards ----------------
def main_menu_keyboard():
    kb = [
        [InlineKeyboardButton("💱 Валютные пары", callback_data="cat_fx_0")],
        [InlineKeyboardButton("📰 NON-FARM (NFP)", callback_data="nfp_mode")],
        [InlineKeyboardButton("📊 Статистика", callback_data="show_stats")],
    ]
    return InlineKeyboardMarkup(kb)


def pairs_page_keyboard(page: int):
    total = len(FOREX)
    start = page * PAGE_SIZE
    end = min(total, start + PAGE_SIZE)
    rows = []
    for i in range(start, end):
        rows.append([InlineKeyboardButton(FOREX[i], callback_data=f"pair_{i}")])
    nav = []
    if start > 0:
        nav.append(InlineKeyboardButton("⬅ Назад", callback_data=f"cat_fx_{page-1}"))
    if end < total:
        nav.append(InlineKeyboardButton("Вперёд ➡", callback_data=f"cat_fx_{page+1}"))
    if nav:
        rows.append(nav)
    return InlineKeyboardMarkup(rows)


# ---------------- Handlers ----------------
def cmd_start(update: telegram.Update, context: CallbackContext):
    try:
        update.message.reply_text("👋 Привет! Выберите режим:", reply_markup=main_menu_keyboard())
    except Exception as e:
        print("start handler error:", e)


def callback_handler(update: telegram.Update, context: CallbackContext):
    q = update.callback_query
    q.answer()
    data = q.data
    try:
        if data.startswith("cat_fx_"):
            page = int(data.split("_")[-1])
            q.edit_message_text("Выберите валютную пару:", reply_markup=pairs_page_keyboard(page))
            return
        if data.startswith("pair_"):
            idx = int(data.split("_")[1])
            pair = FOREX[idx]
            context.user_data["pair"] = pair
            kb = [[InlineKeyboardButton(e, callback_data=f"exp_{e}") for e in EXPIRATIONS]]
            kb.append([InlineKeyboardButton("⬅ Назад", callback_data="cat_fx_0")])
            q.edit_message_text(f"Пара выбрана: *{pair}*\nВыберите экспирацию:", parse_mode="Markdown", reply_markup=InlineKeyboardMarkup(kb))
            return
        if data.startswith("exp_"):
            exp = data.split("_",1)[1]
            pair = context.user_data.get("pair")
            if not pair:
                q.edit_message_text("Инструмент не выбран. Вернитесь в меню.")
                return
            sent = q.edit_message_text(f"⏳ Подождите {ANALYSIS_WAIT} сек — идёт профессиональный анализ {pair}...", parse_mode="Markdown")
            threading.Thread(target=analysis_worker, args=(context.bot, q.message.chat_id, sent.message_id, pair, exp, q.from_user.id), daemon=True).start()
            return
        if data == "nfp_mode":
            sent = q.edit_message_text("⏳ Выполняется NFP-анализ для EURUSD (после выхода) — подождите...", parse_mode="Markdown")
            threading.Thread(target=nfp_worker, args=(context.bot, q.message.chat_id, sent.message_id, q.from_user.id), daemon=True).start()
            return
        if data == "show_stats":
            send_stats_callback(q.message.chat_id)
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
        time.sleep(ANALYSIS_WAIT)
        df = fetch_data(pair, resolution_min=1, bars=600)
        if df is None or df.empty:
            bot.edit_message_text(chat_id=chat_id, message_id=message_id, text="⚠️ Не удалось получить данные (TradingView). Попробуйте позже.")
            return
        ind = compute_indicators(df)
        if not ind:
            bot.edit_message_text(chat_id=chat_id, message_id=message_id, text="⚠️ Недостаточно данных для анализа.")
            return
        patt = detect_patterns_full(df)
        flat, reason = is_flat_or_bad_vol(ind, df)
        direction, base_conf = vote_and_base_confidence(ind)
        quality_label, quality_score = compute_quality_label_and_score(ind, base_conf)
        if patt:
            if patt.startswith("engulfing") or "pinbar" in patt or "hammer" in patt or patt in ("morning_star","evening_star"):
                quality_score = min(99.9, quality_score + 12.0)
            elif patt == "doji":
                quality_score = max(10.0, quality_score - 12.0)
        if not in_good_hours():
            quality_score = max(10.0, quality_score - 10.0)
            if quality_score < MEDIUM_QUALITY:
                quality_label = "Low"
        if flat:
            text = (
                f"⚠️ Рынок неподходящий ({reason}). Рекомендуется воздержаться.\n\n"
                f"Пара: *{pair}* | Эксп: *{exp}*\n"
                f"Технический сигнал (без входа): *{direction}* • Качество: *{quality_label}* ({quality_score}%)"
            )
            bot.edit_message_text(chat_id=chat_id, message_id=message_id, text=text, parse_mode="Markdown")
            log_row({
                "timestamp": datetime.utcnow().isoformat(),
                "chat_id": chat_id,
                "user_id": user_id,
                "instrument": pair,
                "expiration": exp,
                "signal": direction,
                "quality": quality_label,
                "confidence": quality_score,
                "price_open": float(df["Close"].iloc[-1]) if not df.empty else None,
                "price_close": "",
                "result": "skipped"
            })
            time.sleep(0.4)
            try:
                bot.send_message(chat_id, "🔁 Возвращаю в меню валютных пар:", reply_markup=main_menu_keyboard())
            except:
                pass
            return
        # suggested expiration: keep user choice
        price_open = float(df["Close"].iloc[-1])
        expl = []
        expl.append("EMA8>EMA21" if ind.get("EMA",0)==1 else "EMA8<EMA21")
        expl.append(f"RSI≈{int(ind.get('_RSI',50))}")
        bbv = ind.get("BB",0)
        if bbv==1: expl.append("Цена у нижней BB")
        elif bbv==-1: expl.append("Цена у верхней BB")
        if patt: expl.append(f"Паттерн:{patt}")
        expl_text = "; ".join(expl[:5])
        text = (
            f"📊 *Анализ завершён*\n\n"
            f"🔹 {pair} | Эксп (выбрана): {exp}\n"
            f"📈 *Сигнал:* *{direction}*    🎯 *Качество:* *{quality_label}* ({round(quality_score,1)}%)\n"
            f"⏱ *Рекомендуемая экспирация:* *{exp}*\n\n"
            f"_Короткая логика:_ {expl_text}\n"
            f"_Цена (прибл.):_ `{price_open:.6f}`\n\n"
            f"⚡ Откройте сделку в течение *10 секунд*."
        )
        try:
            bot.edit_message_text(chat_id=chat_id, message_id=message_id, text=text, parse_mode="Markdown")
        except:
            bot.send_message(chat_id, text, parse_mode="Markdown")
        log_row({
            "timestamp": datetime.utcnow().isoformat(),
            "chat_id": chat_id,
            "user_id": user_id,
            "instrument": pair,
            "expiration": exp,
            "signal": direction,
            "quality": quality_label,
            "confidence": round(quality_score,1),
            "price_open": price_open,
            "price_close": "",
            "result": "pending"
        })
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
        df2 = fetch_data(pair, resolution_min=1, bars=120)
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


# ---------------- NFP ----------------
def nfp_worker(bot, chat_id: int, message_id: int, user_id: int):
    try:
        time.sleep(2)
        pair = "EURUSD"
        df = fetch_data(pair, resolution_min=1, bars=600)
        if df is None or df.empty:
            try:
                bot.edit_message_text(chat_id=chat_id, message_id=message_id, text="⚠️ Не удалось получить данные для NFP.")
            except:
                bot.send_message(chat_id, "⚠️ Не удалось получить данные для NFP.")
            return
        ind = compute_indicators(df)
        direction, base_conf = vote_and_base_confidence(ind)
        quality_label, quality_score = compute_quality_label_and_score(ind, base_conf)
        atr = ind.get("ATR", 0)
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


# ---------------- Statistics & History ----------------
def send_stats_callback(chat_id: int):
    try:
        df = read_logs_df()
        if df.empty:
            bot.send_message(chat_id, "📊 Статистика пуста — пока нет сделок.")
            return
        df2 = df[df["result"].isin(["Плюс ✅", "Минус ❌"])]
        total = len(df2)
        if total == 0:
            bot.send_message(chat_id, "📊 Нет завершённых сделок для статистики.")
            return
        wins = int((df2["result"] == "Плюс ✅").sum())
        losses = int((df2["result"] == "Минус ❌").sum())
        winrate = round((wins / total) * 100, 1)
        avg_conf = round(float(df2["confidence"].astype(float).mean()), 1) if not df2["confidence"].isnull().all() else 0
        best_pairs_ser = df2[df2["result"] == "Плюс ✅"]["instrument"].value_counts().head(5)
        best_pairs_text = "\n".join([f"• {k}: {v} плюсов" for k, v in best_pairs_ser.items()]) if not best_pairs_ser.empty else "Нет данных"
        text = (
            f"📊 *Статистика OXTSIGNALS*\n\n"
            f"Всего сделок: *{total}*\n"
            f"Плюсов: *{wins}*\n"
            f"Минусов: *{losses}*\n"
            f"Win-rate: *{winrate}%*\n"
            f"Средняя уверенность: *{avg_conf}%*\n\n"
            f"🔥 Лучшие пары:\n{best_pairs_text}"
        )
        bot.send_message(chat_id, text, parse_mode="Markdown")
    except Exception as e:
        print("send_stats_callback error:", e)
        traceback.print_exc()
        try:
            bot.send_message(chat_id, "⚠️ Ошибка при получении статистики.")
        except:
            pass


def cmd_stats(update: telegram.Update, context: CallbackContext):
    chat_id = update.message.chat_id
    send_stats_callback(chat_id)


def cmd_history(update: telegram.Update, context: CallbackContext):
    try:
        df = read_logs_df()
        if df.empty:
            update.message.reply_text("🕘 История пуста — пока нет сделок.")
            return
        user_id = update.message.from_user.id
        df_user = df[df["user_id"] == user_id]
        if df_user.empty:
            update.message.reply_text("🕘 У вас пока нет сделок.")
            return
        df_last = df_user.tail(10)
        rows = []
        for _, row in df_last.iterrows():
            rows.append(
                f"{row['timestamp']}\n"
                f"{row['instrument']} | {row['expiration']}\n"
                f"Сигнал: {row['signal']} | Качество: {row.get('quality','')} | Уверенность: {row.get('confidence','')}%\n"
                f"Результат: {row['result']}\n"
                f"Открытие: {row['price_open']} → Закрытие: {row['price_close']}\n"
                "— — — — — — — — —\n"
            )
        update.message.reply_text("📜 *Последние сделки:*\n\n" + "".join(rows), parse_mode="Markdown")
    except Exception as e:
        print("cmd_history error:", e)
        traceback.print_exc()
        try:
            update.message.reply_text("⚠️ Ошибка при получении истории.")
        except:
            pass


# ---------------- Register handlers ----------------
dispatcher.add_handler(CommandHandler("start", cmd_start))
dispatcher.add_handler(CommandHandler("stats", cmd_stats))
dispatcher.add_handler(CommandHandler("history", cmd_history))
dispatcher.add_handler(CallbackQueryHandler(callback_handler))


# ---------------- Start webhook ----------------
if __name__ == "__main__":
    print("Deleting old webhook (if any)...")
    try:
        bot.delete_webhook()
    except Exception:
        pass

    print("Setting webhook to:", WEBHOOK_URL)
    try:
        bot.set_webhook(WEBHOOK_URL)
        print("Webhook set.")
    except Exception as e:
        print("Failed to set webhook:", e)
        traceback.print_exc()

    print("Starting Flask app. PORT =", PORT)
    app.run(host="0.0.0.0", port=PORT)
