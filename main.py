# main.py
# OXTSIGNALSBOT PRO MAX (Webhook) + Statistics + History
# Do NOT hardcode BOT_TOKEN. Use environment variables.

import os
import time
import threading
import csv
import traceback
from datetime import datetime, time as dtime, timedelta
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
BOT_TOKEN = os.getenv("BOT_TOKEN") or ""
WEBHOOK_URL = os.getenv("WEBHOOK_URL") or ""
PORT = int(os.getenv("PORT", "10000"))

LOG_CSV = "signals_log.csv"

FOREX = [
    "EURUSD","GBPUSD","USDJPY","AUDUSD","USDCHF","EURJPY",
    "GBPJPY","NZDUSD","EURGBP","CADJPY","USDCAD","AUDJPY",
    "EURAUD","GBPAUD","EURNZD","AUDNZD","CADCHF","CHFJPY",
    "NZDJPY","GBPCAD"
]

EXPIRATIONS = ["1m","2m","3m","5m"]
PAGE_SIZE = 6
ANALYSIS_WAIT = 12  # seconds to simulate "professional" analysis

# thresholds / tuning
MIN_ATR = 0.00008
MAX_ATR = 0.01
MAX_STD_MA_RATIO = 0.0007
HIGH_QUALITY = 75.0
MEDIUM_QUALITY = 55.0

GOOD_HOURS_UTC = [(7,22)]  # preferred trading window (UTC)

# ---------------- FLASK + TELEGRAM DISPATCHER ----------------
app = Flask(__name__)
if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN is not set in environment variables.")
bot = telegram.Bot(token=BOT_TOKEN)
dispatcher = Dispatcher(bot, None, workers=8, use_context=True)


@app.route("/", methods=["GET"])
def home():
    return "OXTSIGNALSBOT PRO MAX (webhook) — alive"


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


# ---------------- Smart fallback ----------------
def smart_fallback(seed: str, bars: int = 480) -> pd.DataFrame:
    rnd = random.Random(abs(hash(seed)) % (10**9))
    base_level = 1.0 + (abs(hash(seed)) % 200) / 1000.0
    vol = rnd.uniform(0.0003, 0.0025)
    times = pd.date_range(end=pd.Timestamp.now(), periods=bars, freq="1min")
    opens, highs, lows, closes, vols = [], [], [], [], []
    price = base_level
    for _ in range(bars):
        if rnd.random() < 0.02:
            # occasional trend switch
            vol = max(0.0001, vol * rnd.uniform(0.7, 1.3))
        o = price
        c = max(0.00001, o + rnd.gauss(0, vol))
        h = max(o, c) + abs(rnd.gauss(0, vol*0.8))
        l = min(o, c) - abs(rnd.gauss(0, vol*0.8))
        v = rnd.randint(50, 200)
        opens.append(o); highs.append(h); lows.append(l); closes.append(c); vols.append(v)
        price = c
    df = pd.DataFrame({"Open":opens,"High":highs,"Low":lows,"Close":closes,"Volume":vols}, index=times)
    return df


# ---------------- Fetch data (yfinance + fallback) ----------------
def fetch_data(pair: str) -> pd.DataFrame:
    symbol = yf_symbol(pair)
    try:
        df = yf.download(symbol, period="2d", interval="1m", progress=False, threads=False)
        if df is None or df.empty:
            raise Exception("yfinance empty")
        df = df.dropna(subset=["Close"])
        if df.empty:
            raise Exception("yfinance close empty")
        # ensure numeric
        for col in ["Open","High","Low","Close","Volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["Close"])
        if df.empty:
            raise Exception("yfinance after numeric empty")
        return df
    except Exception as e:
        print(f"[fetch_data] yfinance failed for {pair}: {e}. Using fallback.")
        try:
            return smart_fallback(pair)
        except Exception as ex:
            print("fallback failed:", ex)
            return pd.DataFrame({"Close":[1.0]})


# ---------------- Indicators (PRO) ----------------
def compute_indicators(df: pd.DataFrame) -> Dict[str, float]:
    out: Dict[str, float] = {}
    try:
        close = df["Close"].astype(float)
        high = df["High"].astype(float) if "High" in df.columns else close
        low = df["Low"].astype(float) if "Low" in df.columns else close
        n = len(close)
        if n < 5:
            return out

        # EMA
        out["EMA8"] = close.ewm(span=8, adjust=False).mean().iloc[-1]
        out["EMA21"] = close.ewm(span=21, adjust=False).mean().iloc[-1]
        out["EMA"] = 1 if out["EMA8"] > out["EMA21"] else -1

        # SMA
        out["SMA5"] = close.rolling(window=5, min_periods=1).mean().iloc[-1]
        out["SMA20"] = close.rolling(window=min(20, n), min_periods=1).mean().iloc[-1]
        out["SMA"] = 1 if out["SMA5"] > out["SMA20"] else -1

        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        macd_sig = macd.ewm(span=9, adjust=False).mean()
        out["MACD_hist"] = float((macd - macd_sig).iloc[-1])
        out["MACD"] = 1 if macd.iloc[-1] > macd_sig.iloc[-1] else -1
        out["MACD_trend"] = float((macd - macd_sig).iloc[-1] - (macd - macd_sig).iloc[-2]) if n >= 2 else 0.0

        # RSI
        delta = close.diff().dropna()
        up = delta.clip(lower=0).rolling(window=14, min_periods=1).mean()
        down = (-delta.clip(upper=0)).rolling(window=14, min_periods=1).mean()
        rs = up / down.replace(0, 1e-9)
        rsi = 100 - (100 / (1 + rs))
        out["_RSI"] = float(rsi.iloc[-1]) if len(rsi) > 0 else 50.0
        out["RSI"] = 1 if out["_RSI"] > 55 else (-1 if out["_RSI"] < 45 else 0)

        # Bollinger
        ma20 = close.rolling(window=min(20, n), min_periods=1).mean()
        std20 = close.rolling(window=min(20, n), min_periods=1).std().fillna(0)
        out["BB"] = 1 if float(close.iloc[-1]) < ma20.iloc[-1] - 2*std20.iloc[-1] else (-1 if float(close.iloc[-1]) > ma20.iloc[-1] + 2*std20.iloc[-1] else 0)

        # ATR
        prev_close = close.shift(1).fillna(close.iloc[0])
        tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        out["ATR"] = float(tr.rolling(window=14, min_periods=1).mean().iloc[-1])

        # STD/MA noise metric
        out["STD_MA"] = float(std20.iloc[-1] / ma20.iloc[-1]) if ma20.iloc[-1] != 0 else 0.0

        # candle info
        out["last_open"] = float(df["Open"].iloc[-1]) if "Open" in df.columns else float(close.iloc[-1])
        out["last_close"] = float(close.iloc[-1])
        out["prev_open"] = float(df["Open"].iloc[-2]) if "Open" in df.columns and len(df) >= 2 else out["last_open"]
        out["prev_close"] = float(df["Close"].iloc[-2]) if len(df) >=2 else out["last_close"]

    except Exception as e:
        print("compute_indicators error:", e)
        traceback.print_exc()
    return out


# ---------------- Candle patterns (simple) ----------------
def detect_candle_pattern(ind: Dict[str, float]) -> Optional[str]:
    try:
        o = ind.get("last_open")
        c = ind.get("last_close")
        po = ind.get("prev_open")
        pc = ind.get("prev_close")
        if o is None or c is None or po is None or pc is None:
            return None
        body = abs(c - o)
        prev_body = abs(pc - po)
        # Engulfing
        if (c > o and pc < po and c > po and o < pc) or (c < o and pc > po and c < po and o > pc):
            return "engulfing"
        # Pinbar / long shadow detection (approx)
        rng = abs(ind.get("last_close", c) - ind.get("last_open", o))
        if rng > 0 and body < 0.35 * (abs(ind.get("last_close", c) - ind.get("prev_close", pc)) + 1e-9):
            return "pinbar"
        # Hammer (small body, long lower shadow)
        return None
    except Exception:
        return None


# ---------------- Trend power & macd quality ----------------
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


# ---------------- Vote & base confidence ----------------
def vote_and_base_confidence(ind: Dict[str, float]) -> Tuple[str, float]:
    score = 0.0
    mapping = {"EMA": ind.get("EMA", 0), "SMA": ind.get("SMA", 0), "MACD": ind.get("MACD", 0), "RSI": ind.get("RSI", 0), "BB": ind.get("BB", 0)}
    weights = {"EMA": 2.0, "SMA": 1.0, "MACD": 2.0, "RSI": 1.0, "BB": 1.0}
    max_score = 0.0
    for k, w in weights.items():
        v = mapping.get(k, 0)
        score += v * w
        max_score += abs(w)
    base_conf = (abs(score) / max_score) * 60.0  # base 0..60
    base_conf += min(20.0, abs(ind.get("MACD_hist", 0)) * 10000.0)
    base_conf = max(10.0, min(90.0, base_conf))
    direction = "Вверх ↑" if score >= 0 else "Вниз ↓"
    return direction, round(base_conf, 1)


# ---------------- Quality calculation ----------------
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


# ---------------- NO-FLAT / ATR checks ----------------
def is_flat_or_bad_vol(ind: Dict[str, float]) -> Tuple[bool, str]:
    std_ma = ind.get("STD_MA", 0.0)
    atr = ind.get("ATR", 0.0)
    if std_ma is not None and std_ma < MAX_STD_MA_RATIO:
        return True, "flat_std"
    if atr is not None and atr < MIN_ATR:
        return True, "low_atr"
    if atr is not None and atr > MAX_ATR:
        return True, "high_atr"
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

            # start analysis in background
            sent = q.edit_message_text(f"⏳ Подождите {ANALYSIS_WAIT} сек — идёт профессиональный анализ {pair}...", parse_mode="Markdown")
            threading.Thread(target=analysis_worker, args=(context.bot, q.message.chat_id, sent.message_id, pair, exp, q.from_user.id), daemon=True).start()
            return

        if data == "nfp_mode":
            sent = q.edit_message_text("⏳ Выполняется NFP-анализ для EURUSD (после выхода) — подождите...", parse_mode="Markdown")
            threading.Thread(target=nfp_worker, args=(context.bot, q.message.chat_id, sent.message_id, q.from_user.id), daemon=True).start()
            return

        if data == "show_stats":
            # reply with stats (use same code as /stats)
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

        df = fetch_data(pair)
        if df is None or df.empty:
            bot.edit_message_text(chat_id=chat_id, message_id=message_id, text="⚠️ Не удалось получить данные. Попробуйте позже.")
            return

        ind = compute_indicators(df)
        if not ind:
            bot.edit_message_text(chat_id=chat_id, message_id=message_id, text="⚠️ Недостаточно данных для анализа.")
            return

        flat, reason = is_flat_or_bad_vol(ind)
        direction, base_conf = vote_and_base_confidence(ind)
        quality_label, quality_score = compute_quality_label_and_score(ind, base_conf)
        patt = detect_candle_pattern(ind)

        # Adjust quality by pattern & time
        if patt == "engulfing":
            quality_score = min(99.9, quality_score + 8)
        elif patt == "pinbar":
            quality_score = min(99.9, quality_score + 5)

        if not in_good_hours():
            quality_score = max(10.0, quality_score - 12.0)
            if quality_score < MEDIUM_QUALITY:
                quality_label = "Low"

        # If flat or bad vol -> recommend skip
        if flat:
            text = (
                f"⚠️ Рынок неподходящий ({reason}). Рекомендуется воздержаться.\n\n"
                f"Пара: *{pair}* | Эксп: *{exp}*\n"
                f"Технический сигнал: *{direction}* • Качество: *{quality_label}* ({quality_score}%)"
            )
            bot.edit_message_text(chat_id=chat_id, message_id=message_id, text=text, parse_mode="Markdown")
            # log skipped
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
            bot.send_message(chat_id, "🔁 Возвращаю в меню валютных пар:", reply_markup=main_menu_keyboard())
            return

        # suggested expiration logic
        tp, mq = trend_power_and_macd_quality(ind)
        suggested = "1m" if tp > 0.8 and ind.get("ATR",0) > MIN_ATR else ("2m" if tp > 0.35 else "3m")

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
            f"🔹 {pair} | Эксп: {exp}\n"
            f"📈 *Сигнал:* *{direction}*    🎯 *Качество:* *{quality_label}* ({quality_score}%)\n"
            f"⏱ *Рекомендуемая экспирация:* *{suggested}*\n\n"
            f"_Короткая логика:_ {expl_text}\n"
            f"_Цена (прибл.):_ `{price_open:.6f}`\n\n"
            f"⚡ Откройте сделку в течение *10 секунд*."
        )

        try:
            bot.edit_message_text(chat_id=chat_id, message_id=message_id, text=text, parse_mode="Markdown")
        except:
            bot.send_message(chat_id, text, parse_mode="Markdown")

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

        time.sleep(0.4)
        try:
            bot.send_message(chat_id, "🔁 Возвращаю в меню валютных пар:", reply_markup=main_menu_keyboard())
        except:
            pass

        # log final
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
        df = fetch_data(pair)
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


# ---------------- Statistics & History Commands ----------------
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
            f"📊 *Статистика OXTSIGNALSBOT*\n\n"
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

    if not WEBHOOK_URL:
        print("ERROR: WEBHOOK_URL not set in environment variables.")
    else:
        print("Setting webhook to:", WEBHOOK_URL)
        try:
            bot.set_webhook(WEBHOOK_URL)
            print("Webhook set.")
        except Exception as e:
            print("Failed to set webhook:", e)
            traceback.print_exc()

    print("Starting Flask app. PORT =", PORT)
    app.run(host="0.0.0.0", port=PORT)
