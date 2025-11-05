# main.py
# OXTSIGNALSBOT PRO — режим: только реальные данные (yfinance, interval=1m)
# Требования (requirements.txt):
# python-telegram-bot==13.15
# pandas
# numpy
# yfinance
# flask
# requests

import os
import time
import threading
import csv
import traceback
from datetime import datetime, timedelta
from typing import Dict, Tuple

import pandas as pd
import numpy as np
import yfinance as yf
import requests
from flask import Flask
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Updater, CommandHandler, CallbackQueryHandler, CallbackContext

# ---------------- CONFIG ----------------
BOT_TOKEN = os.getenv("BOT_TOKEN") or ""  # <- обязательно в Render env
ANALYSIS_WAIT = 20                        # секунд ожидания анализа
PAGE_SIZE = 6
LOG_CSV = "signals_log.csv"

# Инструменты
FOREX = [
    "EURUSD","GBPUSD","USDJPY","AUDUSD","USDCHF","EURJPY",
    "GBPJPY","NZDUSD","EURGBP","CADJPY","USDCAD","AUDJPY",
    "EURAUD","GBPAUD","EURNZD","AUDNZD","CADCHF","CHFJPY",
    "NZDJPY","GBPCAD"
]

EXPIRATIONS = ["1m","2m","3m","5m"]

# веса индикаторов для голосования
WEIGHTS = {"EMA":2,"SMA":2,"MACD":2,"RSI":1,"BB":1}

# PRO правила (по твоим настройкам)
CONSECUTIVE_WINS_LIMIT = 5
PAUSE_MINUTES_AFTER_WINS = 5
COOLDOWN_SECONDS = 2   # защита от частых запросов в чате

# yfinance параметры
YF_INTERVAL = "1m"
YF_PERIOD = "2d"

# ---------------- FLASK (keep-alive) ----------------
app = Flask(__name__)
@app.route("/")
def index():
    return "OXTSIGNALSBOT PRO (yfinance 1m) is alive"

def keep_alive():
    t = threading.Thread(target=lambda: app.run(host="0.0.0.0", port=8080), daemon=True)
    t.start()

# ---------------- Logging helpers ----------------
def ensure_log():
    if not os.path.exists(LOG_CSV):
        with open(LOG_CSV, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "timestamp","chat_id","user_id","instrument","expiration",
                "signal","confidence","price_open","price_close","result"
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

# ---------------- Fetch real data (yfinance only, with retries) ----------------
def fetch_data_yf(pair: str, period: str = YF_PERIOD, interval: str = YF_INTERVAL, retries: int = 3, pause: float = 1.0) -> pd.DataFrame:
    ticker = yf_symbol(pair)
    last_err = None
    for attempt in range(retries):
        try:
            df = yf.download(ticker, period=period, interval=interval, progress=False, threads=False)
            if df is None or df.empty:
                last_err = "empty"
                time.sleep(pause)
                continue
            if "Close" not in df.columns:
                last_err = "no Close"
                time.sleep(pause)
                continue
            df = df.dropna(subset=["Close"])
            if df.empty:
                last_err = "dropna empty"
                time.sleep(pause)
                continue
            # coerce numeric
            for col in ["Open","High","Low","Close","Volume"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna(subset=["Close"])
            if df.empty:
                last_err = "close all NaN"
                time.sleep(pause)
                continue
            return df
        except Exception as e:
            last_err = str(e)
            time.sleep(pause)
    # If we arrive here — return None to signal upstream that data not available
    print(f"[fetch_data_yf] failed for {pair}: {last_err}")
    return None

# ---------------- Indicators ----------------
def compute_indicators(df: pd.DataFrame) -> Dict[str, float]:
    out = {}
    try:
        close = df["Close"].astype(float)
        high = df["High"].astype(float)
        low = df["Low"].astype(float)
        n = len(close)
        if n < 5:
            return out

        # EMA 8 vs 21
        ema8 = close.ewm(span=8, adjust=False).mean().iloc[-1]
        ema21 = close.ewm(span=21, adjust=False).mean().iloc[-1]
        out["EMA"] = 1 if ema8 > ema21 else -1

        # SMA 5 vs 20
        sma5 = close.rolling(window=5, min_periods=1).mean().iloc[-1]
        sma20 = close.rolling(window=min(20,n), min_periods=1).mean().iloc[-1]
        out["SMA"] = 1 if sma5 > sma20 else -1

        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        macd_sig = macd.ewm(span=9, adjust=False).mean()
        out["MACD"] = 1 if macd.iloc[-1] > macd_sig.iloc[-1] else -1
        out["MACD_mag"] = float(abs((macd - macd_sig).iloc[-1]))

        # RSI
        delta = close.diff().dropna()
        up = delta.clip(lower=0).rolling(window=14, min_periods=1).mean()
        down = (-delta.clip(upper=0)).rolling(window=14, min_periods=1).mean()
        rs = up / down.replace(0, 1e-9)
        rsi = 100 - (100 / (1 + rs))
        rsi_val = float(rsi.iloc[-1]) if len(rsi) > 0 else 50.0
        out["_RSI"] = rsi_val
        out["RSI"] = 1 if rsi_val > 55 else (-1 if rsi_val < 45 else 0)

        # Bollinger Bands
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

    except Exception as e:
        print("compute_indicators error:", e)
        traceback.print_exc()
    return out

# ---------------- Voting & confidence ----------------
def vote_and_confidence(ind: Dict[str, float]) -> Tuple[str, float]:
    score = 0.0
    max_score = 0.0
    mapping = {"EMA": ind.get("EMA",0), "SMA": ind.get("SMA",0), "MACD": ind.get("MACD",0), "RSI": ind.get("RSI",0), "BB": ind.get("BB",0)}
    for k,w in WEIGHTS.items():
        v = mapping.get(k,0)
        score += v * w
        max_score += abs(w)
    confidence = (abs(score) / max_score * 100) if max_score > 0 else 0.0
    macd_mag = ind.get("MACD_mag", 0.0)
    confidence = confidence + min(10.0, macd_mag * 1000.0)
    confidence = max(0.0, min(99.9, confidence))
    direction = "Вверх ↑" if score >= 0 else "Вниз ↓"
    return direction, round(confidence, 1)

# ---------------- Per-chat stats & locks ----------------
chat_stats: Dict[int, Dict] = {}   # chat_id -> stats
locks: Dict[int, threading.Lock] = {}

def get_stats(chat_id: int) -> Dict:
    st = chat_stats.get(chat_id)
    if not st:
        st = {"wins":0,"losses":0,"consec_wins":0,"consec_losses":0,"paused_until":None,"last_signal_time":None}
        chat_stats[chat_id] = st
    return st

def get_lock(chat_id: int) -> threading.Lock:
    if chat_id not in locks:
        locks[chat_id] = threading.Lock()
    return locks[chat_id]

# ---------------- Keyboards ----------------
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

# ---------------- Bot handlers ----------------
def cmd_start(update: Update, context: CallbackContext):
    kb = [
        [InlineKeyboardButton("💱 Валютные пары", callback_data="cat_fx_0")],
        [InlineKeyboardButton("📰 NON-FARM (NFP)", callback_data="nfp_mode")]
    ]
    update.message.reply_text("👋 Выберите режим:", reply_markup=InlineKeyboardMarkup(kb))

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

            stats = get_stats(q.message.chat_id)
            # paused check
            if stats.get("paused_until") and datetime.utcnow() < stats["paused_until"]:
                q.answer(f"Авто-пауза до {stats['paused_until'].strftime('%H:%M:%S UTC')}", show_alert=True)
                return

            # cooldown check
            last = stats.get("last_signal_time")
            if last and (datetime.utcnow() - last).total_seconds() < COOLDOWN_SECONDS:
                q.answer("Подождите немного перед новым запросом.", show_alert=True)
                return

            lock = get_lock(q.message.chat_id)
            if not lock.acquire(blocking=False):
                q.answer("Анализ уже идёт — подождите.", show_alert=True)
                return

            sent = q.edit_message_text(f"⏳ Подождите {ANALYSIS_WAIT} сек — идёт анализ {pair} ...", parse_mode="Markdown")
            threading.Thread(target=analysis_worker, args=(context.bot, q.message.chat_id, sent.message_id, pair, exp, q.from_user.id, lock), daemon=True).start()
            return

        if data == "new_signal":
            kb = [
                [InlineKeyboardButton("💱 Валютные пары", callback_data="cat_fx_0")],
                [InlineKeyboardButton("📰 NON-FARM (NFP)", callback_data="nfp_mode")]
            ]
            q.edit_message_text("Выберите режим:", reply_markup=InlineKeyboardMarkup(kb))
            return

        if data == "nfp_mode":
            lock = get_lock(q.message.chat_id)
            if not lock.acquire(blocking=False):
                q.answer("Анализ уже идёт — подождите.", show_alert=True)
                return
            sent = q.edit_message_text("⏳ Выполняется NFP-анализ для EURUSD (после выхода) — подождите...", parse_mode="Markdown")
            threading.Thread(target=nfp_worker, args=(context.bot, q.message.chat_id, sent.message_id, q.from_user.id, lock), daemon=True).start()
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
def analysis_worker(bot, chat_id: int, message_id: int, pair: str, exp: str, user_id: int, lock: threading.Lock):
    stats = get_stats(chat_id)
    try:
        stats["last_signal_time"] = datetime.utcnow()
        # wait before analysis (gives time to gather ticks)
        time.sleep(ANALYSIS_WAIT)

        df = fetch_data_yf(pair)
        if df is None or df.empty:
            bot.edit_message_text(chat_id=chat_id, message_id=message_id, text="⚠️ Не удалось получить данные (yfinance). Попробуйте позже.")
            return

        ind = compute_indicators(df)
        if not ind:
            bot.edit_message_text(chat_id=chat_id, message_id=message_id, text="⚠️ Недостаточно данных для анализа.")
            return

        direction, conf = vote_and_confidence(ind)

        # короткая логика (5-8 частей)
        expl = []
        expl.append("EMA8>EMA21" if ind.get("EMA",0)==1 else "EMA8<EMA21")
        rsi_v = ind.get("_RSI",50)
        expl.append(f"RSI≈{int(rsi_v)}")
        bb = ind.get("BB",0)
        if bb == 1:
            expl.append("Цена у нижней полосы BB")
        elif bb == -1:
            expl.append("Цена у верхней полосы BB")
        expl.append(f"MACD_mag={round(ind.get('MACD_mag',0),6)}")
        expl_text = "; ".join(expl[:5])

        price_open = float(df["Close"].iloc[-1])

        text = (
            f"📊 *Анализ завершён*\n\n"
            f"🔹 {pair} | Эксп: {exp}\n"
            f"📈 *Сигнал:* *{direction}*   🎯 *Уверенность:* *{conf}%*\n\n"
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
            "confidence": conf,
            "price_open": price_open,
            "price_close": "",
            "result": "pending"
        })

        # schedule finalize
        seconds = exp_to_seconds(exp)
        threading.Timer(seconds, finalize_worker, args=(bot, chat_id, message_id, pair, exp, direction, conf, price_open, user_id)).start()

    except Exception as e:
        print("analysis_worker error:", e)
        traceback.print_exc()
        try:
            bot.send_message(chat_id, "⚠️ Ошибка при анализе. Попробуйте снова.")
        except:
            pass
    finally:
        try:
            if lock and lock.locked():
                lock.release()
        except:
            pass

# ---------------- Finalize worker ----------------
def finalize_worker(bot, chat_id: int, message_id: int, pair: str, exp: str, direction: str, conf: float, price_open: float, user_id: int):
    stats = get_stats(chat_id)
    try:
        df2 = fetch_data_yf(pair)
        if df2 is None or df2.empty:
            price_close = price_open
        else:
            price_close = float(df2["Close"].iloc[-1])

        win = (direction.startswith("Вверх") and price_close > price_open) or (direction.startswith("Вниз") and price_close < price_open)
        result = "Плюс ✅" if win else "Минус ❌"

        pause_text = None
        if win:
            stats["wins"] = stats.get("wins",0) + 1
            stats["consec_wins"] = stats.get("consec_wins",0) + 1
            stats["consec_losses"] = 0
            if stats["consec_wins"] >= CONSECUTIVE_WINS_LIMIT:
                stats["paused_until"] = datetime.utcnow() + timedelta(minutes=PAUSE_MINUTES_AFTER_WINS)
                pause_text = f"🔥 Серия из {CONSECUTIVE_WINS_LIMIT} плюсов! Делаем паузу {PAUSE_MINUTES_AFTER_WINS} минут."
                stats["consec_wins"] = 0
        else:
            stats["losses"] = stats.get("losses",0) + 1
            stats["consec_losses"] = stats.get("consec_losses",0) + 1
            stats["consec_wins"] = 0

        final_text = (
            f"✅ *Сделка завершена*\n\n"
            f"*{pair}* | Эксп: *{exp}*\n"
            f"*Сигнал:* *{direction}*    *Результат:* *{result}*\n"
            f"*Уверенность:* *{conf}%\n\n"
            f"_Открытие:_ `{price_open:.6f}`\n"
            f"_Закрытие:_ `{price_close:.6f}`"
        )
        if pause_text:
            final_text += f"\n\n{pause_text}"

        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("📊 Получить новый сигнал", callback_data="new_signal")],
            [InlineKeyboardButton("🔁 Выбрать другую пару", callback_data="cat_fx_0")]
        ])

        try:
            bot.edit_message_text(chat_id=chat_id, message_id=message_id, text=final_text, parse_mode="Markdown", reply_markup=kb)
        except:
            try:
                bot.send_message(chat_id, final_text, parse_mode="Markdown", reply_markup=kb)
            except:
                pass

        log_row({
            "timestamp": datetime.utcnow().isoformat(),
            "chat_id": chat_id,
            "user_id": user_id,
            "instrument": pair,
            "expiration": exp,
            "signal": direction,
            "confidence": conf,
            "price_open": price_open,
            "price_close": price_close,
            "result": result
        })

    except Exception as e:
        print("finalize_worker error:", e)
        traceback.print_exc()

# ---------------- NFP worker (after news) ----------------
def nfp_worker(bot, chat_id: int, message_id: int, user_id: int, lock: threading.Lock):
    try:
        # short wait to allow market to reflect news
        time.sleep(3)
        pair = "EURUSD"
        df = fetch_data_yf(pair, period="1d", interval="1m")
        if df is None or df.empty:
            bot.edit_message_text(chat_id=chat_id, message_id=message_id, text="⚠️ Не удалось получить данные для NFP.")
            return

        ind = compute_indicators(df)
        if not ind:
            bot.edit_message_text(chat_id=chat_id, message_id=message_id, text="⚠️ Недостаточно данных для NFP.")
            return

        direction, conf = vote_and_confidence(ind)

        # ATR for expiration suggestion
        try:
            high = df["High"].astype(float); low = df["Low"].astype(float); prev = df["Close"].shift(1).fillna(df["Close"].iloc[0])
            tr = pd.concat([high-low, (high-prev).abs(), (low-prev).abs()], axis=1).max(axis=1)
            atr_val = tr.rolling(window=14, min_periods=1).mean().iloc[-1]
        except:
            atr_val = None

        if atr_val and atr_val > 0.0025:
            suggested_exp = "1m"
        elif atr_val and atr_val > 0.0010:
            suggested_exp = "2m"
        else:
            suggested_exp = "3m"

        expl = []
        expl.append("NFP (после выхода)")
        expl.append("EMA8>EMA21" if ind.get("EMA",0)==1 else "EMA8<EMA21")
        expl.append(f"RSI≈{int(ind.get('_RSI',50))}")
        if atr_val: expl.append(f"ATR≈{round(atr_val,6)}")
        expl_text = "; ".join(expl[:5])

        text = (
            f"📰 *NFP Анализ (EURUSD, после выхода)*\n\n"
            f"📈 *Рекомендация:* *{direction}*\n"
            f"⏱ *Рекомендуемая экспирация:* *{suggested_exp}*\n"
            f"🎯 *Уверенность:* *{conf}%*\n\n"
            f"_Короткая логика:_ {expl_text}\n\n"
            f"📌 Нажмите «Получить новый сигнал» чтобы вернуться в меню."
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
    finally:
        try:
            if lock and lock.locked():
                lock.release()
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

    print("OXTSIGNALSBOT PRO (yfinance 1m) started (polling)")
    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()
