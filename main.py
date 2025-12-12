# main.py
import os
import io
import asyncio
import threading
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf
import pandas_ta as ta
import mplfinance as mpf
from aiogram import Bot, Dispatcher, types
from aiogram.utils import executor

# -------------------- Конфиг --------------------
TG_TOKEN = os.getenv("TG_TOKEN") or "ВАШ_TELEGRAM_TOKEN"
CANDLES_LIMIT = int(os.getenv("CANDLES_LIMIT", 500))  # по умолчанию 500 свечей

# Валютные пары и таймфреймы
PAIRS = [
    "EURUSD=X","GBPUSD=X","USDJPY=X","AUDUSD=X","USDCAD=X","USDCHF=X",
    "EURJPY=X","GBPJPY=X","AUDJPY=X","EURGBP=X","EURAUD=X","GBPAUD=X",
    "CADJPY=X","CHFJPY=X","EURCAD=X","GBPCAD=X","AUDCAD=X","AUDCHF=X","CADCHF=X"
]
EXPIRATIONS = [1, 3, 5, 10]  # минуты

# Файл для хранения пользователей
USERS_FILE = "users.txt"

bot = Bot(token=TG_TOKEN)
dp = Dispatcher(bot)

# -------------------- Пользователи --------------------
def load_users():
    try:
        with open(USERS_FILE, "r") as f:
            return set(int(line.strip()) for line in f.readlines())
    except:
        return set()

def save_user(user_id):
    users = load_users()
    if user_id not in users:
        users.add(user_id)
        with open(USERS_FILE, "w") as f:
            for u in users:
                f.write(f"{u}\n")

# -------------------- Telegram Handlers --------------------
@dp.message_handler(commands=['start'])
async def start_handler(message: types.Message):
    save_user(message.from_user.id)
    await message.reply("Привет! Я буду присылать тебе торговые сигналы 24/7.")

# -------------------- Получение свечей --------------------
def fetch_ohlcv_yf(symbol: str, exp_minutes: int, limit: int = CANDLES_LIMIT) -> pd.DataFrame:
    if exp_minutes == 1:
        interval = "1m"
        df = yf.download(tickers=symbol, period="2d", interval=interval, progress=False)
        df = df.rename(columns=str.lower)[['open','high','low','close','volume']]
        return df.tail(limit)
    else:
        df1 = yf.download(tickers=symbol, period="2d", interval="1m", progress=False)
        df1 = df1.rename(columns=str.lower)[['open','high','low','close','volume']]
        rule = f"{exp_minutes}T"
        df_res = pd.DataFrame()
        df_res['open'] = df1['open'].resample(rule).first()
        df_res['high'] = df1['high'].resample(rule).max()
        df_res['low'] = df1['low'].resample(rule).min()
        df_res['close'] = df1['close'].resample(rule).last()
        df_res['volume'] = df1['volume'].resample(rule).sum()
        df_res.dropna(inplace=True)
        return df_res.tail(limit)

# -------------------- Индикаторы --------------------
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[['open','high','low','close','volume']] = df[['open','high','low','close','volume']].astype(float)
    df['ema9'] = ta.ema(df['close'], length=9)
    df['ema21'] = ta.ema(df['close'], length=21)
    df['sma50'] = ta.sma(df['close'], length=50)
    macd = ta.macd(df['close'])
    df['macd'] = macd.iloc[:,0]
    df['macd_signal'] = macd.iloc[:,1]
    df['rsi14'] = ta.rsi(df['close'], length=14)
    st = ta.stoch(df['high'], df['low'], df['close'])
    df['stoch_k'] = st.iloc[:,0]
    df['stoch_d'] = st.iloc[:,1]
    bb = ta.bbands(df['close'])
    df['bb_upper'] = bb.iloc[:,0]
    df['bb_lower'] = bb.iloc[:,2]
    df['atr14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['adx14'] = ta.adx(df['high'], df['low'], df['close']).iloc[:,0]
    df['cci20'] = ta.cci(df['high'], df['low'], df['close'], length=20)
    df['obv'] = ta.obv(df['close'], df['volume'])
    df['wr'] = ta.wr(df['high'], df['low'], df['close'])
    df['ema5'] = ta.ema(df['close'], length=5)
    df['ema10'] = ta.ema(df['close'], length=10)
    df['ema20'] = ta.ema(df['close'], length=20)
    try:
        stt = ta.supertrend(df['high'], df['low'], df['close'], length=10, multiplier=3.0)
        df['supertrend_dir'] = stt.iloc[:,2]
    except:
        df['supertrend_dir'] = 0
    df['mom10'] = ta.mom(df['close'], length=10)
    try:
        ichi = ta.ichimoku(df['high'], df['low'], df['close'])
        df['ichimoku_conv'] = ichi.iloc[:,0]
    except:
        df['ichimoku_conv'] = float('nan')
    return df

# -------------------- Голосование индикаторов --------------------
def indicator_vote(latest: pd.Series, df: pd.DataFrame) -> dict:
    votes, weights, explanation = [], [], []
    def add(name, sign, weight):
        votes.append(sign)
        weights.append(weight)
        label = "BUY" if sign==1 else ("SELL" if sign==-1 else "NEUTRAL")
        explanation.append(f"{name}: {label} (w={weight})")
    add("EMA(9/21)", 1 if latest['ema9']>latest['ema21'] else (-1 if latest['ema9']<latest['ema21'] else 0), 1.0)
    try:
        if latest['ema5']>latest['ema10']>latest['ema20']:
            add("EMA Ribbon",1,0.9)
        elif latest['ema5']<latest['ema10']<latest['ema20']:
            add("EMA Ribbon",-1,0.9)
        else: add("EMA Ribbon",0,0.9)
    except: add("EMA Ribbon",0,0.9)
    add("SMA50", 1 if latest['close']>latest['sma50'] else (-1 if latest['close']<latest['sma50'] else 0), 0.8)
    add("MACD",1 if latest['macd']>latest['macd_signal'] else (-1 if latest['macd']<latest['macd_signal'] else 0),1.0)
    if latest['rsi14']<30: add("RSI",1,0.7)
    elif latest['rsi14']>70: add("RSI",-1,0.7)
    else: add("RSI",0,0.7)
    if latest['stoch_k']>latest['stoch_d'] and latest['stoch_k']<80: add("Stochastic",1,0.6)
    elif latest['stoch_k']<latest['stoch_d'] and latest['stoch_k']>20: add("Stochastic",-1,0.6)
    else: add("Stochastic",0,0.6)
    if latest['close']>latest['bb_upper']: add("Bollinger",1,0.5)
    elif latest['close']<latest['bb_lower']: add("Bollinger",-1,0.5)
    else: add("Bollinger",0,0.5)
    if latest['adx14']>25: add("ADX Trend",1 if latest['ema9']>latest['ema21'] else -1,1.2)
    else: add("ADX Trend",0,0.5)
    stdir = latest.get('supertrend_dir',0)
    add("Supertrend",1 if stdir==1 else (-1 if stdir==-1 else 0),1.2)
    if latest['cci20']<-100: add("CCI",1,0.5)
    elif latest['cci20']>100: add("CCI",-1,0.5)
    else: add("CCI",0,0.5)
    add("Momentum",1 if latest['mom10']>0 else (-1 if latest['mom10']<0 else 0),0.6)
    slope = latest['obv']-df['obv'].iloc[-3] if len(df['obv'])>=3 else 0
    add("OBV",1 if slope>0 else (-1 if slope<0 else 0),0.4)
    if latest['wr']<-80: add("Williams %R",1,0.4)
    elif latest['wr']>-20: add("Williams %R",-1,0.4)
    else: add("Williams %R",0,0.4)
    add("Ichimoku(conv)",1 if latest['close']>latest['ichimoku_conv'] else -1,0.6)

    votes_sum = sum(v*w for v,w in zip(votes, weights))
    max_possible = sum(abs(w) for w in weights) or 1.0
    confidence = min(100,int(abs(votes_sum)/max_possible*100))
    direction = "HOLD" if abs(votes_sum)<0.15*max_possible else ("BUY" if votes_sum>0 else "SELL")
    return {"score":votes_sum,"max_score":max_possible,"confidence":confidence,"direction":direction,"explanation":explanation}

# -------------------- График --------------------
def plot_chart(df: pd.DataFrame) -> io.BytesIO:
    plot_df = df[['open','high','low','close','volume']].tail(150)
    addplots = [mpf.make_addplot(df['ema9'].tail(150)), mpf.make_addplot(df['ema21'].tail(150))]
    if 'bb_upper' in df and 'bb_lower' in df:
        addplots.append(mpf.make_addplot(df['bb_upper'].tail(150)))
        addplots.append(mpf.make_addplot(df['bb_lower'].tail(150)))
    buf = io.BytesIO()
    mpf.plot(plot_df,type='candle',style='yahoo',volume=True,addplot=addplots,savefig=buf,tight_layout=True)
    buf.seek(0)
    return buf

# -------------------- Автономная рассылка --------------------
async def send_signal_to_all(pair, timeframe):
    df = fetch_ohlcv_yf(pair, timeframe)
    df_ind = compute_indicators(df)
    latest = df_ind.iloc[-1]
    res = indicator_vote(latest, df_ind)
    if res["confidence"] < 60:
        return
    chart_buf = plot_chart(df_ind)
    dir_map = {"BUY":"🔺 ПОКУПКА","SELL":"🔻 ПРОДАЖА","HOLD":"⚠️ НЕОДНОЗНАЧНО"}
    text = (f"📊 Торговый сигнал\nПара: {pair}\nТаймфрейм: {timeframe} мин\n"
            f"Направление: {dir_map[res['direction']]}\nУверенность: {res['confidence']}%")
    users = load_users()
    for user_id in users:
        try:
            await bot.send_photo(chat_id=user_id, photo=chart_buf, caption=text)
        except Exception as e:
            print(f"Ошибка при отправке {user_id}: {e}")

async def main_loop():
    while True:
        for pair in PAIRS:
            for timeframe in EXPIRATIONS:
                try:
                    await send_signal_to_all(pair, timeframe)
                except Exception as e:
                    print(f"Ошибка {pair} {timeframe} мин: {e}")
        await asyncio.sleep(60)  # проверка каждую минуту

# -------------------- Запуск --------------------
if __name__=="__main__":
    # запуск polling в отдельном потоке для пользователей
    threading.Thread(target=lambda: executor.start_polling(dp, skip_updates=True)).start()
    # запуск автономного цикла 24/7
    asyncio.run(main_loop())
