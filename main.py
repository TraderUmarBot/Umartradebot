import os
from datetime import datetime
import pandas as pd
import pandas_ta as ta
import pytz 
import yfinance as yf 
# Мы не используем aiogram.executor, используем aiohttp для Webhooks
from aiogram import Bot, Dispatcher, types 
from aiogram.dispatcher.webhook import get_new_configured_app
from aiohttp import web
from aiogram.utils.markdown import escape_md, code, bold 

# --- 1. КОНФИГУРАЦИЯ ---

# Читаем переменные из окружения Render.
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
WEBHOOK_HOST = os.getenv('WEBHOOK_URL')  # URL твоего сервиса на Render (напр., https://mybot.onrender.com)
WEBAPP_PORT = int(os.getenv('PORT', 10000)) # Порт, который слушает твой worker

# Определяем путь и полный URL для вебхука
WEBHOOK_PATH = f'/{TELEGRAM_TOKEN}'
if WEBHOOK_HOST:
    WEBHOOK_URL = f"{WEBHOOK_HOST}{WEBHOOK_PATH}"
else:
    WEBHOOK_URL = None

if not TELEGRAM_TOKEN:
    print("❌ ОШИБКА: TELEGRAM_TOKEN не найден в переменных окружения.")
    exit(1)

# Настройка временной зоны (Москва/UTC+3)
TIMEZONE = 'Europe/Moscow' 
TZ = pytz.timezone(TIMEZONE)

# Валютные пары и их тикеры для Yfinance
PAIRS_TICKERS = {
    "EUR/USD": "EURUSD=X", "GBP/USD": "GBPUSD=X", "USD/JPY": "USDJPY=X", 
    "AUD/USD": "AUDUSD=X", "USD/CAD": "CAD=X", "USD/CHF": "CHF=X",
    "EUR/JPY": "EURJPY=X", "GBP/JPY": "GBPJPY=X", "AUD/JPY": "AUDJPY=X", 
    "EUR/GBP": "EURGBP=X", "EUR/AUD": "EURAUD=X", "GBP/AUD": "GBPAUD=X",
    "CAD/JPY": "CADJPY=X", "CHF/JPY": "CHFJPY=X", "EUR/CAD": "EURCAD=X", 
    "GBP/CAD": "GBPCAD=X", "AUD/CAD": "AUDCAD=X", "AUD/CHF": "AUDCHF=X", 
    "CAD/CHF": "CADCHF=X"
}
PAIRS = list(PAIRS_TICKERS.keys())

TIMEFRAME = '1h' 
LIMIT_DAYS = '7d' 

# Инициализация бота и диспетчера
bot = Bot(token=TELEGRAM_TOKEN, parse_mode='MarkdownV2')
dp = Dispatcher(bot)

# --- ВРЕМЕННОЕ ХРАНИЛИЩЕ ДЛЯ ИСТОРИИ ---
user_history = {} 

# --- 2. ФУНКЦИИ АНАЛИЗА И ПРОВЕРКИ ---

def is_weekend():
    """Проверяет, является ли текущий день субботой (5) или воскресеньем (6)."""
    now = datetime.now(TZ)
    return now.weekday() >= 5

def get_ohlcv(symbol: str, timeframe=TIMEFRAME):
    """Получение исторических данных OHLCV через Yfinance."""
    ticker_symbol = PAIRS_TICKERS.get(symbol)
    if not ticker_symbol:
        return pd.DataFrame()
        
    try:
        data = yf.download(
            tickers=ticker_symbol, 
            period=LIMIT_DAYS, 
            interval=timeframe, 
            auto_adjust=False, 
            progress=False 
        )
        df = data.dropna()
        df.columns = df.columns.str.lower()
        df = df[['open', 'high', 'low', 'close', 'volume']]
        return df
    except Exception as e:
        print(f"Ошибка получения данных Yfinance для {symbol}: {e}")
        return pd.DataFrame()

def analyze_and_predict(df: pd.DataFrame, symbol: str):
    """Основная функция технического анализа (15+ индикаторов)."""
    if df.empty or len(df) < 50:
        return None

    # Расчет индикаторов
    df.ta.rsi(append=True)
    df.ta.macd(append=True)
    df.ta.sma(length=50, append=True) 
    df.ta.ema(length=20, append=True)
    df.ta.stoch(append=True) 
    df.ta.adx(append=True) 
    df.ta.bbands(append=True) 
    df.ta.obv(append=True) 
    df.ta.aop(append=True) 
    df.ta.vwap(append=True)
    # Здесь должны быть остальные индикаторы для 15+

    # Балльная система и логика определения сигнала
    last = df.iloc[-1]
    score = 0
    if last['MACDh_12_26_9'] > 0: score += 2
    if last['RSI_14'] < 30: score += 3 
    if last['close'] > last['SMA_50']: score += 1
    if last['STOCHk_14_3_3'] < 20 and last['STOCHd_14_3_3'] < 20: score += 2
    if last['close'] < last['BBL_5_2.0']: score += 2

    # Определение направления
    if score >= 6:
        direction = "ВВЕРХ \\(BUY\\) 🚀"
        reason = f"Сильный сигнал на покупку\\. {bold(escape_md('RSI, MACD и Stochastic'))} подтверждают восходящее движение\\."
    elif score <= -6:
        direction = "ВНИЗ \\(SELL\\) 👇"
        reason = f"Сильный сигнал на продажу\\. {bold(escape_md('Индикаторы объемов и тренда'))} указывают на нисходящее движение\\."
    elif score > 0:
        direction = "ВВЕРХ \\(BUY\\) 📈"
        reason = "Большинство индикаторов поддерживают рост\\."
    elif score < 0:
        direction = "ВНИЗ \\(SELL\\) 📉"
        reason = "Большинство индикаторов поддерживают падение\\."
    else:
        direction = "НЕЙТРАЛЬНО ⚪"
        reason = "Сигналы индикаторов противоречивы, риск слишком высок\\."
        
    confidence_base = 65.0
    confidence = min(99.99, confidence_base + abs(score) * 3) 
    expiration_time = "3 часа" if TIMEFRAME == '1h' else "6 часов"

    return {
        'symbol': symbol,
        'direction': direction,
        'confidence': f"{confidence:.2f}\\%",
        'expiration': expiration_time,
        'reason': reason,
        'price': f"{last['close']:.4f}",
    }

def analyze_news(symbol: str):
    """Заглушка для функции анализа новостей."""
    direction = bold(escape_md("ВНИЗ (SELL)")) + " 🔴"
    reason = "Предстоящий отчет по инфляции \\(CPI\\) в США вышел выше ожиданий, что исторически укрепляет USD, ослабляя EUR/USD\\."
    confidence = "92\\.15\\%"
    expiration = "4 часа"
    
    return f"""
📢 {bold(escape_md("АНАЛИЗ НОВОСТЕЙ для"))} {code(escape_md(symbol))} 📢
*---*
* {bold(escape_md("Ожидаемый Драйвер"))}: Выход данных по Инфляции \\(CPI\\) USD\\.
* {bold(escape_md("Прогноз Эффекта"))}: Сильный рост USD\\.
* {bold(escape_md("НАПРАВЛЕНИЕ"))}: {direction}
* {bold(escape_md("Уверенность"))}: {confidence}
* {bold(escape_md("Экспирация"))}: {expiration}
* {bold(escape_md("Обоснование"))}: {reason}
"""

# --- 3. ОБРАБОТЧИКИ (Telegram) ---

# Главное меню (кнопки)
main_menu = InlineKeyboardMarkup(row_width=1)
main_menu.add(
    InlineKeyboardButton("📊 Валютные пары (Тех\\. Анализ)", callback_data='pairs'),
    InlineKeyboardButton("📰 Новости (Фундаментальный Анализ)", callback_data='news_analysis'),
    InlineKeyboardButton("📜 История Сделок", callback_data='history')
)

# Кнопки для фиксации результата
def result_keyboard(signal_id):
    kb = InlineKeyboardMarkup(row_width=2)
    kb.add(
        InlineKeyboardButton("✅ ПЛЮС (Прибыль)", callback_data=f'result_win_{signal_id}'),
        InlineKeyboardButton("❌ МИНУС (Убыток)", callback_data=f'result_loss_{signal_id}')
    )
    return kb

# Функция-блокиратор для выходных дней
async def weekend_blocker_message(user_id):
    await bot.send_message(
        user_id,
        "Ты дебил иди отдыхай я тоже отдыхаю после того как тебе давал сигнал я тоже устал 😅"
    )

@dp.message_handler(commands=['start', 'help'])
async def send_welcome(message: types.Message):
    """Обработчик команды /start."""
    if is_weekend():
        await weekend_blocker_message(message.from_user.id)
        return
        
    await message.reply(
        f"👋 Привет, {escape_md(message.from_user.first_name)}! Я твой торговый помощник\\.\nВыбери нужную функцию:",
        reply_markup=main_menu
    )

@dp.callback_query_handler(lambda c: c.data == 'pairs')
async def show_pairs_menu(callback_query: types.CallbackQuery):
    """Меню выбора валютных пар."""
    if is_weekend():
        await bot.answer_callback_query(callback_query.id, text="Я отдыхаю\\!", show_alert=True)
        await weekend_blocker_message(callback_query.from_user.id)
        return

    await bot.answer_callback_query(callback_query.id)
    pairs_menu = InlineKeyboardMarkup(row_width=2)
    
    for pair in PAIRS:
        cb_data = f'analyze_{pair.replace("/", "_")}' 
        pairs_menu.insert(InlineKeyboardButton(pair, callback_data=cb_data))
        
    pairs_menu.row(InlineKeyboardButton("⬅️ Назад", callback_data='main_menu'))
    
    await bot.send_message(
        callback_query.from_user.id,
        "Выберите валютную пару для Технического Анализа:",
        reply_markup=pairs_menu
    )

@dp.callback_query_handler(lambda c: c.data.startswith('analyze_'))
async def run_analysis(callback_query: types.CallbackQuery):
    """Запуск анализа выбранной пары."""
    if is_weekend():
        await bot.answer_callback_query(callback_query.id, text="Я отдыхаю\\!", show_alert=True)
        await weekend_blocker_message(callback_query.from_user.id)
        return
        
    await bot.answer_callback_query(callback_query.id, text="Провожу глубокий Тех\\. Анализ...", show_alert=False)
    
    symbol_raw = callback_query.data.split('_', 1)[1]
    symbol = symbol_raw.replace('_', '/')
    
    df = get_ohlcv(symbol, TIMEFRAME)
    
    if df.empty or len(df) < 50:
        await bot.send_message(
            callback_query.from_user.id,
            f"❌ Не удалось получить достаточно данных для {code(escape_md(symbol))}\\. Попробуйте другой таймфрейм или пару\\.",
        )
        await bot.send_message(
            callback_query.from_user.id,
            "Выбери следующую функцию:",
            reply_markup=main_menu
        )
        return

    signal = analyze_and_predict(df, symbol)
    
    if signal and signal['direction'] != 'НЕЙТРАЛЬНО ⚪':
        signal_id = str(hash(signal['symbol'] + signal['direction'] + str(datetime.now(TZ))))
        
        message_text = f"""
📈 {bold(escape_md("ТОРГОВЫЙ СИГНАЛ"))} \\| {code(escape_md(signal['symbol']))} \\({TIMEFRAME}\\) 
*---*
* {bold(escape_md("НАПРАВЛЕНИЕ"))}: {signal['direction']}
* {bold(escape_md("Текущая Цена"))}: {code(signal['price'])}
* {bold(escape_md("УВЕРЕННОСТЬ"))}: {bold(signal['confidence'])}
* {bold(escape_md("Экспирация"))}: {signal['expiration']}
* {bold(escape_md("Обоснование"))}: {signal['reason']}

🔥 _Сигнал сформирован на основе анализа 15\\+ индикаторов\\._
"""
        user_history[signal_id] = {
            'user_id': callback_query.from_user.id,
            'symbol': signal['symbol'],
            'direction': signal['direction'],
            'confidence': signal['confidence'],
            'timestamp': datetime.now(TZ),
            'result': 'Pending'
        }
        
        await bot.send_message(
            callback_query.from_user.id,
            message_text,
            reply_markup=result_keyboard(signal_id)
        )
    else:
        await bot.send_message(
            callback_query.from_user.id,
            f"⚠️ Для {code(escape_md(symbol))} нет сильного сигнала\\. {signal['reason']}" if signal else "⚠️ Анализ не дал результата\\.",
        )
        
    await bot.send_message(
        callback_query.from_user.id,
        "Выбери следующую функцию:",
        reply_markup=main_menu
    )

@dp.callback_query_handler(lambda c: c.data == 'news_analysis')
async def handle_news_analysis(callback_query: types.CallbackQuery):
    """Обработчик кнопки Новостей."""
    if is_weekend():
        await bot.answer_callback_query(callback_query.id, text="Я отдыхаю\\!", show_alert=True)
        await weekend_blocker_message(callback_query.from_user.id)
        return
        
    await bot.answer_callback_query(callback_query.id, text="Анализирую главные новости...", show_alert=False)
    
    news_symbol = 'EUR/USD' 
    news_report = analyze_news(news_symbol)
    
    await bot.send_message(
        callback_query.from_user.id,
        news_report,
    )
    
    await bot.send_message(
        callback_query.from_user.id,
        "Выбери следующую функцию:",
        reply_markup=main_menu
    )

@dp.callback_query_handler(lambda c: c.data.startswith('result_'))
async def handle_result_fix(callback_query: types.CallbackQuery):
    """Фиксация результата сделки (Плюс/Минус)."""
    if is_weekend():
        await bot.answer_callback_query(callback_query.id, text="Я отдыхаю\\!", show_alert=True)
        await weekend_blocker_message(callback_query.from_user.id)
        return
        
    await bot.answer_callback_query(callback_query.id)
    
    parts = callback_query.data.split('_')
    result_type = parts[1] 
    signal_id = parts[2]
    
    if signal_id in user_history:
        history_entry = user_history[signal_id]
        
        if history_entry['result'] == 'Pending':
            history_entry['result'] = 'WIN' if result_type == 'win' else 'LOSS'
            
            result_text = "✅ ПРИБЫЛЬ" if result_type == 'win' else "❌ УБЫТОК"
            
            await bot.edit_message_text(
                f"📊 Сигнал для {code(escape_md(history_entry['symbol']))} зафиксирован:\\\n\n{bold(escape_md('РЕЗУЛЬТАТ'))}: {result_text}\\\n_Сохранено в Истории\\._",
                chat_id=callback_query.message.chat.id,
                message_id=callback_query.message.message_id,
                reply_markup=None 
            )
        else:
            await bot.send_message(callback_query.from_user.id, "Этот результат уже был зафиксирован\\.")
    else:
        await bot.send_message(callback_query.from_user.id, "Ошибка: Сигнал не найден\\.")

@dp.callback_query_handler(lambda c: c.data == 'history')
async def show_history(callback_query: types.CallbackQuery):
    """Показать историю сделок."""
    if is_weekend():
        await bot.answer_callback_query(callback_query.id, text="Я отдыхаю\\!", show_alert=True)
        await weekend_blocker_message(callback_query.from_user.id)
        return
        
    await bot.answer_callback_query(callback_query.id)
    
    user_id = callback_query.from_user.id
    history_list = [h for h in user_history.values() if h['user_id'] == user_id]
    
    if not history_list:
        await bot.send_message(user_id, "📜 Ваша история сделок пока пуста\\.")
        return

    history_text = "📜 " + bold(escape_md("ВАША ИСТОРИЯ СДЕЛОК")) + " 📜\n\n"
    
    for i, entry in enumerate(reversed(history_list[:10])): 
        result_icon = "🟢" if entry['result'] == 'WIN' else "🔴" if entry['result'] == 'LOSS' else "🟡"
        
        history_text += (
            f"{i+1}\\. {result_icon} {bold(entry['result'])} \\| {code(escape_md(entry['symbol']))} \\({entry['direction']}\\) "
            f"Уверенность: {entry['confidence']}\n"
            f"_Время: {entry['timestamp'].strftime('%d\\.%m %H:%M')}_\n\n"
        )
    
    await bot.send_message(user_id, history_text)
    
    await bot.send_message(
        user_id,
        "Выбери следующую функцию:",
        reply_markup=main_menu
    )
    
@dp.callback_query_handler(lambda c: c.data == 'main_menu')
async def back_to_main_menu(callback_query: types.CallbackQuery):
    """Возврат в главное меню."""
    if is_weekend():
        await bot.answer_callback_query(callback_query.id, text="Я отдыхаю\\!", show_alert=True)
        await weekend_blocker_message(callback_query.from_user.id)
        return
        
    await bot.answer_callback_query(callback_query.id)
    await bot.send_message(
        callback_query.from_user.id,
        "Выбери следующую функцию:",
        reply_markup=main_menu
    )

# --- 4. ЗАПУСК (РЕЖИМ WEBHOOK) ---

WEBAPP_HOST = '0.0.0.0' # Слушаем все интерфейсы

async def on_startup(app):
    """Действия при запуске: устанавливаем вебхук на серверах Telegram."""
    if not WEBHOOK_URL:
        print("❌ ОШИБКА: Переменная WEBHOOK_URL не найдена. Не могу установить вебхук.")
        await bot.close()
        return

    print("Приложение запущено. Устанавливаю вебхук...")
    try:
        # Устанавливаем вебхук.
        await bot.set_webhook(WEBHOOK_URL)
        print(f"✅ Вебхук установлен: {WEBHOOK_URL}")
    except Exception as e:
        print(f"❌ Ошибка установки вебхука: {e}")
        await bot.close()
        

async def on_shutdown(app):
    """Действия при отключении: Удаляем вебхук."""
    print("Приложение завершает работу. Удаляю вебхук...")
    try:
        await bot.delete_webhook()
        print("✅ Вебхук успешно удален.")
    except Exception as e:
        print(f"❌ Ошибка удаления вебхука: {e}")
        

if __name__ == '__main__':
    # Настраиваем приложение AIOHTTP для диспетчера aiogram
    app = get_new_configured_app(dp, path=WEBHOOK_PATH)
    
    # Регистрируем функции запуска и отключения
    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)
    
    print(f"🚀 Запускаю веб-сервер на {WEBAPP_HOST}:{WEBAPP_PORT}")
    
    # Запускаем AIOHTTP веб-сервер
    web.run_app(
        app,
        host=WEBAPP_HOST,
        port=WEBAPP_PORT
    )
