import telebot
import random
import time
from threading import Thread
import os

TOKEN = os.getenv("BOT_TOKEN") 

bot = telebot.TeleBot(TOKEN)

# --- СПИСОК АКТИВОВ ---
currency_pairs = [
    "EUR/USD OTC", "GBP/USD OTC", "USD/JPY OTC", "AUD/USD OTC", "USD/CHF OTC",
    "EUR/JPY OTC", "GBP/JPY OTC", "NZD/USD OTC", "EUR/GBP OTC", "CAD/JPY OTC",
    "USD/CAD OTC", "AUD/JPY OTC", "EUR/AUD OTC", "GBP/AUD OTC", "EUR/NZD OTC",
    "AUD/NZD OTC", "CAD/CHF OTC", "CHF/JPY OTC", "NZD/JPY OTC", "GBP/CAD OTC"
]

crypto_pairs = [
    "BITCOIN OTC", "BITCOIN ETF", "ETHERIUM OTC", "POLYGON OTC", "CARDANO OTC",
    "TRON OTC", "TONCOIN OTC", "BNB OTC", "CHAINLINK OTC", "SOLANA OTC",
    "DOGECOIN OTC", "POLKADOT OTC"
]

stock_pairs = [
    "APPLE OTC", "CISCO OTC", "AMERICAN OTC", "INTEL OTC", "TESLA OTC",
    "AMAZON OTC", "ADVANCED MICRO DEVICES OTC", "ALIBABA OTC", "NETFLIX OTC",
    "BOEING COMPANY OTC", "FACEBOOK INC OTC"
]

# --- ФУНКЦИИ ДЛЯ АНАЛИЗА ---
def analyze_market(asset):
    """Имитация анализа рынка с использованием индикаторов"""
    direction = random.choice(["ВВЕРХ", "ВНИЗ"])
    accuracy = random.randint(80, 95)  # Проходимость
    return direction, accuracy

# --- ГЛАВНОЕ МЕНЮ ---
@bot.message_handler(commands=['start'])
def start(message):
    markup = telebot.types.ReplyKeyboardMarkup(resize_keyboard=True)
    markup.add("💱 Валютные пары", "📈 Акции", "💰 Криптовалюты")
    bot.send_message(message.chat.id, "👋 Привет! Я твой торговый бот-сигнальщик 📊\n\nВыбери категорию:", reply_markup=markup)

# --- ВЫБОР АКТИВА ---
@bot.message_handler(func=lambda m: m.text in ["💱 Валютные пары", "📈 Акции", "💰 Криптовалюты"])
def choose_asset(message):
    category = message.text
    if category == "💱 Валютные пары":
        assets = currency_pairs
    elif category == "📈 Акции":
        assets = stock_pairs
    else:
        assets = crypto_pairs

    markup = telebot.types.InlineKeyboardMarkup(row_width=2)
    for a in assets[:5]:
        markup.add(telebot.types.InlineKeyboardButton(a, callback_data=f"asset_{a}"))
    markup.add(
        telebot.types.InlineKeyboardButton("➡ Вперед", callback_data=f"next_{category}_1")
    )
    bot.send_message(message.chat.id, f"Выбери актив из категории {category}:", reply_markup=markup)

# --- ПАГИНАЦИЯ ---
@bot.callback_query_handler(func=lambda c: c.data.startswith("next_"))
def next_assets(call):
    parts = call.data.split("_")
    category = parts[1]
    page = int(parts[2])

    assets = {
        "💱 Валютные пары": currency_pairs,
        "📈 Акции": stock_pairs,
        "💰 Криптовалюты": crypto_pairs
    }[category]

    start_idx = page * 5
    end_idx = start_idx + 5

    markup = telebot.types.InlineKeyboardMarkup(row_width=2)
    for a in assets[start_idx:end_idx]:
        markup.add(telebot.types.InlineKeyboardButton(a, callback_data=f"asset_{a}"))

    if start_idx > 0:
        markup.add(telebot.types.InlineKeyboardButton("⬅ Назад", callback_data=f"next_{category}_{page-1}"))
    if end_idx < len(assets):
        markup.add(telebot.types.InlineKeyboardButton("➡ Вперед", callback_data=f"next_{category}_{page+1}"))

    bot.edit_message_reply_markup(call.message.chat.id, call.message.message_id, reply_markup=markup)

# --- АНАЛИЗ АКТИВА ---
@bot.callback_query_handler(func=lambda c: c.data.startswith("asset_"))
def signal(call):
    asset = call.data.replace("asset_", "")
    direction, accuracy = analyze_market(asset)
    msg = f"""
📊 Сигнал по активу: *{asset}*
📈 Направление: *{direction}*
⏱ Время экспирации: *1 минута*
🎯 Точность сигнала: *{accuracy}%*
"""
    markup = telebot.types.ReplyKeyboardMarkup(resize_keyboard=True)
    markup.add("💹 Получить новый сигнал", "🏠 Главное меню")
    bot.send_message(call.message.chat.id, msg, parse_mode="Markdown", reply_markup=markup)

# --- НОВЫЙ СИГНАЛ ---
@bot.message_handler(func=lambda m: m.text == "💹 Получить новый сигнал")
def new_signal(message):
    start(message)

# --- ГЛАВНОЕ МЕНЮ ---
@bot.message_handler(func=lambda m: m.text == "🏠 Главное меню")
def to_main(message):
    start(message)

# --- ЗАПУСК БОТА ---
def run_bot():
    while True:
        try:
            bot.polling(none_stop=True, timeout=60)
        except Exception as e:
            print("Ошибка:", e)
            time.sleep(5)

Thread(target=run_bot).start()
