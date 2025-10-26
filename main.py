import telebot
from telebot import types
import random
import time

# === Настройки ===
BOT_TOKEN = "8315023641:AAHLJg0kDC0xtTpgVh8wCd5st1Yjpoe39GM "  # вставь свой токен сюда
bot = telebot.TeleBot(BOT_TOKEN)

# === Списки активов ===
currency_pairs = [
    "EURUSD OTC", "GBPUSD OTC", "USDJPY OTC", "USDCHF OTC", "AUDUSD OTC",
    "NZDUSD OTC", "USDCAD OTC", "EURGBP OTC", "EURJPY OTC", "EURCHF OTC",
    "GBPJPY OTC", "AUDJPY OTC", "EURCAD OTC", "EURNZD OTC", "AUDNZD OTC",
    "CADJPY OTC", "CHFJPY OTC", "NZDJPY OTC", "GBPCAD OTC", "GBPCHF OTC"
]

stocks = [
    "APPLE OTC", "CISCO OTC", "AMERICAN OTC", "INTEL OTC", "TESLA OTC",
    "AMAZON OTC", "ADVANCED MICRO DEVICES OTC", "ALIBABA OTC", "NETFLIX OTC",
    "BOEING COMPANY OTC", "FACEBOOK INC OTC", "GOOGLE OTC", "MICROSOFT OTC",
    "NIKE OTC", "COCA-COLA OTC", "SONY OTC", "FORD OTC", "ADOBE OTC",
    "VISA OTC", "PEPSICO OTC"
]

crypto = [
    "BITCOIN OTC", "BITCOIN ETF", "ETHERIUM OTC", "POLYGON OTC",
    "CARDANO OTC", "TRON OTC", "TONCOIN OTC", "BNB OTC", "CHAINLINK OTC",
    "SOLANA OTC", "DOGECOIN OTC", "POLKADOT OTC"
]

# === Команда /start ===
@bot.message_handler(commands=['start'])
def start(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    btn1 = types.KeyboardButton("💱 Валютные пары")
    btn2 = types.KeyboardButton("📈 Акции")
    btn3 = types.KeyboardButton("💰 Криптовалюты")
    markup.add(btn1, btn2, btn3)
    bot.send_message(
        message.chat.id,
        "👋 Привет! Я бот-сигнальщик. Выбери категорию, чтобы получить торговый сигнал:",
        reply_markup=markup
    )

# === Обработка кнопок ===
@bot.message_handler(func=lambda message: message.text in ["💱 Валютные пары", "📈 Акции", "💰 Криптовалюты"])
def choose_asset(message):
    if message.text == "💱 Валютные пары":
        show_assets(message, currency_pairs, "валютную пару")
    elif message.text == "📈 Акции":
        show_assets(message, stocks, "акцию")
    elif message.text == "💰 Криптовалюты":
        show_assets(message, crypto, "криптовалюту")

def show_assets(message, assets, asset_type):
    markup = types.InlineKeyboardMarkup()
    for asset in assets[:5]:
        markup.add(types.InlineKeyboardButton(asset, callback_data=asset))
    markup.add(types.InlineKeyboardButton("➡️ Вперед", callback_data=f"next_{asset_type}_5"))
    bot.send_message(message.chat.id, f"Выбери {asset_type}:", reply_markup=markup)

# === Перелистывание активов ===
@bot.callback_query_handler(func=lambda call: call.data.startswith("next_"))
def next_assets(call):
    parts = call.data.split("_")
    asset_type = parts[1]
    start_index = int(parts[2])
    assets = currency_pairs if asset_type == "валютную" else stocks if asset_type == "акцию" else crypto
    markup = types.InlineKeyboardMarkup()
    for asset in assets[start_index:start_index + 5]:
        markup.add(types.InlineKeyboardButton(asset, callback_data=asset))
    if start_index + 5 < len(assets):
        markup.add(types.InlineKeyboardButton("➡️ Вперед", callback_data=f"next_{asset_type}_{start_index + 5}"))
    markup.add(types.InlineKeyboardButton("⬅️ Назад", callback_data=f"back_{asset_type}_{max(start_index - 5, 0)}"))
    bot.edit_message_text(chat_id=call.message.chat.id, message_id=call.message.message_id, text=f"Выбери {asset_type}:", reply_markup=markup)

@bot.callback_query_handler(func=lambda call: call.data.startswith("back_"))
def back_assets(call):
    parts = call.data.split("_")
    asset_type = parts[1]
    start_index = int(parts[2])
    assets = currency_pairs if asset_type == "валютную" else stocks if asset_type == "акцию" else crypto
    markup = types.InlineKeyboardMarkup()
    for asset in assets[start_index:start_index + 5]:
        markup.add(types.InlineKeyboardButton(asset, callback_data=asset))
    if start_index > 0:
        markup.add(types.InlineKeyboardButton("⬅️ Назад", callback_data=f"back_{asset_type}_{max(start_index - 5, 0)}"))
    if start_index + 5 < len(assets):
        markup.add(types.InlineKeyboardButton("➡️ Вперед", callback_data=f"next_{asset_type}_{start_index + 5}"))
    bot.edit_message_text(chat_id=call.message.chat.id, message_id=call.message.message_id, text=f"Выбери {asset_type}:", reply_markup=markup)

# === Сигналы ===
@bot.callback_query_handler(func=lambda call: call.data in currency_pairs + stocks + crypto)
def send_signal(call):
    direction = random.choice(["📈 ВВЕРХ", "📉 ВНИЗ"])
    duration = random.choice(["1 минута", "3 минуты", "5 минут"])
    bot.send_message(call.message.chat.id, f"⚡ Сигнал для {call.data}\nНаправление: {direction}\nВремя: {duration}")

    # симуляция результата
    time.sleep(3)
    result = random.choice(["✅ Сигнал в плюс", "❌ Сигнал в минус"])
    bot.send_message(call.message.chat.id, f"📊 Результат: {result}")

    # кнопка нового сигнала
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    markup.add(types.KeyboardButton("📥 Получить новый сигнал"))
    bot.send_message(call.message.chat.id, "Хочешь получить новый сигнал?", reply_markup=markup)

@bot.message_handler(func=lambda message: message.text == "📥 Получить новый сигнал")
def new_signal(message):
    start(message)

# === Запуск ===
bot.polling(none_stop=True)
