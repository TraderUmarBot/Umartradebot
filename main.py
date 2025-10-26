
# main.py
# Требует: pyTelegramBotAPI (pip install pyTelegramBotAPI)
import os
import time
import random
import threading
from datetime import datetime
from typing import List

import telebot
from telebot import types

# -----------------------
# 8315023641:AAHLJg0kDC0xtTpgVh8wCd5st1Yjpoe39GM
# -----------------------
BOT_TOKEN = "8315023641:AAHLJg0kDC0xtTpgVh8wCd5st1Yjpoe39GM"  # <-- замените на ваш токен, в кавычках
bot = telebot.TeleBot(BOT_TOKEN, parse_mode='Markdown')

# -----------------------
# Настройки / списки
# -----------------------
PAGE_SIZE = 5
EXPIRATIONS = ["30s", "1m", "2m", "3m"]

CURRENCIES = [
    "EUR/USD OTC","GBP/USD OTC","USD/JPY OTC","AUD/USD OTC","USD/CHF OTC",
    "EUR/JPY OTC","GBP/JPY OTC","NZD/USD OTC","EUR/GBP OTC","CAD/JPY OTC",
    "USD/CAD OTC","AUD/JPY OTC","EUR/AUD OTC","GBP/AUD OTC","EUR/NZD OTC",
    "AUD/NZD OTC","CAD/CHF OTC","CHF/JPY OTC","NZD/JPY OTC","GBP/CAD OTC"
]

STOCKS = [
    "APPLE OTC","CISCO OTC","AMERICAN OTC","INTEL OTC","TESLA OTC",
    "AMAZON OTC","ADVANCED MICRO DEVICES OTC","ALIBABA OTC","NETFLIX OTC",
    "BOEING COMPANY OTC","FACEBOOK INC OTC"
]

CRYPTOS = [
    "BITCOIN OTC","BITCOIN ETF","ETHERIUM OTC","POLYGON OTC","CARDANO OTC",
    "TRON OTC","TONCOIN OTC","BNB OTC","CHAINLINK OTC","SOLANA OTC",
    "DOGECOIN OTC","POLKADOT OTC"
]

# -----------------------
# Утилиты
# -----------------------
def make_page_keyboard(items: List[str], page: int, prefix: str) -> types.InlineKeyboardMarkup:
    """
    prefix: 'pair' | 'stock' | 'crypto'
    builds inline keyboard with items[page*PAGE_SIZE : page*PAGE_SIZE+PAGE_SIZE]
    """
    start = page * PAGE_SIZE
    end = min(len(items), start + PAGE_SIZE)
    kb = types.InlineKeyboardMarkup()
    buttons = []
    for i in range(start, end):
        buttons.append([types.InlineKeyboardButton(items[i], callback_data=f"{prefix}_idx_{i}")])
    nav_row = []
    if page > 0:
        nav_row.append(types.InlineKeyboardButton("⬅ Назад", callback_data=f"{prefix}_page_{page-1}"))
    if end < len(items):
        nav_row.append(types.InlineKeyboardButton("Вперед ➡", callback_data=f"{prefix}_page_{page+1}"))
    if nav_row:
        buttons.append(nav_row)
    kb.rows = []  # ensure fresh
    for row in buttons:
        kb.add(*row)
    return kb

def expiration_to_seconds(exp: str) -> int:
    e = exp.lower().strip()
    if e.endswith('s'):
        return int(e[:-1])
    if e.endswith('m'):
        return int(e[:-1]) * 60
    return 60

def deterministic_signal(instrument: str) -> str:
    """
    Deterministic-ish signal: uses hash of instrument + current minute to produce stable decisions during that minute.
    Returns "Вверх ↑" or "Вниз ↓".
    """
    minute = int(time.time() // 60)
    seed = hash(f"{instrument}|{minute}")
    rnd = random.Random(seed)
    return "Вверх ↑" if rnd.random() > 0.5 else "Вниз ↓"

# -----------------------
# Команда /start
# -----------------------
@bot.message_handler(commands=['start'])
def cmd_start(message: types.Message):
    kb = types.InlineKeyboardMarkup()
    kb.add(types.InlineKeyboardButton("💱 Валюты", callback_data="cat_pair_page_0"))
    kb.add(types.InlineKeyboardButton("📈 Акции", callback_data="cat_stock_page_0"))
    kb.add(types.InlineKeyboardButton("🪙 Крипто", callback_data="cat_crypto_page_0"))
    bot.send_message(message.chat.id, "Привет! Выберите категорию:", reply_markup=kb)

# -----------------------
# Обработка callback'ов
# -----------------------
@bot.callback_query_handler(func=lambda c: True)
def callback_handler(call: types.CallbackQuery):
    data = call.data

    # category pages
    if data.startswith("cat_pair_page_") or data.startswith("cat_stock_page_") or data.startswith("cat_crypto_page_"):
        page = int(data.split("_")[-1])
        if "pair" in data:
            bot.edit_message_text(chat_id=call.message.chat.id,
                                  message_id=call.message.message_id,
                                  text="Выберите валютную пару:",
                                  reply_markup=make_page_keyboard(CURRENCIES, page, "pair"))
        elif "stock" in data:
            bot.edit_message_text(chat_id=call.message.chat.id,
                                  message_id=call.message.message_id,
                                  text="Выберите акцию:",
                                  reply_markup=make_page_keyboard(STOCKS, page, "stock"))
        else:
            bot.edit_message_text(chat_id=call.message.chat.id,
                                  message_id=call.message.message_id,
                                  text="Выберите крипто инструмент:",
                                  reply_markup=make_page_keyboard(CRYPTOS, page, "crypto"))
        bot.answer_callback_query(call.id)
        return

    # page navigation for prefix_page_N
    if data.startswith("pair_page_") or data.startswith("stock_page_") or data.startswith("crypto_page_"):
        page = int(data.split("_")[-1])
        if data.startswith("pair_page_"):
            bot.edit_message_text(chat_id=call.message.chat.id, message_id=call.message.message_id,
                                  text="Выберите валютную пару:", reply_markup=make_page_keyboard(CURRENCIES, page, "pair"))
        elif data.startswith("stock_page_"):
            bot.edit_message_text(chat_id=call.message.chat.id, message_id=call.message.message_id,
                                  text="Выберите акцию:", reply_markup=make_page_keyboard(STOCKS, page, "stock"))
        else:
            bot.edit_message_text(chat_id=call.message.chat.id, message_id=call.message.message_id,
                                  text="Выберите крипто инструмент:", reply_markup=make_page_keyboard(CRYPTOS, page, "crypto"))
        bot.answer_callback_query(call.id)
        return

    # item selected: pair_idx_i or stock_idx_i or crypto_idx_i
    if data.startswith("pair_idx_") or data.startswith("stock_idx_") or data.startswith("crypto_idx_"):
        idx = int(data.split("_")[-1])
        if data.startswith("pair_idx_"):
            instrument = CURRENCIES[idx]
            back_cb = "cat_pair_page_0"
        elif data.startswith("stock_idx_"):
            instrument = STOCKS[idx]
            back_cb = "cat_stock_page_0"
        else:
            instrument = CRYPTOS[idx]
            back_cb = "cat_crypto_page_0"

        # store chosen instrument in user's session (simple)
        # telebot doesn't have context, use in-memory dict keyed by user id
        user_id = str(call.from_user.id)
        user_chosen_instrument[user_id] = instrument

        # build expirations keyboard
        kb = types.InlineKeyboardMarkup()
        for e in EXPIRATIONS:
            kb.add(types.InlineKeyboardButton(e, callback_data=f"exp_{e}"))
        kb.add(types.InlineKeyboardButton("⬅ Назад к списку", callback_data=back_cb))

        bot.edit_message_text(chat_id=call.message.chat.id,
                              message_id=call.message.message_id,
                              text=f"Вы выбрали: *{instrument}*\n\nВыберите экспирацию:",
                              reply_markup=kb)
        bot.answer_callback_query(call.id)
        return

    # expiration chosen
    if data.startswith("exp_"):
        exp = data.split("_", 1)[1]
        user_id = str(call.from_user.id)
        instrument = user_chosen_instrument.get(user_id, "—")
        signal = deterministic_signal(instrument)
        # send a new message with the opened signal and keep message id to edit later
        sent = bot.send_message(chat_id=call.message.chat.id,
                                text=(f"🎯 *Сигнал открыт!*\n\nИнструмент: *{instrument}*\n"
                                      f"Экспирация: *{exp}*\nСигнал: *{signal}*\n\n⏳ Ждём завершения..."))
        seconds = expiration_to_seconds(exp)
        # schedule result
        t = threading.Timer(seconds, finalize_result, args=(call.message.chat.id, sent.message_id, instrument, exp, signal, user_id))
        t.daemon = True
        t.start()
        bot.answer_callback_query(call.id)
        return

    # after result buttons
    if data == "new_signal":
        # show categories again
        kb = types.InlineKeyboardMarkup()
        kb.add(types.InlineKeyboardButton("💱 Валюты", callback_data="cat_pair_page_0"))
        kb.add(types.InlineKeyboardButton("📈 Акции", callback_data="cat_stock_page_0"))
        kb.add(types.InlineKeyboardButton("🪙 Крипто", callback_data="cat_crypto_page_0"))
        bot.edit_message_text(chat_id=call.message.chat.id, message_id=call.message.message_id,
                              text="Выберите категорию:", reply_markup=kb)
        bot.answer_callback_query(call.id)
        return

    # choose_other button
    if data == "choose_other":
        user_id = str(call.from_user.id)
        instr = user_chosen_instrument.get(user_id)
        if instr in CURRENCIES:
            bot.edit_message_text(chat_id=call.message.chat.id, message_id=call.message.message_id,
                                  text="Выберите валютную пару:", reply_markup=make_page_keyboard(CURRENCIES, 0, "pair"))
        elif instr in STOCKS:
            bot.edit_message_text(chat_id=call.message.chat.id, message_id=call.message.message_id,
                                  text="Выберите акцию:", reply_markup=make_page_keyboard(STOCKS, 0, "stock"))
        elif instr in CRYPTOS:
            bot.edit_message_text(chat_id=call.message.chat.id, message_id=call.message.message_id,
                                  text="Выберите крипто:", reply_markup=make_page_keyboard(CRYPTOS, 0, "crypto"))
        else:
            kb = types.InlineKeyboardMarkup()
            kb.add(types.InlineKeyboardButton("💱 Валюты", callback_data="cat_pair_page_0"))
            kb.add(types.InlineKeyboardButton("📈 Акции", callback_data="cat_stock_page_0"))
            kb.add(types.InlineKeyboardButton("🪙 Крипто", callback_data="cat_crypto_page_0"))
            bot.edit_message_text(chat_id=call.message.chat.id, message_id=call.message.message_id,
                                  text="Выберите категорию:", reply_markup=kb)
        bot.answer_callback_query(call.id)
        return

    # fallback
    bot.answer_callback_query(call.id, text="Неизвестная команда.")

# -----------------------
# Хранилище выбранных инструментов для пользователей (в памяти)
# -----------------------
user_chosen_instrument = {}

# -----------------------
# Финализатор результата (через таймер)
# -----------------------
def finalize_result(chat_id: int, message_id: int, instrument: str, exp: str, signal: str, user_id: str):
    """
    Эмулирует закрытие сделки: даёт цену открытия (прибл.) и закрытия, определяет Плюс/Минус корректно.
    """
    try:
        # approximate price_open from deterministic pseudo-random
        seed = abs(hash(f"{instrument}|open"))
        rnd = random.Random(seed)
        price_open = round(1 + rnd.uniform(0.0001, 0.005), 6)

        # price_close moves in direction of signal (to simulate realistic probability)
        move = random.uniform(0.00005, 0.004)
        if signal == "Вверх ↑":
            price_close = round(price_open + move, 6)
        else:
            price_close = round(price_open - move, 6)

        # determine correct result
        if (signal == "Вверх ↑" and price_close > price_open) or (signal == "Вниз ↓" and price_close < price_open):
            result_text = "Плюс ✅"
        else:
            result_text = "Минус ❌"

        # build reply markup with next actions
        kb = types.InlineKeyboardMarkup()
        kb.add(types.InlineKeyboardButton("📊 Получить новый сигнал", callback_data="new_signal"))
        kb.add(types.InlineKeyboardButton("🔁 Выбрать другой инструмент", callback_data="choose_other"))

        text = (f"✅ *Сделка завершена!*\n\nИнструмент: *{instrument}*\n"
                f"Экспирация: *{exp}*\nСигнал: *{signal}*\nРезультат: *{result_text}*\n\n"
                f"_Цена открытия:_ `{price_open:.6f}`\n_Цена закрытия:_ `{price_close:.6f}`")

        # edit message
        try:
            bot.edit_message_text(chat_id=chat_id, message_id=message_id, text=text, reply_markup=kb)
        except Exception:
            # if edit fails (message changed), send new message
            bot.send_message(chat_id=chat_id, text=text, reply_markup=kb)
    except Exception as e:
        print("finalize_result error:", e)

# -----------------------
# Запуск polling
# -----------------------
if __name__ == "__main__":
    if BOT_TOKEN == "8315023641:AAHLJg0kDC0xtTpgVh8wCd5st1Yjpoe39GM " or not BOT_TOKEN:
        print("Укажи BOT_TOKEN в коде перед запуском.")
    else:
        print("Бот запущен. Ctrl+C для остановки.")
        bot.infinity_polling(timeout=20, long_polling_timeout = 5)
