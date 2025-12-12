# bot_config.py
# ⚠ Не пушим в публичный репозиторий с токеном

import os

# Получаем токен из Environment Variable (рекомендовано для Render)
TG_TOKEN = os.getenv("TG_TOKEN") or "ВАШ_TELEGRAM_TOKEN"

# Сколько свечей брать для анализа (по умолчанию 500)
CANDLES_LIMIT = 500
