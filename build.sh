#!/usr/bin/env bash
# УСТАНАВЛИВАЕМ PYTHON 3.10 РУКАМИ
pyenv install -s 3.10.13
pyenv global 3.10.13

# Устанавливаем зависимости
pip install --upgrade pip
pip install -r requirements.txt
