@echo off
:: Отключаем вывод команд (чистый экран)
:: Устанавливаем кодировку UTF-8 для поддержки русских букв в консоли
chcp 65001 >nul
:: Заголовок окна
title VGG19 Image Classifier
:: Переходим в папку, где находится сам bat-файл (%~dp0)
:: Это гарантирует, что скрипт работает из папки проекта, даже если вызван из другого места
cd /d "%~dp0"

echo Starting server...

:: Если папки виртуального окружения venv не существует – создаём её
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    :: Активируем окружение
    call venv\Scripts\activate
    :: Устанавливаем зависимости из requirements.txt
    pip install -r requirements.txt
) else (
    :: Если venv уже есть – просто активируем
    call venv\Scripts\activate
)

:: Запускаем приложение
python app.py

:: Если приложение завершится (например, по Ctrl+C), показываем сообщение и ждём нажатия клавиши
pause