# -*- coding: utf-8 -*-
"""
Веб-приложение для классификации изображений с помощью предобученной модели VGG19.
Автоматически открывает браузер при запуске.
"""

# ==================== Импорт стандартных библиотек ====================
import os               # для работы с переменными окружения и путями к файлам
import logging          # для записи логов (информация о работе сервера)
import tempfile         # для создания временной папки для загруженных файлов
import webbrowser       # для автоматического открытия браузера
import threading        # для запуска браузера в отдельном потоке (чтобы не блокировать сервер)
import time             # для задержки перед открытием браузера (чтобы сервер успел запуститься)
from datetime import datetime  # для формирования уникальных имён файлов с меткой времени

# ==================== Импорт сторонних библиотек ====================
import numpy as np                 # для работы с многомерными массивами (тензоры изображений)
from flask import Flask, request, render_template, jsonify  # веб-фреймворк
from werkzeug.utils import secure_filename   # защита от опасных имён файлов (например, "../etc/passwd")
from tensorflow.keras.applications import VGG19   # сама модель нейросети
from tensorflow.keras.applications.vgg19 import preprocess_input, decode_predictions  # подготовка данных и расшифровка ответов
from PIL import Image               # библиотека для работы с изображениями (Pillow)

# ==================== Отключение предупреждений TensorFlow ====================
# Уровень логирования TensorFlow: 0=все, 1=INFO, 2=WARNING, 3=ERROR
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Отключаем оптимизации oneDNN (убирает предупреждения о плавающей точке)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ==================== Конфигурация приложения ====================
# Множество разрешённых расширений файлов (можно добавлять новые)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
# Максимальный размер загружаемого файла – 16 мегабайт
MAX_CONTENT_LENGTH = 16 * 1024 * 1024

# Настройка логирования: выводим сообщения уровня INFO и выше в консоль
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)   # создаём логгер с именем текущего модуля

# Создаём экземпляр Flask-приложения
app = Flask(__name__)
# Устанавливаем ограничение на размер запроса
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
# Создаём временную папку для хранения загруженных изображений (автоматически удаляется при перезагрузке)
app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()

# ==================== Загрузка модели (глобальный объект) ====================
model = None   # сначала модель не загружена

def load_model():
    """
    Загружает предобученную модель VGG19 с весами ImageNet.
    Модель загружается только один раз и сохраняется в глобальной переменной.
    """
    global model
    if model is None:
        logger.info("Загрузка VGG19 (веса ImageNet)...")
        # VGG19 – глубокая свёрточная нейросеть, обученная на 1.2 млн изображений (1000 классов)
        # weights='imagenet' указывает скачать готовые веса (около 500 МБ)
        model = VGG19(weights='imagenet')
        logger.info("Модель успешно загружена.")
    return model

# Вызываем загрузку модели сразу при старте приложения (чтобы первый запрос не ждал)
load_model()

# ==================== Вспомогательные функции ====================
def allowed_file(filename):
    """
    Проверяет, имеет ли файл разрешённое расширение.
    :param filename: имя файла (например, 'dog.jpg')
    :return: True если расширение входит в ALLOWED_EXTENSIONS, иначе False
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path, target_size=(224, 224)):
    """
    Загружает изображение, преобразует его в формат, готовый для VGG19.
    Шаги:
    1. Открыть изображение через PIL и принудительно перевести в RGB (отбрасываем альфа-канал).
    2. Изменить размер до 224×224 пикселей (требование VGG19).
    3. Преобразовать в массив NumPy (значения 0..255).
    4. Добавить размерность батча: (1, 224, 224, 3) – модель ожидает список изображений.
    5. Применить предобработку VGG19: вычитание средних значений пикселей по каналам (BGR).
    :param image_path: путь к файлу изображения
    :param target_size: целевой размер (ширина, высота)
    :return: тензор (numpy array) формы (1, 224, 224, 3)
    """
    # Открываем изображение и конвертируем в RGB (если PNG с прозрачностью, альфа-канал отбрасывается)
    img = Image.open(image_path).convert('RGB')
    # Изменяем размер
    img = img.resize(target_size)
    # Преобразуем PIL Image в numpy-массив (высота, ширина, каналы)
    img_array = np.array(img)
    # Добавляем ось батча (batch dimension) – теперь размер (1, height, width, channels)
    img_array = np.expand_dims(img_array, axis=0)
    # Нормализуем: вычитаем средние значения ImageNet (R=123.68, G=116.78, B=103.94) и меняем порядок каналов на BGR
    img_array = preprocess_input(img_array)
    return img_array

# ==================== Маршруты (routes) ====================
@app.route('/', methods=['GET'])
def index():
    """
    Корневой маршрут. Возвращает HTML-страницу с формой загрузки файла.
    Шаблон должен лежать в папке templates/index.html
    """
    # render_template ищет файл в подпапке 'templates' относительно app.py
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Обрабатывает POST-запрос с загруженным изображением.
    Ожидает поле 'file' с бинарными данными картинки.
    Возвращает JSON с топ-5 предсказаниями или сообщение об ошибке.
    """
    # 1. Проверяем, есть ли файл в запросе
    if 'file' not in request.files:
        return jsonify({'error': 'Файл не передан'}), 400   # HTTP 400 Bad Request

    file = request.files['file']

    # 2. Проверяем, что пользователь выбрал файл (а не отправил пустую форму)
    if file.filename == '':
        return jsonify({'error': 'Файл не выбран'}), 400

    # 3. Проверяем расширение файла
    if not allowed_file(file.filename):
        return jsonify({'error': f'Недопустимый тип. Разрешены: {", ".join(ALLOWED_EXTENSIONS)}'}), 400

    # 4. Безопасно сохраняем файл во временную папку
    # secure_filename удаляет опасные символы (например, '/', '..')
    filename = secure_filename(file.filename)
    # Добавляем временную метку, чтобы избежать коллизий при одновременных загрузках
    safe_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
    file.save(filepath)
    logger.info(f"Файл сохранён: {filepath}")

    # 5. Обрабатываем изображение и получаем предсказания
    try:
        # Предобработка: преобразуем картинку в тензор (1,224,224,3)
        img_tensor = preprocess_image(filepath)

        # Выполняем forward pass через модель (вероятности для 1000 классов)
        predictions = model.predict(img_tensor, verbose=0)   # verbose=0 отключает прогресс-бар

        # decode_predictions конвертирует вектор вероятностей в список из топ-5
        # результат: список кортежей (id, название_класса, вероятность)
        top5 = decode_predictions(predictions, top=5)[0]   # [0] берём первый (единственный) элемент батча

        # Преобразуем в удобный JSON-совместимый формат
        result = [{'label': label, 'probability': float(prob)} for (_, label, prob) in top5]

        return jsonify({'success': True, 'predictions': result})

    except Exception as e:
        # Логируем полную информацию об ошибке (traceback)
        logger.exception("Ошибка при предсказании")
        # Возвращаем сообщение об ошибке клиенту с HTTP 500 (Internal Server Error)
        return jsonify({'error': f'Ошибка обработки: {str(e)}'}), 500

    finally:
        # 6. Удаляем временный файл, чтобы не засорять диск
        try:
            os.remove(filepath)
            logger.debug(f"Временный файл удалён: {filepath}")
        except Exception as e:
            logger.warning(f"Не удалось удалить {filepath}: {e}")

# ==================== Автоматическое открытие браузера ====================
def open_browser():
    """
    Функция, запускаемая в отдельном потоке.
    Ждёт 1.5 секунды (чтобы сервер успел стартовать) и открывает браузер по адресу http://127.0.0.1:5000
    """
    time.sleep(1.5)   # задержка
    webbrowser.open('http://127.0.0.1:5000')

# ==================== Точка входа при запуске ====================
if __name__ == '__main__':
    # Запускаем поток для открытия браузера (daemon=True – поток завершится при закрытии основного)
    threading.Thread(target=open_browser, daemon=True).start()
    # Запускаем Flask-сервер:
    # host='0.0.0.0' – доступно с любого IP в локальной сети (можно зайти с телефона/другого ПК)
    # port=5000 – стандартный порт Flask
    # debug=True – автоматическая перезагрузка при изменении кода, подробные ошибки в браузере (для разработки)
    # threaded=True – поддержка нескольких запросов одновременно
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)