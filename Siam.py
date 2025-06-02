"""
Сиамская нейронная сеть для поиска похожих изображений
Поддерживает форматы: PGM, JPG, JPEG, PNG
Сохраняет результаты сравнения в папку results
"""

import os
import tkinter as tk
from tkinter import filedialog
import numpy as np
import tensorflow as tf
from keras import layers, Model, Sequential
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime

# ==============================================
# КОНФИГУРАЦИЯ МОДЕЛИ И ПАРАМЕТРОВ
# ==============================================

# Размер изображений для обработки (ширина, высота)
IMG_SIZE = (256, 256)  # Изменено на 256x256

# Количество цветовых каналов (3 для RGB)
CHANNELS = 3

# Размер батча для обработки данных
BATCH_SIZE = 16

# Путь к папке с обучающими данными
DATASET_PATH = r'C:\Users\maks_\PycharmProjects\PythonProject2\dataset'

# Папка для сохранения результатов
SAVE_DIR = os.path.join(os.path.dirname(__file__), 'results')

# Поддерживаемые расширения файлов
SUPPORTED_EXTENSIONS = [
    '.pgm', '.PGM',
    '.jpg', '.jpeg', '.png',
    '.JPG', '.JPEG', '.PNG'
]


# ==============================================
# АРХИТЕКТУРА НЕЙРОННОЙ СЕТИ
# ==============================================

def create_model(input_shape):
    """
    Создает архитектуру сиамской нейронной сети
    Возвращает:
    - model: Полная модель для обучения
    - base_network: Базовая сеть для извлечения признаков
    """
    # Базовая сеть для обработки изображений
    base_network = Sequential([
        # Входной слой (формат изображения)
        layers.Input(shape=input_shape),

        # Сверточные слои для извлечения признаков
        layers.Conv2D(32, 3, activation='relu'),  # 32 фильтра 3x3
        layers.MaxPooling2D(),  # Уменьшение размерности
        layers.Conv2D(64, 3, activation='relu'),  # 64 фильтра 3x3
        layers.MaxPooling2D(),  # Уменьшение размерности

        # Преобразование в вектор признаков
        layers.Flatten(),
        layers.Dense(256, activation='relu')  # Полносвязный слой
    ])

    # Два параллельных входа для сиамской архитектуры
    input_a = layers.Input(shape=input_shape)
    input_b = layers.Input(shape=input_shape)

    # Генерация векторных представлений (эмбеддингов)
    embedding_a = base_network(input_a)
    embedding_b = base_network(input_b)

    # Слой для вычисления евклидова расстояния между эмбеддингами
    distance = layers.Lambda(
        lambda x: tf.sqrt(tf.reduce_sum(tf.square(x[0] - x[1]), axis=1)),
        name='distance_layer'
    )([embedding_a, embedding_b])

    # Собираем полную модель
    model = Model(
        inputs=[input_a, input_b],
        outputs=distance,
        name='siamese_model'
    )

    return model, base_network


# ==============================================
# КЛАСС ДЛЯ РАБОТЫ С МОДЕЛЬЮ
# ==============================================

class SiameseModel:
    def __init__(self, model_path=None):
        """
        Инициализация модели
        model_path: Путь к сохраненным весам модели (если есть)
        """
        # Создаем модель с правильным порядком размерностей (высота, ширина)
        self.model, self.base_network = create_model(
            (IMG_SIZE[1], IMG_SIZE[0], CHANNELS)  # Исправлен порядок размеров
        )

        # Загрузка весов если указан путь
        if model_path:
            self.model.load_weights(model_path)

        # Загрузка и подготовка данных
        self.dataset, self.image_paths = self.load_dataset()

        # Проверка наличия данных
        if not self.image_paths:
            raise ValueError("В папке dataset нет изображений!")

        # Генерация векторных представлений для всего датасета
        self.embeddings = self.base_network.predict(
            self.dataset.batch(BATCH_SIZE),
            verbose=1
        )

    def load_dataset(self):
        """
        Загрузка и предобработка изображений из папки dataset
        Возвращает:
        - tf.data.Dataset: Подготовленный датасет
        - list: Список путей к изображениям
        """
        image_paths = []
        # Рекурсивный обход папки dataset
        for root, _, files in os.walk(DATASET_PATH):
            for file in files:
                # Проверка расширения файла
                ext = os.path.splitext(file)[1].lower()
                if ext in SUPPORTED_EXTENSIONS:
                    path = os.path.join(root, file)
                    # Пропускаем пустые файлы
                    if os.path.getsize(path) > 0:
                        image_paths.append(path)

        # Создаем tf.data.Dataset из путей к файлам
        dataset = tf.data.Dataset.from_tensor_slices(image_paths)

        # Преобразование путей в изображения
        dataset = dataset.map(
            lambda x: tf.py_function(self.load_image, [x], tf.float32),
            num_parallel_calls=tf.data.AUTOTUNE  # Параллельная обработка
        ).filter(lambda x: x is not None)  # Фильтрация ошибок

        # Оптимизация производительности
        return dataset.cache().prefetch(tf.data.AUTOTUNE), image_paths

    def load_image(self, path):
        """
        Загрузка и предобработка одного изображения
        Возвращает:
        - tf.Tensor: Нормализованный и измененный размер тензор изображения
        """
        try:
            # Декодирование пути из тензора
            path_str = path.numpy().decode('utf-8')

            # Загрузка изображения
            img = Image.open(path_str)

            # Конвертация в RGB при необходимости
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Преобразование в numpy array и нормализация
            img = np.array(img, dtype=np.float32) / 255.0

            # Изменение размера с правильным порядком (высота, ширина)
            img = tf.image.resize(img, (IMG_SIZE[1], IMG_SIZE[0]))  # Исправлено

            return img
        except Exception as e:
            print(f"Ошибка загрузки {path_str}: {str(e)}")
            return None

    def find_similar(self, query_path):
        """
        Поиск наиболее похожего изображения в датасете
        Возвращает:
        - tuple: (путь к похожему изображению, расстояние)
        """
        # Предобработка входного изображения
        query_img = self.preprocess_image(query_path)
        if query_img is None:
            return None

        # Получение векторного представления для запроса
        query_embedding = self.base_network.predict(
            tf.expand_dims(query_img, axis=0),
            verbose=0
        )

        # Вычисление расстояний до всех изображений в датасете
        distances = tf.norm(self.embeddings - query_embedding, axis=1)

        # Находим индекс минимального расстояния
        index = np.argmin(distances)

        return (self.image_paths[index], distances[index].numpy())

    def preprocess_image(self, path):
        """
        Предобработка изображения для предсказания
        Возвращает:
        - tf.Tensor: Подготовленный тензор изображения
        """
        try:
            img = Image.open(path)
            # Конвертация в RGB
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Нормализация и ресайз с правильным порядком (высота, ширина)
            img = np.array(img, dtype=np.float32) / 255.0
            img = tf.image.resize(img, (IMG_SIZE[1], IMG_SIZE[0]))  # Исправлено

            return img
        except Exception as e:
            print(f"Ошибка обработки {path}: {str(e)}")
            return None

    def save_pair(self, query_path, result):
        """
        Сохраняет пару изображений в папку results
        query_path: путь к исходному изображению
        result: кортеж (путь к результату, расстояние)
        """
        try:
            # Создаем папку если не существует
            os.makedirs(SAVE_DIR, exist_ok=True)

            # Генерируем уникальный префикс из текущего времени
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Сохраняем исходное изображение
            query_name = f"query_{timestamp}{os.path.splitext(query_path)[1]}"
            query_save_path = os.path.join(SAVE_DIR, query_name)
            Image.open(query_path).save(query_save_path)

            # Сохраняем результат с указанием расстояния
            result_path, distance = result
            result_name = f"result_{timestamp}_{distance:.4f}{os.path.splitext(result_path)[1]}"
            result_save_path = os.path.join(SAVE_DIR, result_name)
            Image.open(result_path).save(result_save_path)

            print(f"Изображения сохранены:\n{query_save_path}\n{result_save_path}")
            return True
        except Exception as e:
            print(f"Ошибка сохранения: {str(e)}")
            return False
    def plot_training_metrics(self):
        """Визуализация метрик обучения"""
        plt.figure(figsize=(12, 5))
        
        # График точности
        plt.subplot(1, 2, 1)
        plt.plot(self.history['accuracy'], 'b-o', linewidth=2)
        plt.title('Динамика точности', fontsize=14)
        plt.xlabel('Эпохи', fontsize=12)
        plt.ylabel('Точность', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.ylim(0, 1.1)

        # График потерь
        plt.subplot(1, 2, 2)
        plt.plot(self.history['loss'], 'r-s', linewidth=2)
        plt.title('Динамика потерь', fontsize=14)
        plt.xlabel('Эпохи', fontsize=12)
        plt.ylabel('Потери', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.ylim(-0.1, 1.1)

        plt.suptitle('Метрики обучения модели', fontsize=16)
        plt.tight_layout()
        plt.show() 
    def plot_and_save_results(self, query_path, result):
        """
        Визуализирует и сохраняет результаты сравнения
        """
        # Создаем график
        plt.figure(figsize=(8, 4))

        # Отображаем исходное изображение
        plt.subplot(1, 2, 1)
        img = Image.open(query_path).convert('RGB')
        plt.imshow(img)
        plt.title("Исходное изображение")
        plt.axis('off')

        # Отображаем результат
        plt.subplot(1, 2, 2)
        result_path, distance = result
        img = Image.open(result_path).convert('RGB')
        plt.imshow(img)
        plt.title(f"Расстояние: {distance:.2f}")
        plt.axis('off')

        # Настраиваем и показываем график
        plt.tight_layout()
        plt.show()

        # Сохраняем изображения
        self.save_pair(query_path, result)


# ==============================================
# ОСНОВНАЯ ЛОГИКА ПРОГРАММЫ
# ==============================================

if __name__ == "__main__":
    try:
        # Инициализация модели
        model = SiameseModel()
        print("Модель успешно инициализирована!")

        # Настройка GUI для выбора файла
        root = tk.Tk()
        root.withdraw()  # Скрываем основное окно

        # Выбор файла через диалоговое окно
        file_path = filedialog.askopenfilename(
            title="Выберите изображение",
            filetypes=[("Изображения", "*.pgm;*.jpg;*.jpeg;*.png")]
        )

        if file_path:
            # Поиск похожего изображения
            result = model.find_similar(file_path)

            if result:
                # Визуализация и сохранение результатов
                model.plot_and_save_results(file_path, result)
            else:
                print("Похожих изображений не найдено!")

    except Exception as e:
        print(f"Критическая ошибка: {str(e)}")
