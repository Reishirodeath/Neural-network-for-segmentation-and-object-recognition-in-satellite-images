"""
Полный код нейросети U-Net для сегментации объектов (buildings, roads, forest)
с подробными комментариями.
"""

# Импорт необходимых библиотек
import numpy as np
import cv2
import os
import albumentations as A
import tensorflow as tf
"""

Цветовая схема масок:
- Здания(buildings): Красный[0, 0, 255]
в
BGR
- Дороги(roads): Синий[255, 0, 0]
в
BGR
- Лес(forest): Зеленый[0, 255, 0]
в
BGR
"""

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import albumentations as A
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# ---------------------------
# 1. Загрузка и подготовка данных
# ---------------------------

def load_data(img_dir, mask_dir, img_size=(256, 256), num_classes=3):
    """
Загружает изображения и соответствующие маски.
Преобразует цветные маски в one - hot encoded формат.
"""
    images = []
    masks = []

    # Цветовая кодировка классов (BGR)
    class_colors = {
        0: [0, 0, 255],    # Здания - красный
        1: [0, 255, 0],    # Лес - зеленый
        2: [255, 0, 0]     # Дороги - синий
    }

    # Получаем только файлы с изображениями
    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    mask_files = [f for f in os.listdir(mask_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    common_files = list(set(img_files) & set(mask_files))

    for img_name in common_files:
        # Загрузка изображения
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        img = cv2.resize(img, img_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Конвертация в RGB
        images.append(img)

        # Загрузка и обработка маски
        mask_path = os.path.join(mask_dir, img_name)
        mask = cv2.imread(mask_path)
        if mask is None:
            continue

        mask = cv2.resize(mask, img_size)
        mask_class = np.zeros((*img_size, num_classes), dtype=np.uint8)

        # Преобразование цветной маски в one-hot формат
        for class_idx, color in class_colors.items():
            mask_class[:, :, class_idx] = np.all(np.abs(mask - color) < 40, axis=-1).astype(np.uint8)

        masks.append(mask_class)

    return np.array(images) / 255.0, np.array(masks)

# ---------------------------
# 2. Аугментация данных
# ---------------------------

def get_augmentation():
    """
Создает набор аугментаций для изображений и масок.
Все преобразования применяются синхронно к изображениям и маскам.
"""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3)
    ], additional_targets={'mask': 'mask'})

def apply_augmentation(images, masks, augmentation):
    """
Применяет аугментацию к пакету данных
"""
    augmented_images = []
    augmented_masks = []

    for img, mask in zip(images, masks):
        augmented = augmentation(image=img, mask=mask)
        augmented_images.append(augmented['image'])
        augmented_masks.append(augmented['mask'])

    return np.array(augmented_images), np.array(augmented_masks)

# ---------------------------
# 3. Архитектура U-Net
# ---------------------------

def unet(input_size=(256, 256, 3), num_classes=3):
    """
    Создает модель U - Net с указанными параметрами
    """
    inputs = Input(input_size)

    # Энкодер
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(p1)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(p2)
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    # Центральный блок
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(p3)
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(c4)

    # Декодер
    u5 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c4)
    u5 = concatenate([u5, c3])
    c5 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(u5)
    c5 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(c5)

    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c2])
    c6 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(u6)
    c6 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c1])
    c7 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(u7)
    c7 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(c7)

    outputs = Conv2D(num_classes, (1, 1), activation='softmax')(c7)

    return Model(inputs=[inputs], outputs=[outputs])

# ---------------------------
# 4. Обучение модели
# ---------------------------

def train_model():
    """
Обучает модель U - Net с аугментацией данных
"""
    # Параметры
    IMG_SIZE = (256, 256)
    BATCH_SIZE = 8
    EPOCHS = 50
    NUM_CLASSES = 3

    # Загрузка данных
    X, y = load_data('data/images', 'data/masks', IMG_SIZE, NUM_CLASSES)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Аугментация
    aug = get_augmentation()
    X_train_aug, y_train_aug = apply_augmentation(X_train, y_train, aug)

    # Создание модели
    model = unet(input_size=(*IMG_SIZE, 3), num_classes=NUM_CLASSES)

    # Компиляция
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=NUM_CLASSES)]
    )

    # Колбэки
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            'best_model.h5',
            save_best_only=True,
            monitor='val_loss',
            mode='min'
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=5
        )
    ]

    # Обучение
    history = model.fit(
        X_train_aug, y_train_aug,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    return model, history

# ---------------------------
# 5. Визуализация результатов
# ---------------------------

def visualize_predictions(model, img_dir, num_samples=3):
    """
Визуализирует предсказания модели
"""
    test_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:num_samples]

    for img_name in test_files:
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        original_h, original_w = img.shape[:2]
        img_resized = cv2.resize(img, (256, 256))
        img_normalized = img_resized / 255.0

        # Предсказание
        pred = model.predict(np.expand_dims(img_normalized, axis=0))[0]
        class_mask = np.argmax(pred, axis=-1)

        # Создание цветной маски
        color_mask = np.zeros((256, 256, 3), dtype=np.uint8)
        color_mask[class_mask == 0] = [0, 0, 255]  # Здания
        color_mask[class_mask == 1] = [0, 255, 0]  # Лес
        color_mask[class_mask == 2] = [255, 0, 0]  # Дороги

        # Визуализация
        plt.figure(figsize=(18, 6))

        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title("Исходное изображение")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(cv2.cvtColor(cv2.resize(color_mask, (original_w, original_h)), cv2.COLOR_BGR2RGB))
        plt.title("Предсказание модели")
        plt.axis('off')

        plt.subplot(1, 3, 3)
        for i, (color, name) in enumerate(zip(
            [[0, 0, 255], [0, 255, 0], [255, 0, 0]],
            ["Здания", "Лес", "Дороги"]
        )):
            plt.plot([], [], 'o', color=np.array(color[::-1])/255, label=name, markersize=10)
        plt.legend(loc='center')
        plt.axis('off')
        plt.title("Легенда классов")

        plt.tight_layout()
        plt.show()

# ---------------------------
# 6. Основной блок
# ---------------------------

if __name__ == "__main__":
    # Проверка GPU
    print("Доступно GPU:", len(tf.config.list_physical_devices('GPU')))

    # Обучение
    model, history = train_model()

    # Визуализация
    if model is not None:
        visualize_predictions(model, 'data/test_images', num_samples=5)
# Графики обучения
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Точность (обучение)')
        plt.plot(history.history['val_accuracy'], label='Точность (валидация)')
        plt.title('Кривые точности')
        plt.xlabel('Эпоха')
        plt.ylabel('Точность')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Потери (обучение)')
        plt.plot(history.history['val_loss'], label='Потери (валидация)')
        plt.title('Кривые потерь')
        plt.xlabel('Эпоха')
        plt.ylabel('Потери')
        plt.legend()

        plt.tight_layout()
        plt.show()
