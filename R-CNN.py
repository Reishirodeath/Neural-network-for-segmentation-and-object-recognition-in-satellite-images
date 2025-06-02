"""
Скрипт для детекции объектов с использованием предобученной TensorFlow модели.
Основной функционал:
- Загрузка frozen модели TensorFlow
- Обработка изображений и выполнение детекции
- Фильтрация пересекающихся bbox с помощью метрики IoU
- Сохранение результатов в файл
"""

import numpy as np
import os
import tensorflow as tf
import cv2
import math
import glob

def iou_inter(new_box, iou_box, pred_class, NUM_CLASSES):
    """
    Вычисление Intersection over Union (IoU) для нового бокса относительно всех существующих
    Args:
        new_box (list): Координаты нового бокса [xmin, ymin, xmax, ymax]
        iou_box (np.array): Массив существующих боксов
        pred_class (np.array): Массив меток классов
        NUM_CLASSES (int): Количество классов
    Returns:
        float: Максимальное значение IoU для нового бокса
    """
    final_iou = 0
    for i in range(NUM_CLASSES):
        if pred_class[i] == 1:
            # Вычисление координат пересечения
            xA = max(iou_box[i,0], new_box[0])
            yA = max(iou_box[i,1], new_box[1])
            xB = min(iou_box[i,2], new_box[2])
            yB = min(iou_box[i,3], new_box[3])

            # Вычисление площадей
            inter_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)
            box_a_area = (iou_box[i,2] - iou_box[i,0] + 1) * (iou_box[i,3] - iou_box[i,1] + 1)
            box_b_area = (new_box[2] - new_box[0] + 1) * (new_box[3] - new_box[1] + 1)

            # Расчет IoU
            iou = inter_area / float(box_a_area + box_b_area - inter_area)
            final_iou = max(final_iou, iou)
    return final_iou

def run(output_file, input_dir):
    """
    Основная функция для выполнения детекции объектов
    Args:
        output_file (str): Путь для сохранения результатов
        input_dir (str): Директория с входными изображениями
    """
    # Конфигурация модели
    MODEL_DIR = 'C:/Users/MAKS/models-master/research/object_detection/mdanadoelo'
    PATH_TO_CKPT = MODEL_DIR + '/frozen_inference_graph.pb'
    NUM_CLASSES = 1
    MAX_IOU = 0.7  # Порог для фильтрации пересечений

    # Инициализация файла результатов
    result_file = open(output_file, 'w+')

    # Загрузка frozen модели TensorFlow
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    # Поиск тестовых изображений
    test_images = glob.glob(os.path.join(input_dir, '*.jpg'))

    # Выполнение детекции
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            # Получение тензоров модели
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            for image_path in test_images:
                # Загрузка изображения
                image = cv2.imread(image_path)
                height, width = image.shape[:2]
                image_name = os.path.basename(image_path)

                # Подготовка входных данных
                image_np_expanded = np.expand_dims(image, axis=0)

                # Выполнение inference
                boxes, scores, classes, _ = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                # Постобработка результатов
                boxes = np.squeeze(boxes)
                scores = np.squeeze(scores)
                classes = np.squeeze(classes).astype(np.int32)

                # Фильтрация и обработка боксов
                pred_class = np.zeros(NUM_CLASSES, dtype=int)
                boxes_for_iou = np.zeros((NUM_CLASSES, 4), dtype=int)
                output_boxes = []
                posStr = ''

                for i in range(len(scores)):
                    if scores[i] > 0.5:  # Порог уверенности
                        class_id = classes[i] - 1
                        if pred_class[class_id] == 0:
                            # Конвертация координат
                            ymin = math.floor(boxes[i,0] * height)
                            xmin = math.floor(boxes[i,1] * width)
                            ymax = math.floor(boxes[i,2] * height)
                            xmax = math.floor(boxes[i,3] * width)
                            new_box = [xmin, ymin, xmax, ymax]

                            # Проверка пересечений
                            iou = iou_inter(new_box, boxes_for_iou, pred_class, NUM_CLASSES)
                            if iou < MAX_IOU:
                                boxes_for_iou[class_id] = new_box
                                pred_class[class_id] = 1

                                # Форматирование результатов
                                csv_line = f"[{xmin},{ymin},{xmax-xmin},{ymax-ymin},{class_id+1}]"
                                posStr += csv_line if i == 0 else f",{csv_line}"

                # Запись в файл
                result_file.write(f"{image_name}:{posStr}\n")

    result_file.close()

# Пример вызова функции
# run('results.txt', 'test_images')