"""
Скрипт для создания масок в формате png из xml
"""
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw

# Парсинг XML
tree = ET.parse("annotations.xml")  # Убедитесь, что файл аннотаций называется 'annotations.xml'
root = tree.getroot()

# Извлечение меток и их цветов
labels = {}
for label in root.findall('./meta/task/labels/label'):
    name = label.find('name').text.strip()
    color_hex = label.find('color').text.strip()
    # Конвертация HEX в RGB
    color_rgb = tuple(int(color_hex[i + 1:i + 3], 16) for i in (0, 2, 4))
    labels[name] = color_rgb

# Поиск изображения с id=4
image_elem = None
for image in root.findall('image'):
    if image.get('id') == '0':
        image_elem = image
        break

if image_elem is None:
    raise ValueError("Не найдено")

width = int(image_elem.get('width'))
height = int(image_elem.get('height'))

# Создание пустой маски
mask = Image.new('RGB', (width, height), (0, 0, 0))
draw = ImageDraw.Draw(mask)

# Сбор полигонов
polygons = []
for polygon in image_elem.findall('polygon'):
    label = polygon.get('label')
    points_str = polygon.get('points')
    z_order = int(polygon.get('z_order', 0))
    # Парсинг точек
    points = []
    for pair in points_str.split(';'):
        x, y = map(float, pair.strip().split(','))
        points.append((x, y))
    polygons.append({
        'label': label,
        'points': points,
        'z_order': z_order
    })

# Сортировка полигонов по z_order и порядку в XML
polygons_sorted = sorted(polygons, key=lambda x: (x['z_order'], polygons.index(x)))

# Отрисовка полигонов на маске

for poly in polygons_sorted:
    label = poly['label']
    points = poly['points']
    color = labels.get(label, (0, 0, 0))  # Черный цвет, если метка не найдена
    # Конвертация координат в целые числа
    int_points = [(int(round(x)), int(round(y))) for (x, y) in points]
    draw.polygon(int_points, fill=color)

# Сохранение маски
mask.save('mask2.png')
print("Маска успешно сохранена как mask1.png")