# необходимо написать код, который будет распоознавать объекты на изображении

import torch
from PIL import Image
import requests
from io import BytesIO

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
model.eval()
url = "https://c.pxhere.com/photos/20/52/photo-47141.jpg!d"  # Вставьте ссылку на ваше изображение
response = requests.get(url)
img = Image.open(BytesIO(response.content))

# Применяем модель для обнаружения объектов на изображении
results = model(img)

# Показываем результат
results.show()  # Показывает изображение с отмеченными объектами
# Получаем детекции
predictions = results.xywh[0]  # Координаты ограничивающих рамок
print(predictions)

# Получаем имена классов объектов
labels = results.names
print(labels)

# Получаем метки классов для каждого объекта
class_ids = predictions[:, -1].tolist()
for class_id in class_ids:
    print(labels[int(class_id)])