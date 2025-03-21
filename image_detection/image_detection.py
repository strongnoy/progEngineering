# необходимо написать код, который будет распоознавать объекты на изображении

import torch
from PIL import Image
import requests
from io import BytesIO

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
model.eval()
url = "https://c.pxhere.com/photos/20/52/photo-47141.jpg!d"
response = requests.get(url)
img = Image.open(BytesIO(response.content))

results = model(img)

results.show()
predictions = results.xywh[0]
print(predictions)


labels = results.names
print(labels)

class_ids = predictions[:, -1].tolist()
for class_id in class_ids:
    print(labels[int(class_id)])