from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import sys

# Установка кодировки консоли на UTF-8
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Загрузка модели
model = load_model('stairs_model_5.h5')

# Загрузка и изменение размера изображения
img = image.load_img('input_AI_image.png', target_size=(200, 300))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

# Нормализация данных
img_array /= 255.0

# Получение предсказания от нейросети
prediction = model.predict(img_array)

# Вывод результата
print(f'Model prediction: {prediction[0][0]}')
