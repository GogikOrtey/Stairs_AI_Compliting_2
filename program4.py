import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# Загрузка модели
model = load_model('stairs_model_6.h5')

# Загрузка изображения
image = load_img('input_AI_image.png', target_size=(200, 300))

# Преобразование изображения в массив
image = img_to_array(image)

# Нормализация изображения
image = image / 255.0

# Расширение размерности
image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])

# Прогнозирование результата
result = model.predict(image)[0][0]

# Вывод результата
print(f"Вероятность наличия объекта: {result:.2f}")
