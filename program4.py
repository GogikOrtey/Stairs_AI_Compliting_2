import tensorflow as tf
import time

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# Загрузка модели
model = load_model('stairs_model_6.h5')



# Вот отсюда будет бесконечный цикл, который раз в секунду берёт изображение и выводит результат

# Записываем время начала выполнения
start_time = time.time()

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


# Записываем время окончания выполнения
end_time = time.time()

# Вычисляем и печатаем общее время выполнения
print(f"Время выполнения: {end_time - start_time:.2f} секунд")

# Вот здесь конец цикла