print("Нейросеть загружается, ждите")
print("")

import tensorflow as tf
import time

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# Загрузка модели
model = load_model('stairs_model_6.h5')
print("")
print("Можете свернуть это окно, пусай оно работает в фоновом режиме")
print("")

while True:
    try:
        # # Записываем время начала выполнения
        # start_time = time.time()

        # Загрузка изображения, и изменение его размера
        image = load_img('input_AI_image.png', target_size=(200, 300))

        # Преобразование изображения в массив
        image = img_to_array(image)

        # Нормализация изображения
        image = image / 255.0

        # Расширение размерности
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])

        # Прогнозирование результата
        result = model.predict(image)[0][0]

        # Округление до 5 знаков после запятой
        result = round(result, 5)

        # Вывод результата
        print("Вероятность наличия объекта:" + str(result))

        # Открытие файла в режиме записи
        with open('result.txt', 'w') as file:
            # Запись числа в файл
            file.write(str(result))


        # # Записываем время окончания выполнения
        # end_time = time.time()

        # # Вычисляем и печатаем общее время выполнения
        # print(f"Время выполнения: {end_time - start_time:.2f} секунд")

        # В среднем, предсказание нейросети занимает 0.15 секунд


        # Пауза на одну секунду
        time.sleep(1)
        # print("Прошла секунда")
    
    # Мы ловим исключения, которые могу произойти, в частности при
    # загрузке изображения, и сохранении результатов в текстовый файл
    except:
        pass # И ничего не делаем. Программа работает дальше
    