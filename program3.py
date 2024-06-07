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


# # import h5py

# # with h5py.File('stairs_model_5.h5', 'r+') as f:
# #   # Find the DepthwiseConv2D layer configuration (might need adjustments)
# #   layer_config = f['model_weights']['denseblock1_conv']
# #   # Remove the 'groups' key if it exists
# #   if 'groups' in layer_config.attrs:
# #     del layer_config.attrs['groups']
# #   # Save the modified model
# #   f.save()

# print("123")

# # Версия питона
# import sys
# print(sys.version)

# # Версия keras
# import keras
# print(keras.__version__)

# # Версия tensorflow
# import tensorflow as tf
# print(tf.__version__)
