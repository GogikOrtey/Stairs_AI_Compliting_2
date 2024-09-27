**Готовая программа находится в папке "Релиз"**

* input_AI_image.png - Входное изображение, для скрипта
* program6.py - Программа. Она долго запускается (1-7 минут), и после запуска каждую секунду прогоняет входное изображение через модель. Результат (вероятность наличия лестницы на изображении, от 0 до 1) - записывает в выходной файл. При любых исключениях и ошибках (приемущественно файловой системы) - игнорирует ошибки, и продолжает свою работу.
* result.txt - Сюда выводится вероятность наличия лестницы на изображении
* stairs_model_6.h5 - Обученная модель. Обучалась здесь: https://colab.research.google.com/drive/1bwFZTtIug13DxH7ZUGL2AgEGqlG1ywOW?usp=sharing Для обучения использовалась предобученная модель MobileNetV2, что позволило не эксперементировать с гиперпараметрами а изначально выбрать самые подходящие для этой задачи - распознавания изображений, а также сэкономить время и ресурсы на обучении модели. Она показывает высокие результаты (> 97% accuracy). 

  Изображения использовавшиеся для обучения также лежат в этом репозитории.

Данная программа автоматически запускается и работает параллельно основной дипломной программе на Unity.

Для обучения и использования модели были взяты библиотеки keras и tensorflow

Итоговый билд программы на питоне получился объёмным (около 400МБ), тогда как файл весов нейросети весит всего 24МБ. Я думаю, что воспользовался немного не теми библиотеками, именно для использования модели. Но время поджимало, и я оставил эти библиотеки, не смотря на размер.
