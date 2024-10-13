# ML_Face

---

Проверка изображения с лицом осуществляется по следующим признакам:

1) faces_count — количество обнаруженных лиц; 
2) is_face_in_frame — положение лица относительно рамки в кадре (лицо должно помещаться в центр экрана); 
3) eyes_distance — проверка расстояния между глазами (лицо слишком близко или далеко от камеры); 
4) mono_background — анализ однородности фона за лицом; 
5) head_pose — углы наклона головы: yaw (поворот), pitch (наклон), roll (наклон вбок); 
6) face_obstruction — наличие препятствий на лице (очки, маски, другие объекты); 
7) brightness, CV — проверка освещённости лица (brightness) и равномерности освещения (CV); 
8) blurriness — анализ резкости изображения; 
9) eyes_closed — проверка, открыты глаза или закрыты;
10) neutral_expression — проверка нейтральности выражения лица;
11) spoofing_check — проверка на наличие реального человека в кадре (антиспуфинг).


## Установка и запуск

1) Установите зависимости:
```pip install -r requirements.txt```
2) Убедитесь, что на устройстве установлена библиотека OpenCV для работы с камерой:
```pip install opencv-python```
3) Модуль готов к использованию:
   - принимает изображение и проверяет его по вышеуказанным признакам; 
   - возвращает состояние проверки изображения **status** и сообщение об ошибке **message**. 
   - Пример использования:
   ```python
    from FaceParams import FaceParams

    face_params = FaceParams()
    image = cv2.imread("example.jpg")
    rectangle_points = [0, 400, 0, 400]

    status, message = image_process(face_params, image, rectangle_points)
    print(f"ok: {status}, message: {message}")
   ```


## Результаты

 - **status**: возвращает True или False в зависимости от того, прошло ли изображение проверку. 
 - **message**: возвращает строку с описанием ошибки, если изображение не прошло проверку.


## Используемые библиотеки:

Все используемые пакеты имеют лицензии, позволяющие их использование в коммерческих продуктах с указанием авторства. Среди них:

 - OpenCV: https://opencv.org/ (Apache 2.0)
 - Mediapipe: https://mediapipe.dev/ (Apache 2.0)
 - Numpy: https://numpy.org/ (BSD)
 - TensorFlow: https://www.tensorflow.org/ (Apache 2.0)
 - DeepFace: https://github.com/serengil/deepface (MIT)
