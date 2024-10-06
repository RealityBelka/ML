import cv2
import tkinter as tk
from deepface import DeepFace
from FaceParameters import FaceParameters
from GUIApp import GUIApp

import os
import warnings
import absl.logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Отключаем логи TensorFlow
warnings.filterwarnings("ignore")  # Отключаем все предупреждения Python
absl.logging.set_verbosity(absl.logging.ERROR)  # Отключаем логи absl

def main():

    face_params = FaceParameters()

    # Создание GUI
    root = tk.Tk()
    gui_app = GUIApp(root)

    # Запуск видео захвата через OpenCV
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cap.set(cv2.CAP_PROP_FPS, 30)

    frame_counter = 0
    while True:
        frame_counter += 1
        if frame_counter % 10 != 0:  # Обрабатываем только каждый 10 кадр
            continue

        success, frame = cap.read()
        if not success:
            break

        # Обработка параметров лица
        face_count = face_params.get_face_count(frame)
        head_angles = face_params.get_head_angles(frame)
        eye_distance = face_params.get_eye_distance(frame)
        face_expression = face_params.get_face_expression(frame)
        eyes_closed = face_params.are_eyes_closed(frame)

        # Обновление GUI параметров
        gui_app.update_parameters(face_count, head_angles, eye_distance, face_expression, eyes_closed)

        # Отображение видео в окне OpenCV
        cv2.imshow('Face Parameters Camera', frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Нажатие ESC для выхода
            break

        # Обновляем Tkinter интерфейс
        root.update_idletasks()
        root.update()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()