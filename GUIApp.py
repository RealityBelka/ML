import tkinter as tk


class GUIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Parameters Analysis")

        # Параметры лица
        self.label_face_count = tk.Label(self.root, text="Количество лиц: ")
        self.label_face_count.pack()

        self.label_head_pose = tk.Label(self.root, text="Углы головы: ")
        self.label_head_pose.pack()

        self.label_eye_distance = tk.Label(self.root, text="Расстояние между глазами: ")
        self.label_eye_distance.pack()

        self.label_head_size = tk.Label(self.root, text="Размер головы: ")
        self.label_head_size.pack()

        self.label_obstruction = tk.Label(self.root, text="Перекрытие лица: ")
        self.label_obstruction.pack()

        self.label_expression = tk.Label(self.root, text="Выражение лица: ")
        self.label_expression.pack()

        self.label_spoofing = tk.Label(self.root, text="Реальное лицо: ")
        self.label_spoofing.pack()

        self.label_eyes_closed = tk.Label(self.root, text="Глаза закрыты: ")
        self.label_eyes_closed.pack()

        self.label_illumination = tk.Label(self.root, text="Освещённость: ")
        self.label_illumination.pack()

        self.label_distorsion = tk.Label(self.root, text="Дисторсия: ")
        self.label_distorsion.pack()

        self.label_background = tk.Label(self.root, text="Задний фон: ")
        self.label_background.pack()

    def update_parameters(self,
                          face_count=None,
                          head_pose=None,
                          eyes_distance=None,
                          head_size=None,
                          is_obstructed=None,
                          is_neutral=None,
                          is_real=None,
                          eyes_closed=None,
                          illumination=None,
                          distortion=None,
                          background=None
                          ):
        """Обновляет значения параметров в GUI"""
        self.label_face_count.config(text=f"Количество лиц: {face_count}")
        self.label_head_pose.config(text=f"Поворот головы: {head_pose}")
        self.label_eye_distance.config(text=f"Расстояние между глазами: {eyes_distance}")
        self.label_head_size.config(text=f"Размер головы: {head_size}")
        self.label_obstruction.config(text=f"Перекрытие лица: {is_obstructed}")
        self.label_expression.config(text=f"Нейтральное выражение лица: {is_neutral}")
        self.label_spoofing.config(text=f"Реальное лицо: {is_real}")
        self.label_eyes_closed.config(text=f"Глаза закрыты: {eyes_closed}")
        self.label_illumination.config(text=f"Освещённость: {illumination}")
        self.label_distorsion.config(text=f"Дисторсия: {distortion}")
        self.label_background.config(text=f"Задний фон: {background}")
