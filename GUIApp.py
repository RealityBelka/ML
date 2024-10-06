import tkinter as tk


class GUIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Parameters Analysis")

        # Параметры лица
        self.label_face_count = tk.Label(self.root, text="Количество лиц: ")
        self.label_face_count.pack()

        self.label_head_angles = tk.Label(self.root, text="Углы головы: ")
        self.label_head_angles.pack()

        self.label_eye_distance = tk.Label(self.root, text="Расстояние между глазами: ")
        self.label_eye_distance.pack()

        self.label_expression = tk.Label(self.root, text="Выражение лица: ")
        self.label_expression.pack()

        self.label_eyes_closed = tk.Label(self.root, text="Глаза закрыты: ")
        self.label_eyes_closed.pack()

    def update_parameters(self, face_count, head_angles, eye_distance, face_expression, eyes_closed):
        """Обновляет значения параметров в GUI"""
        self.label_face_count.config(text=f"Количество лиц: {face_count}")
        self.label_head_angles.config(text=f"Углы головы: Yaw: {head_angles['yaw']}, Pitch: {head_angles['pitch']}, Roll: {head_angles['roll']}")
        self.label_eye_distance.config(text=f"Расстояние между глазами: {eye_distance}")
        self.label_expression.config(text=f"Выражение лица: {face_expression}")
        self.label_eyes_closed.config(text=f"Глаза закрыты: {'Да' if eyes_closed else 'Нет'}")
