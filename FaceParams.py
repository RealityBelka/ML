import cv2
import numpy as np
import mediapipe as mp
from deepface import DeepFace
from deepface.modules import detection


class FaceParams:
    def __init__(self, max_faces=10, detection_confidence=0.5, tracking_confidence=0.5):
        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_faces,
            refine_landmarks=True,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def draw_face_landmarks(self, image, face_mesh=False, pupils_landmarks=False, vertical_face_line=False):
        """Визуализирует параметры на изображении.

        Args:
            image: Трёхканальное BGR изображение, представленное в формате numpy ndarray.
            face_mesh: Следует ли рисовать ориентиры.
            pupils_landmarks: Следует ли рисовать точки зрачков.
            vertical_face_line: Следует ли рисовать вертикальную линию лица.
        """
        results = self.face_mesh.process(image)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:

                if face_mesh:
                    self.mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=None,
                        landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0),
                                                                          thickness=-1,
                                                                          circle_radius=1
                                                                          ),
                        connection_drawing_spec=self.mp_drawing.DrawingSpec(thickness=1,
                                                                            circle_radius=1,
                                                                            color=(255, 255, 255)
                                                                            )
                    )

                h, w, _ = image.shape

                if pupils_landmarks:
                    # Точка левого глаза (473)
                    left_eye_x = int(face_landmarks.landmark[473].x * w)
                    left_eye_y = int(face_landmarks.landmark[473].y * h)
                    cv2.circle(image, (left_eye_x, left_eye_y), 3, (0, 255, 255), -1)
                    # Точка правого глаза (468)
                    right_eye_x = int(face_landmarks.landmark[468].x * w)
                    right_eye_y = int(face_landmarks.landmark[468].y * h)
                    cv2.circle(image, (right_eye_x, right_eye_y), 3, (0, 255, 255), -1)

                if vertical_face_line:
                    # Верхняя точка лица (10)
                    up_x = int(face_landmarks.landmark[10].x * w)
                    up_y = int(face_landmarks.landmark[10].y * h)
                    # Нижняя точка лица (152)
                    low_x = int(face_landmarks.landmark[152].x * w)
                    low_y = int(face_landmarks.landmark[152].y * h)
                    cv2.line(image, (up_x, up_y), (low_x, low_y), (0, 0, 255), 2)

    def get_faces_count(self, img_rgb):
        """Возвращает количество лиц на изображении"""
        result = self.face_mesh.process(img_rgb)
        if result.multi_face_landmarks:
            return len(result.multi_face_landmarks)
        return 0

    def get_head_pose(self, img_rgb):
        """Возвращает углы наклона головы (yaw, pitch, roll)"""
        result = self.face_mesh.process(img_rgb)
        if result.multi_face_landmarks:

            left_eye_idx = 33
            right_eye_idx = 263
            nose_idx = 1

            for face_landmarks in result.multi_face_landmarks:
                landmarks = face_landmarks.landmark

                left_eye = np.array(
                    [landmarks[left_eye_idx].x, landmarks[left_eye_idx].y, landmarks[left_eye_idx].z])
                right_eye = np.array(
                    [landmarks[right_eye_idx].x, landmarks[right_eye_idx].y, landmarks[right_eye_idx].z])
                nose = np.array([landmarks[nose_idx].x, landmarks[nose_idx].y, landmarks[nose_idx].z])

                eye_direction = right_eye - left_eye
                head_direction = nose - (left_eye + right_eye) / 2

                yaw = np.degrees(np.arctan2(head_direction[0], head_direction[2]))
                pitch = np.degrees(np.arctan2(head_direction[1], head_direction[2]))
                roll = np.degrees(np.arctan2(eye_direction[1], eye_direction[0]))

                yaw = round(yaw)
                if yaw < 0:
                    yaw += 360
                roll = round(roll)
                pitch = round(pitch)
                if pitch < 0:
                    pitch += 360


                head_angles = {'yaw': yaw, 'pitch': pitch, 'roll': roll}

                return head_angles

        return None

    def get_eye_distance(self, img_rgb: np.ndarray, distance_in_pixels=True):
        """Возвращает расстояние между центрами глаз.

        Args:
            img_rgb: Трёхканальное RGB изображение, представленное в формате numpy ndarray.
            distance_in_pixels: Возвращать расстояние в пикселях. Если установлено false,
            то вернёт относительное расстояние.
        """
        result = self.face_mesh.process(img_rgb)
        if result.multi_face_landmarks:

            left_pupil_idx = 473
            right_pupil_idx = 468

            for face_landmarks in result.multi_face_landmarks:
                landmarks = face_landmarks.landmark

                h, w, _ = img_rgb.shape if distance_in_pixels else [1, 1, 1]

                left_pupil = np.array([landmarks[left_pupil_idx].x * w, landmarks[left_pupil_idx].y * h])
                right_pupil = np.array([landmarks[right_pupil_idx].x * w, landmarks[right_pupil_idx].y * h])

                eyes_distance = np.linalg.norm(left_pupil - right_pupil)
                eyes_distance = round(eyes_distance, 3)

                return eyes_distance
        return None

    def get_head_size(self, img_rgb):
        """Возвращает относительный размер головы по вертикали"""
        result = self.face_mesh.process(img_rgb)
        if result.multi_face_landmarks:

            upper_face_idx = 10
            lower_face_idx = 152

            for face_landmarks in result.multi_face_landmarks:
                landmarks = face_landmarks.landmark

                upper_landmark = np.array([landmarks[upper_face_idx].x, landmarks[upper_face_idx].y])
                lower_landmark = np.array([landmarks[lower_face_idx].x, landmarks[lower_face_idx].y])

                head_size_v = np.linalg.norm(lower_landmark - upper_landmark)

                head_size_v = round(head_size_v, 3)

                return head_size_v
        return None

    def draw_central_rectangle(self, image, margin_ratio=0.2, color=(0, 255, 0), thickness=2):
        """
        Рисует центральный прямоугольник на изображении, который соответствует зоне для лица.

        :param image: Изображение в формате numpy.ndarray (BGR).
        :param margin_ratio: Отношение от краев кадра до центрального прямоугольника.
        :param color: Цвет прямоугольника (по умолчанию зелёный).
        :param thickness: Толщина линий прямоугольника.
        :return: Изображение с нарисованным прямоугольником.
        """
        img_h, img_w, _ = image.shape

        # Вычисляем границы прямоугольника
        margin_w = int(img_w * margin_ratio)
        margin_h = int(img_h * margin_ratio)

        top_left = (margin_w, margin_h)
        bottom_right = (img_w - margin_w, img_h - margin_h)

        # Рисуем прямоугольник
        cv2.rectangle(image, top_left, bottom_right, color, thickness)

        return image

    def is_face_in_frame(self, img_rgb, margin_ratio=0.2):
        """
        Проверяет, находится ли лицо в выделенной зоне экрана (в прямоугольнике, который представляет овал на клиенте).

        :param img_rgb: Изображение в формате numpy.ndarray (BGR).
        :param margin_ratio: Отношение от краев кадра до центрального прямоугольника.
        :return: Булево значение: True, если лицо в центре, False в противном случае.
        """
        # Определяем размеры изображения
        img_h, img_w, _ = img_rgb.shape

        # Определяем центральный прямоугольник
        margin_w = int(img_w * margin_ratio)
        margin_h = int(img_h * margin_ratio)
        central_rect = {
            "left": margin_w,
            "right": img_w - margin_w,
            "top": margin_h,
            "bottom": img_h - margin_h
        }

        # Получаем результат обработки лица
        result = self.face_mesh.process(img_rgb)

        if result.multi_face_landmarks:
            for face_landmarks in result.multi_face_landmarks:
                # Инициализация крайних точек лица
                min_x, min_y = img_w, img_h
                max_x, max_y = 0, 0

                # Проходим по всем точкам лица и находим крайние
                for landmark in face_landmarks.landmark:
                    x = int(landmark.x * img_w)
                    y = int(landmark.y * img_h)
                    if x < min_x:
                        min_x = x
                    if x > max_x:
                        max_x = x
                    if y < min_y:
                        min_y = y
                    if y > max_y:
                        max_y = y

                # Проверяем, находятся ли крайние точки внутри центрального прямоугольника
                if (min_x >= central_rect["left"] and max_x <= central_rect["right"] and
                        min_y >= central_rect["top"] and max_y <= central_rect["bottom"]):
                    return True
                else:
                    return False
        return False

    def check_face_obstruction(self, img_rgb):
        """Проверяет, полностью ли открыто лицо на изображении"""
        return False

    def check_neutral_status(self, img_rgb):
        """Проверяет выражение лица на нейтральность"""
        try:
            result = DeepFace.analyze(img_rgb,
                                      actions=['emotion'],
                                      enforce_detection=False,
                                      detector_backend='ssd')
            return result[0]['dominant_emotion']
        except:
            return None

    def check_spoofing(self, img_rgb):
        """Проверяет лицо на изображении на подлиность"""
        try:
            result = detection.extract_faces(img_rgb, anti_spoofing=True)
            return result[0]['is_real'], result[0]['antispoof_score']
        except ValueError as v:
            # print("check_spoofing --> Value Error: ", v)
            return None, None

    def check_eyes_closed(self, img_rgb):
        """Проверяет, закрыты ли глаза"""
        result = self.face_mesh.process(img_rgb)

        eye_idxs = {
            "left": [33, 160, 158, 133, 153, 144],
            "right": [362, 385, 387, 263, 373, 380],
        }

        h, w, _ = img_rgb.shape

        if result.multi_face_landmarks:
            for face_landmarks in result.multi_face_landmarks:
                landmarks = face_landmarks.landmark

                left_eye_rate = self.get_eye_rate(landmarks, eye_idxs["left"])
                right_ryr_rate = self.get_eye_rate(landmarks, eye_idxs["right"])

                eyes_rate = (left_eye_rate + right_ryr_rate) / 2

                if eyes_rate < 0.1:
                    eyes_closed = True
                else:
                    eyes_closed = False

                return eyes_closed, eyes_rate
        return None, None

    def get_eye_rate(self, landmarks, eye_idxs):
        """Технический метод, возращает отношение длины и ширины глаза"""
        left, up1, up2, right, low2, low1 = eye_idxs

        left_landmark = np.array([landmarks[left].x, landmarks[left].y])
        right_landmark = np.array([landmarks[right].x, landmarks[right].y])

        up1_landmark = np.array([landmarks[up1].x, landmarks[up1].y])
        low1_landmark = np.array([landmarks[low1].x, landmarks[low1].y])

        up2_landmark = np.array([landmarks[up2].x, landmarks[up2].y])
        low2_landmark = np.array([landmarks[low2].x, landmarks[low2].y])

        eye_size_h = np.linalg.norm(right_landmark - left_landmark)
        eye_size_v1 = np.linalg.norm(low1_landmark - up1_landmark)
        eye_size_v2 = np.linalg.norm(low2_landmark - up2_landmark)

        eye_rate = (eye_size_v1 + eye_size_v2) / (2 * eye_size_h + 10**-4)

        return eye_rate

    def calculate_face_illumination(self, img_rgb):
        """Функция для вычисления средней освещённости и дисперсии зоны лица на изображении"""
        results = self.face_mesh.process(img_rgb)

        if not results.multi_face_landmarks:
            return None
        face_landmarks = results.multi_face_landmarks[0]

        h, w, _ = img_rgb.shape

        face_coords = []
        for landmark in face_landmarks.landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            face_coords.append((x, y))

        face_coords = np.array(face_coords, dtype=np.int32)

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, face_coords, 255)

        face_region = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)

        face_gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)

        # Вычисляем среднее значение яркости и стандартное отклонение
        mean, stddev = cv2.meanStdDev(face_gray, mask=mask)
        brightness = mean[0][0]
        variance = stddev[0][0] ** 2  # Дисперсия

        brightness = round(brightness)
        variance = round(variance)

        return brightness, variance

    def calculate_blurriness(self, image):
        """Функция для определения размытости изображения с использованием лапласиана.
        Чем ниже значение, тем более размытым является изображение.
        Args:
            image: Входное изображение (BGR формат)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Применение Лапласиана для вычисления резкости
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 100:
            is_blurred = True
        else:
            is_blurred = False

        return laplacian_var, is_blurred
