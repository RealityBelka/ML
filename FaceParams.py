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

    def get_faces_count(self, image):
        """Возвращает количество лиц на изображении"""
        # img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(image)
        if result.multi_face_landmarks:
            return len(result.multi_face_landmarks)
        return 0

    def get_head_pose(self, image):
        """Возвращает углы наклона головы (yaw, pitch, roll)"""
        # img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(image)
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

                yaw = np.degrees(np.arctan2(eye_direction[1], eye_direction[0]))
                pitch = np.degrees(np.arctan2(head_direction[2], head_direction[1]))
                roll = np.degrees(np.arctan2(eye_direction[1], eye_direction[2]))

                return {'yaw': yaw, 'pitch': pitch, 'roll': roll}
        return {'yaw': None, 'pitch': None, 'roll': None}

    def get_eye_distance(self, image, distance_in_pixels=True):
        """Возвращает расстояние между центрами глаз.

        Args:
            image: Трёхканальное RGB изображение, представленное в формате numpy ndarray.
            distance_in_pixels: Возвращать расстояние в пикселях. Если установлено false,
            то вернёт относительное расстояние.
        """
        # img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(image)
        if result.multi_face_landmarks:

            left_pupil_idx = 473
            right_pupil_idx = 468

            for face_landmarks in result.multi_face_landmarks:
                landmarks = face_landmarks.landmark

                h, w, _ = image.shape if distance_in_pixels else [1, 1, 1]

                left_pupil = np.array([landmarks[left_pupil_idx].x * w, landmarks[left_pupil_idx].y * h])
                right_pupil = np.array([landmarks[right_pupil_idx].x * w, landmarks[right_pupil_idx].y * h])

                eye_distance = np.linalg.norm(left_pupil - right_pupil)

                return eye_distance
        return None

    def get_head_size(self, image):
        """Возвращает относительный размер головы по вертикали"""
        # img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(image)
        if result.multi_face_landmarks:

            upper_face_idx = 10
            lower_face_idx = 152

            for face_landmarks in result.multi_face_landmarks:
                landmarks = face_landmarks.landmark

                upper_landmark = np.array([landmarks[upper_face_idx].x, landmarks[upper_face_idx].y])
                lower_landmark = np.array([landmarks[lower_face_idx].x, landmarks[lower_face_idx].y])

                head_size_v = np.linalg.norm(lower_landmark - upper_landmark)

                return head_size_v
        return None

    def check_face_obstruction(self, image):
        """Проверяет, полностью ли открыто лицо на изображении"""
        return "[замокано]"

    def check_neutral_status(self, image):
        """Проверяет выражение лица на нейтральность"""
        try:
            result = DeepFace.analyze(image,
                                      actions=['emotion'],
                                      enforce_detection=False,
                                      detector_backend='ssd')
            return result[0]['dominant_emotion']
        except:
            return None

    def check_spoofing(self, image):
        """Проверяет лицо на изображении на подлиность"""
        try:
            result = detection.extract_faces(image, anti_spoofing=True)
            return result[0]['is_real'], result[0]['antispoof_score']
        except:
            return None

    def check_eyes_closed(self, image):
        """Проверяет, закрыты ли глаза"""
        # img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(image)

        eye_idxs = {
            "left": [33, 160, 158, 133, 153, 144],
            "right": [362, 385, 387, 263, 373, 380],
        }

        h, w, _ = image.shape

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
        return None

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

    def calculate_face_brightness(self, image):
        """Функция для вычисления средней освещённости зоны лица на изображении"""
        # img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image)

        if not results.multi_face_landmarks:
            return None
        face_landmarks = results.multi_face_landmarks[0]

        h, w, _ = image.shape

        face_coords = []
        for landmark in face_landmarks.landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            face_coords.append((x, y))

        face_coords = np.array(face_coords, dtype=np.int32)

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, face_coords, 255)

        face_region = cv2.bitwise_and(image, image, mask=mask)

        face_gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)

        brightness = cv2.mean(face_gray, mask=mask)[0]

        return brightness

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




