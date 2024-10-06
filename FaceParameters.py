import cv2
import mediapipe as mp
import numpy as np
from deepface import DeepFace


class FaceParameters:
    def __init__(self):
        # Инициализация MediaPipe для обнаружения лиц
        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(static_image_mode=False, max_num_faces=1)

    def get_face_count(self, image):
        """Возвращает количество лиц на изображении"""
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(img_rgb)
        if result.multi_face_landmarks:
            return len(result.multi_face_landmarks)
        return 0

    def get_head_angles(self, image):
        """Возвращает углы наклона головы (yaw, pitch, roll)"""
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(img_rgb)
        if result.multi_face_landmarks:
            for face_landmarks in result.multi_face_landmarks:
                landmarks = face_landmarks.landmark
                left_eye_idx = 33
                right_eye_idx = 263
                nose_idx = 1

                left_eye = np.array([landmarks[left_eye_idx].x, landmarks[left_eye_idx].y, landmarks[left_eye_idx].z])
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

    def get_eye_distance(self, image):
        """Возвращает расстояние между центрами глаз"""
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(img_rgb)
        if result.multi_face_landmarks:
            for face_landmarks in result.multi_face_landmarks:
                landmarks = face_landmarks.landmark
                left_eye_idx = 33
                right_eye_idx = 263

                left_eye = np.array([landmarks[left_eye_idx].x, landmarks[left_eye_idx].y])
                right_eye = np.array([landmarks[right_eye_idx].x, landmarks[right_eye_idx].y])

                eye_distance = np.linalg.norm(left_eye - right_eye)
                return eye_distance
        return None

    def get_face_expression(self, image):
        """Возвращает выражение лица (нейтральное, радость, грусть и т.д.) с помощью DeepFace"""
        try:
            result = DeepFace.analyze(image, actions=['emotion'], enforce_detection=False)
            return result['dominant_emotion']
        except:
            return None

    def are_eyes_closed(self, image):
        """Возвращает True, если глаза закрыты, иначе False"""
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(img_rgb)
        if result.multi_face_landmarks:
            for face_landmarks in result.multi_face_landmarks:
                landmarks = face_landmarks.landmark
                left_eye_idx = [33, 159]  # Внутренний и внешний угол левого глаза
                right_eye_idx = [263, 386]  # Внутренний и внешний угол правого глаза

                left_eye_opening = np.linalg.norm(
                    np.array([landmarks[left_eye_idx[0]].y]) - np.array([landmarks[left_eye_idx[1]].y]))
                right_eye_opening = np.linalg.norm(
                    np.array([landmarks[right_eye_idx[0]].y]) - np.array([landmarks[right_eye_idx[1]].y]))

                eye_threshold = 0.04  # Условное значение для определения закрытых глаз
                if left_eye_opening < eye_threshold and right_eye_opening < eye_threshold:
                    return True
        return False
