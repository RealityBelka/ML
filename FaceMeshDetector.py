import cv2
import mediapipe as mp
import time


class FaceMeshDetector:
    def __init__(self, max_faces=1, thickness=1, circle_radius=2):
        # Инициализация распознавания лиц и сетки лица
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(max_num_faces=max_faces)
        self.mpDraw = mp.solutions.drawing_utils
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=thickness, circle_radius=circle_radius, color=(0, 255, 0))
        self.pTime = 0

    def detect_face_mesh(self, img):
        # Преобразование изображения в RGB для обработки Mediapipe
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.faceMesh.process(imgRGB)
        return results

    def draw_landmarks(self, img, results):
        # Отрисовка сетки лица и вывод координат ключевых точек
        if results.multi_face_landmarks:
            for faceLms in results.multi_face_landmarks:
                self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS, self.drawSpec,
                                           self.drawSpec)
                for id, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    if id == 1 or id == 33 or id == 263 or id == 168:  # Несколько основных точек
                        cv2.circle(img, (x, y), 5, (255, 0, 0), cv2.FILLED)
                        cv2.putText(img, f'{id}: ({x},{y})', (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)

    def display_fps(self, img):
        # Отображение FPS на изображении
        cTime = time.time()
        fps = 1 / (cTime - self.pTime)
        self.pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    def start_video_capture(self):
        # Подключение камеры
        cap = cv2.VideoCapture(0)
        cap.set(3, 640)  # Width
        cap.set(4, 480)  # Height
        cap.set(10, 100)  # Brightness

        while True:
            success, img = cap.read()
            img = cv2.flip(img, 1)  # Зеркальное отображение

            # Обработка изображения и отрисовка сетки
            results = self.detect_face_mesh(img)
            self.draw_landmarks(img, results)

            # Отображение FPS
            self.display_fps(img)

            # Показ результата
            cv2.imshow('Face Mesh', img)

            # Выход по нажатию ESC
            if cv2.waitKey(20) & 0xFF == 27:
                break

        # Освобождение ресурсов
        cap.release()
        cv2.destroyAllWindows()
