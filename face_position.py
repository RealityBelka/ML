import cv2
import mediapipe as mp


class FacePositionChecker:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True)

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

    def is_face_in_center(self, image, margin_ratio=0.2):
        """
        Проверяет, находится ли лицо в центральной зоне экрана (в прямоугольнике, который представляет овал).

        :param image: Изображение в формате numpy.ndarray (BGR).
        :param margin_ratio: Отношение от краев кадра до центрального прямоугольника.
        :return: Булево значение: True, если лицо в центре, False в противном случае.
        """
        # Определяем размеры изображения
        img_h, img_w, _ = image.shape

        # Определяем центральный прямоугольник
        margin_w = int(img_w * margin_ratio)
        margin_h = int(img_h * margin_ratio)
        central_rect = {
            "left": margin_w,
            "right": img_w - margin_w,
            "top": margin_h,
            "bottom": img_h - margin_h
        }

        # Преобразуем изображение для медиапайп
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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
                    if x < min_x: min_x = x
                    if x > max_x: max_x = x
                    if y < min_y: min_y = y
                    if y > max_y: max_y = y

                # Проверяем, находятся ли крайние точки внутри центрального прямоугольника
                if (min_x >= central_rect["left"] and max_x <= central_rect["right"] and
                        min_y >= central_rect["top"] and max_y <= central_rect["bottom"]):
                    return True
                else:
                    return False
        return False


def main():
    cap = cv2.VideoCapture(0)  # Открываем камеру
    cap.set(3, 640)  # Ширина кадра
    cap.set(4, 480)  # Высота кадра

    face_position_checker = FacePositionChecker()

    while True:
        success, image = cap.read()  # Захват кадра
        if not success:
            print("Не удалось получить изображение с камеры.")
            break

        is_in_center = face_position_checker.is_face_in_center(image)  # Проверка положения лица

        # Отображаем результат в окне
        if is_in_center:
            cv2.putText(image, "Face is in the center", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            face_position_checker.draw_central_rectangle(image, color=(0, 255, 0))
        else:
            cv2.putText(image, "Face is not in the center", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            face_position_checker.draw_central_rectangle(image, color=(0, 0, 255))

        cv2.imshow("Camera Stream", image)

        if cv2.waitKey(1) & 0xFF == 27:  # Нажатие ESC для выхода
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
