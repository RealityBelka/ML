import cv2
import mediapipe as mp

class FacePositionChecker:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

    def draw_central_rectangle(self, image, rectangle_coords, color=(0, 255, 0), thickness=2):
        """
        Рисует центральный прямоугольник на изображении, который соответствует зоне для лица.
        :param image: Изображение в формате numpy.ndarray (BGR).
        :param rectangle_coords: Координаты прямоугольника.
        :param color: Цвет прямоугольника (по умолчанию зелёный).
        :param thickness: Толщина линий прямоугольника.
        :return: Изображение с нарисованным прямоугольником.
        """
        img_h, img_w, _ = image.shape

        # Убедитесь, что координаты рамки в пределах изображения
        left = max(0, rectangle_coords[0])
        right = min(img_w, rectangle_coords[1])
        top = max(0, rectangle_coords[2])
        bottom = min(img_h, rectangle_coords[3])

        top_left = (left, top)
        bottom_right = (right, bottom)

        cv2.rectangle(image, top_left, bottom_right, color, thickness)

        return image

    def is_face_in_frame(self, img_rgb, rectangle_coords):
        """
        Проверяет, находится ли лицо в выделенной зоне экрана (в прямоугольнике, который представляет овал на клиенте).
        :param img_rgb: Изображение в формате numpy.ndarray (BGR).
        :param rectangle_coords: Массив с координатами ограничивающей рамки [left, right, top, bottom].
        :return: Булево значение: True, если лицо в центре, False в противном случае.
        """
        img_h, img_w, _ = img_rgb.shape

        bounding_rectangle = {
            "left": max(0, rectangle_coords[0]),
            "right": min(img_w, rectangle_coords[1]),
            "top": max(0, rectangle_coords[2]),
            "bottom": min(img_h, rectangle_coords[3])
        }

        # Преобразуем изображение в RGB для работы с Mediapipe
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)

        result = self.face_mesh.process(img_rgb)

        if result.multi_face_landmarks:
            for face_landmarks in result.multi_face_landmarks:
                min_x, min_y = img_w, img_h
                max_x, max_y = 0, 0

                # Находим минимальные и максимальные координаты лица
                for landmark in face_landmarks.landmark:
                    x = int(landmark.x * img_w)
                    y = int(landmark.y * img_h)
                    min_x = min(min_x, x)
                    max_x = max(max_x, x)
                    min_y = min(min_y, y)
                    max_y = max(max_y, y)

                # Проверяем, находится ли лицо в пределах прямоугольника
                if (min_x >= bounding_rectangle["left"] and max_x <= bounding_rectangle["right"] and
                        min_y >= bounding_rectangle["top"] and max_y <= bounding_rectangle["bottom"]):
                    return True

        return False


def main():
    cap = cv2.VideoCapture(0)

    face_position_checker = FacePositionChecker()

    while True:
        success, image = cap.read()  # Захват кадра
        if not success:
            print("Не удалось получить изображение с камеры.")
            break

        # Получаем размеры кадра
        img_h, img_w, _ = image.shape

        # Координаты прямоугольника [left, right, top, bottom] с учётом размеров кадра
        rect = [0, 400, 0, 400]

        is_in_center = face_position_checker.is_face_in_frame(image, rect)

        if is_in_center:
            cv2.putText(image, "Face is in the center", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            face_position_checker.draw_central_rectangle(image, rect, color=(0, 255, 0))
        else:
            cv2.putText(image, "Face is not in the center", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            face_position_checker.draw_central_rectangle(image, rect, color=(0, 0, 255))

        cv2.imshow("Camera Stream", image)

        if cv2.waitKey(1) & 0xFF == 27:  # Нажмите ESC для выхода
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
