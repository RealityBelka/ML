import cv2
import numpy as np
from FaceParams import FaceParams


def image_process(face_params, image, rectangle_coords):
    flag = False  # True, если проверка пройдена успешно

    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    '''1'''
    faces_count = face_params.get_faces_count(img_rgb)
    if faces_count > 1:
        return flag, "Больше одного лица на изображении"
    if faces_count < 1:
        return flag, "Лицо не обнаружено"

    '''2'''
    is_in_frame = face_params.is_face_in_frame(image, rectangle_coords)
    if not is_in_frame:
        return flag, "Лицо должно полностью помещаться в рамку"

    '''3'''
    eyes_distance = face_params.get_eye_distance(img_rgb)
    if eyes_distance < 0.1:
        return flag, "Приблизьте телефон к лицу"

    '''3.5'''
    mono_background = face_params.calculate_background_uniformity(image)
    if mono_background > 22:
        return {"ok": flag, "message": "Слишкои пёстрый задний фон"}

    '''4'''
    head_pose = face_params.get_head_pose(img_rgb)
    if not ((140 <= head_pose["yaw"] <= 220) and (80 <= head_pose["pitch"] <= 180) and (-40 <= head_pose["roll"] <= 40)):
        return flag, "Держите голову прямо"

    '''5'''
    is_obstructed = face_params.check_face_obstruction(img_rgb)
    if is_obstructed:
        return flag, "Лицо должно быть полностью открыто"

    '''6'''
    brightness, CV = face_params.calculate_face_illumination(img_rgb)
    if brightness < 50 or 170 < brightness:
        return flag, "Обеспечьте равномерное освещение лица (brightness)"
    if CV > 15:
        return flag, "Обеспечьте равномерное освещение лица (CV)"

    '''7'''
    _, is_blurred = face_params.calculate_blurriness(image)
    if is_blurred:
        return flag, "Отодвиньте телефон от лица для фокусировки"

    '''8'''
    is_neutral = face_params.check_neutral_status(img_rgb)
    if not is_neutral:
        return flag, "Выражение лица должно быть нейтральным"

    '''9'''
    eyes_closed = face_params.check_eyes_closed(img_rgb)
    if eyes_closed:
        return flag, "Откройте глаза"

    '''10'''
    is_real = face_params.check_spoofing(img_rgb)
    if not is_real:
        return flag, "Кажется, в кадре не реальный человек"

    flag = True

    return flag, None


def bytes_to_ndarray(image_bytes):
    """Преобразует изображение из формата bytes в np.ndarray для работы с OpenCV."""
    np_arr = np.frombuffer(image_bytes, np.uint8)

    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    assert image is not None, "Ошибка при декодировании изображения. Проверьте корректность данных."

    return image


def main():
    SHOW_IMAGE = False

    face_params = FaceParams()

    # Декодирование из bytes
    # with open("path_to_image.jpg", "rb") as f:
    #     image_bytes = f.read()
    # image = bytes_to_ndarray(image_bytes)

    # Пример входных данных
    image = cv2.imread("images/good/MS_Cornelius_Imani_CloseUp.jpg")
    rectangle_points = [0, 400, 0, 400]

    h, w, _ = image.shape

    image = cv2.resize(image, (int(w / 3), int(h / 3)))

    ok, message = image_process(face_params, image, rectangle_points)

    if SHOW_IMAGE:
        face_params.draw_face_landmarks(image)
        cv2.imshow("Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return {"ok": ok, "message": message}


if __name__ == "__main__":
    print(main())
