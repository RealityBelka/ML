import cv2
import numpy as np
import json
from FaceParams import FaceParams


def bytes_to_ndarray(image_bytes):
    # Преобразование байтов в numpy массив (массив байт)
    nparr = np.frombuffer(image_bytes, np.uint8)

    # Декодирование изображения из numpy массива в формате OpenCV
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    return image


def main():
    SHOW_IMAGE = True

    face_params = FaceParams()

    image = cv2.imread("images/image.png") # Либо bytes_to_ndarray

    assert image is not None, "Изображение не загружено. Проверьте путь к файлу."

    h, w, _ = image.shape

    image = cv2.resize(image, (int(w / 2), int(h / 2)))

    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Вызов функций определения параметров
    faces_count = face_params.get_faces_count(img_rgb)

    result = {
        "faces_count": faces_count
    }

    if faces_count == 1:
        head_pose = face_params.get_head_pose(img_rgb)
        eyes_distance = face_params.get_eye_distance(img_rgb, True)
        head_size = face_params.get_head_size(img_rgb)
        is_obstructed = face_params.check_face_obstruction(img_rgb)
        emotion_status = face_params.check_neutral_status(img_rgb)
        is_real, antispoof_score = face_params.check_spoofing(img_rgb)
        eyes_closed, eyes_rate = face_params.check_eyes_closed(img_rgb)
        illumination = face_params.calculate_face_brightness(img_rgb)
        distortion, is_blurred = face_params.calculate_blurriness(image)

        result.update({
            "head_pose": head_pose,
            "eyes_distance": eyes_distance,
            "head_size": head_size,
            "is_obstructed": is_obstructed,
            "emotion_status": emotion_status,
            "is_real": is_real,
            "antispoof_score": antispoof_score,
            "eyes_closed": eyes_closed,
            "eyes_rate": eyes_rate,
            "illumination": illumination,
            "distortion": distortion,
            "is_blurred": is_blurred
        })

    else:
        result.update({
            "head_pose": None,
            "eyes_distance": None,
            "head_size": None,
            "is_obstructed": None,
            "emotion_status": None,
            "is_real": None,
            "antispoof_score": None,
            "eyes_closed": None,
            "eyes_rate": None,
            "illumination": None,
            "distortion": None,
            "is_blurred": None
        })

    # Вывод данных в формате JSON
    result_json = json.dumps(result, indent=4)
    print(result_json)

    # Запись данных в файл JSON
    with open("output.json", "w") as outfile:
        json.dump(result, outfile, indent=4)

    if SHOW_IMAGE:
        face_params.draw_face_landmarks(image)
        cv2.imshow("Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
