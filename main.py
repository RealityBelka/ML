import cv2
import tkinter as tk
# from FaceParameters import FaceParameters
from GUIApp import GUIApp
from FaceParams import FaceParams


def main():
    SHOW_IMAGE = False

    face_params = FaceParams()

    image = cv2.imread("images/image.png")
    assert image is not None, "Изображение не загружено. Проверьте путь к файлу."

    image = cv2.resize(image, (479, 679))

    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Вызов функций определения параметров
    faces_count = face_params.get_faces_count(img_rgb)

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

        print("faces_count: ", faces_count)
        print("head_pose: ", head_pose)
        print("eyes_distance: ", eyes_distance)
        print("head_size: ", head_size)
        print("is_obstructed: ", is_obstructed)
        print("emotion_status: ", emotion_status)
        print("is_real: ", is_real, "; antispoof_score: ", antispoof_score)
        print("eyes_closed: ", eyes_closed, "; eyes_rate: ", eyes_rate)
        print("illumination: ", illumination)
        print("distortion: ", distortion, "; is_blurred: ", is_blurred)

    else:
        print("faces_count: ", faces_count)
        print("head_pose: ", None)
        print("eyes_distance: ", None)
        print("head_size: ", None)
        print("is_obstructed: ", None)
        print("emotion_status: ", None)
        print("is_real: ", None, "; antispoof_score: ", None)
        print("eyes_closed: ", None, "; eyes_rate: ", None)
        print("illumination: ", None)
        print("distortion: ", None, "; is_blurred: ", None)

    if SHOW_IMAGE:
        face_params.draw_face_landmarks(image)
        cv2.imshow("Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
