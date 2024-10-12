import cv2
from FaceParams import FaceParams
# from time import time


def image_process(face_params, image):
    flag = False  # True, если проверка пройдена успешно

    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    '''1'''
    faces_count = face_params.get_faces_count(img_rgb)
    if faces_count > 1:
        return flag, "Больше одного лица на изображении"
    if faces_count < 1:
        return flag, "Лицо не обнаружено"

    '''2'''
    # head_size = face_params.get_head_size(img_rgb)  # Выполняет ту же функцию, что eyes_distance
    is_in_frame = face_params.is_face_in_frame(image)
    if is_in_frame:
        face_params.draw_central_rectangle(image, color=(0, 255, 0))
    else:
        face_params.draw_central_rectangle(image, color=(0, 0, 255))
        return flag, "Лицо должно полностью помещаться в рамку"

    '''3'''
    eyes_distance = face_params.get_eye_distance(img_rgb)
    if eyes_distance < 0.1:
        return flag, "Приблизьте телефон к лицу"

    '''4'''
    head_pose = face_params.get_head_pose(img_rgb)
    if not (160 <= head_pose["yaw"] <= 200 and 100 <= head_pose["pitch"] <= 140 and -20 <= head_pose["roll"] <= 20):
        return flag, "Держите голову прямо"

    '''5'''
    # is_obstructed = face_params.check_face_obstruction(img_rgb)

    '''6'''
    illumination = face_params.calculate_face_brightness(img_rgb)

    '''7'''
    # distortion, is_blurred = face_params.calculate_blurriness(image)

    '''8'''
    # not_neutral = face_params.check_neutral_status(img_rgb)

    '''9'''
    # eyes_closed, eyes_rate = face_params.check_eyes_closed(img_rgb)

    '''10'''
    # is_real, antispoof_score = face_params.check_spoofing(img_rgb)




def main():

    SHOW_IMAGE = True

    face_params = FaceParams()

    cap = cv2.VideoCapture(0)

    while True:
        success, image = cap.read()
        if not success:
            break

        h, w, _ = image.shape

        image = cv2.resize(image, (int(w / 3), int(h / 3)))


        if SHOW_IMAGE:
            face_params.draw_face_landmarks(image)
            cv2.imshow("Image", image)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC для выхода
                break

    image.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
