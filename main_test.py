import cv2
import tkinter as tk
from FaceParams import FaceParams
from GUIApp import GUIApp


def main():

    SHOW_IMAGE = True

    face_params = FaceParams()

    root = tk.Tk()
    gui_app = GUIApp(root)

    cap = cv2.VideoCapture(0)

    while True:
        success, image = cap.read()
        if not success:
            break

        h, w, _ = image.shape

        image = cv2.resize(image, (int(w / 3), int(h / 3)))

        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Вызов функций определения параметров
        faces_count = face_params.get_faces_count(img_rgb)

        if faces_count == 1:
            head_pose = face_params.get_head_pose(img_rgb)
            eyes_distance = face_params.get_eye_distance(img_rgb, False)
            is_obstructed = face_params.check_face_obstruction(img_rgb)
            is_neutral = face_params.check_neutral_status(img_rgb)
            is_real = face_params.check_spoofing(img_rgb)
            eyes_closed = face_params.check_eyes_closed(img_rgb)
            illumination = face_params.calculate_face_illumination(img_rgb)
            distortion = face_params.calculate_blurriness(image)
            background = face_params.calculate_background_uniformity(image)

            # Обновление GUI параметров
            gui_app.update_parameters(faces_count,
                                      head_pose=head_pose,
                                      eyes_distance=eyes_distance,
                                      # head_size=head_size,
                                      is_obstructed=is_obstructed,
                                      is_neutral=is_neutral,
                                      is_real=is_real,
                                      eyes_closed=eyes_closed,
                                      illumination=illumination,
                                      distortion=distortion,
                                      background=background
                                      )

            root.update_idletasks()
            root.update()

        if SHOW_IMAGE:
            face_params.draw_face_landmarks(image)
            cv2.imshow("Image", image)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    image.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
