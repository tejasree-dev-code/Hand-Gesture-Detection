import cv2
import mediapipe as mp
import csv
import os
import time

# MediaPipe Hands
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5
)

mp_draw = mp.solutions.drawing_utils

# Gesture names
gestures = ["HELLO", "YES", "NO", "STOP", "PEACE"]

# CSV file
file_name = "gesture_data.csv"

file_exists = os.path.exists(file_name)

with open(file_name, "a", newline="") as file:

    writer = csv.writer(file)

    # Create header only if file is new
    if not file_exists:

        header = []

        for i in range(21):

            header.append("x" + str(i))
            header.append("y" + str(i))
            header.append("z" + str(i))

        header.append("label")

        writer.writerow(header)

    # Open camera
    camera = cv2.VideoCapture(0)

    for gesture in gestures:

        print("Get ready for:", gesture)
        print("Press 's' to start collecting data.")

        while True:

            success, frame = camera.read()

            if not success:
                print("Camera not found")
                break

            cv2.putText(
                frame,
                "Gesture: " + gesture,
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                "Press S to start",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.imshow("Data Collection", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("s"):
                break

            if key == ord("q"):
                camera.release()
                cv2.destroyAllWindows()
                exit()

        count = 0

        while count < 200:

            success, frame = camera.read()

            if not success:
                break

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks:

                hand_landmarks = results.multi_hand_landmarks[0]

                row = []

                for landmark in hand_landmarks.landmark:

                    row.append(landmark.x)
                    row.append(landmark.y)
                    row.append(landmark.z)

                row.append(gesture)

                writer.writerow(row)

                count = count + 1

                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                # Small delay between samples
                time.sleep(0.15)

            cv2.putText(
                frame,
                "Gesture: " + gesture,
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                "Samples: " + str(count) + "/200",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.imshow("Data Collection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):

                camera.release()
                cv2.destroyAllWindows()
                exit()

    camera.release()
    cv2.destroyAllWindows()

print("Data collection completed!")