import cv2
import mediapipe as mp
import joblib
import pyttsx3
import threading
import time


# Load trained ML model
model = joblib.load("gesture_model.pkl")


# Function for speech
def speak(text):

    global speaking

    engine = pyttsx3.init()

    engine.say(text)
    engine.runAndWait()

    speaking = False


# Speech status
speaking = False

# Last gesture that was spoken
last_spoken_gesture = ""

# Time of last speech
last_speech_time = 0

# Minimum time between speeches
cooldown = 1.5


# MediaPipe Hands
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5
)

mp_draw = mp.solutions.drawing_utils


# Open camera
camera = cv2.VideoCapture(0)


while True:

    success, frame = camera.read()

    if not success:
        print("Camera not found")
        break


    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # Detect hand
    results = hands.process(rgb_frame)


    if results.multi_hand_landmarks:

        hand_landmarks = results.multi_hand_landmarks[0]


        # Create features
        features = []


        # Wrist coordinates
        wrist_x = hand_landmarks.landmark[0].x
        wrist_y = hand_landmarks.landmark[0].y
        wrist_z = hand_landmarks.landmark[0].z


        # Normalize landmarks
        for landmark in hand_landmarks.landmark:

            features.append(
                landmark.x - wrist_x
            )

            features.append(
                landmark.y - wrist_y
            )

            features.append(
                landmark.z - wrist_z
            )


        # Predict gesture
        prediction = model.predict([features])

        gesture = prediction[0]


        # Confidence
        probabilities = model.predict_proba([features])[0]

        confidence = max(probabilities) * 100


        # Display gesture
        cv2.putText(
            frame,
            "Gesture: " + gesture,
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )


        # Display confidence
        cv2.putText(
            frame,
            "Confidence: " + str(round(confidence, 2)) + "%",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


        # Current time
        current_time = time.time()


        # Speak new gesture
        if (
            gesture != last_spoken_gesture
            and not speaking
            and current_time - last_speech_time > cooldown
        ):

            speaking = True

            last_spoken_gesture = gesture

            last_speech_time = current_time

            speech_thread = threading.Thread(
                target=speak,
                args=(gesture,)
            )

            speech_thread.daemon = True

            speech_thread.start()


        # Draw landmarks
        mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )


    # Show camera
    cv2.imshow(
        "Real-Time Hand Gesture Recognition",
        frame
    )


    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camera.release()

cv2.destroyAllWindows()