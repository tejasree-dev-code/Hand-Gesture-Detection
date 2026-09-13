# Hand Gesture Detection

## Project Description

Hand Gesture Detection is a real-time AI/ML project that recognizes hand gestures using a webcam and converts them into text and speech.

The project uses OpenCV to capture video, MediaPipe to detect hand landmarks, and a Random Forest machine learning model to classify the gestures.

## Technologies Used

- Python
- OpenCV
- MediaPipe
- NumPy
- Pandas
- Scikit-learn
- Random Forest
- Joblib
- pyttsx3

## How It Works

1. The webcam captures the user's hand.
2. OpenCV processes the video frame.
3. MediaPipe detects the hand and extracts 21 hand landmarks.
4. The landmark coordinates are converted into numerical features.
5. The features are given to the Random Forest model.
6. The model predicts the gesture.
7. The predicted gesture is displayed as text.
8. The gesture can also be converted into speech.

## Gestures Supported

| Gesture | Output |
|---|---|
| Open Palm | HELLO |
| Thumbs Up | YES |
| Thumbs Down | NO |
| Fist | STOP |
| Peace Sign | PEACE |

## Machine Learning Model

A Random Forest Classifier is used for gesture classification.

The dataset contains 2,000 samples with 5 different gesture classes.

The current model achieved approximately **98.75% accuracy** on the test dataset.

## Project Files

- `main.py` - Detects hand landmarks using MediaPipe.
- `collect_data.py` - Collects hand gesture training data.
- `train_model.py` - Trains the Random Forest model.
- `predict.py` - Performs real-time gesture prediction and speech.
- `test_model.py` - Tests the trained model.
- `gesture_data.csv` - Gesture landmark dataset.
- `gesture_model.pkl` - Trained machine learning model.
- `requirements.txt` - Required Python libraries.

## How to Run

```bash
1. pip install -r requirements.txt
Activate it on Windows:
venv\Scripts\activate

2.Install the required libraries
pip install -r requirements.txt

3. Run the gesture prediction
python predict.py
Press q to close the camera window.

Future Improvements:

Collect training samples under different lighting conditions.
Add more hand gestures.
Improve real-world accuracy.
Add more advanced gesture and sign-language recognition.
