import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("gesture_data.csv")

# Separate features and labels
X = data.drop("label", axis=1)
y = data["label"]

X = X.values

# Normalize using wrist
for i in range(len(X)):

    wrist_x = X[i][0]
    wrist_y = X[i][1]
    wrist_z = X[i][2]

    for j in range(0, 63, 3):

        X[i][j] = X[i][j] - wrist_x
        X[i][j + 1] = X[i][j + 1] - wrist_y
        X[i][j + 2] = X[i][j + 2] - wrist_z

# Load trained model
model = joblib.load("gesture_model.pkl")

# Predict
predictions = model.predict(X)

# Calculate accuracy
accuracy = accuracy_score(y, predictions)

print("Accuracy on collected data:", accuracy * 100, "%")