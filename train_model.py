import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Read dataset
data = pd.read_csv("gesture_data.csv")

# Separate features and labels
X = data.drop("label", axis=1)
y = data["label"]

# Convert dataframe to numbers
X = X.values

# Normalize landmarks using the wrist as reference
for i in range(len(X)):

    wrist_x = X[i][0]
    wrist_y = X[i][1]
    wrist_z = X[i][2]

    for j in range(0, 63, 3):

        X[i][j] = X[i][j] - wrist_x
        X[i][j + 1] = X[i][j + 1] - wrist_y
        X[i][j + 2] = X[i][j + 2] - wrist_z

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy * 100, "%")

# Detailed evaluation
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "gesture_model.pkl")

print("\nModel saved successfully!")