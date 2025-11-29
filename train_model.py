import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Load dataset
df = pd.read_csv("road_accidents.csv")  # make sure CSV exists

X = df.drop("severity", axis=1)  # features
y = df["severity"]               # target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the trained model
with open("accident_severity_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("✅ Model saved as accident_severity_model.pkl")
