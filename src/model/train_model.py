import yaml
import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load parameters
with open("params.yaml") as f:
    params = yaml.safe_load(f)

test_size = params["train"]["test_size"]
random_state = params["train"]["random_state"]

# Load data
train_data = pd.read_csv("data/raw/train.csv")

# Drop rows with missing target
train_data.dropna(subset=["Survived"], inplace=True)

# Simple preprocessing (just as an example)
X = train_data[["Pclass", "SibSp", "Parch"]]
y = train_data["Survived"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

# Train model
model = RandomForestClassifier(random_state=random_state)
model.fit(X_train, y_train)

# Save model
os.makedirs("models", exist_ok=True)
with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved to models/model.pkl")
