import pickle
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import yaml
import json
import os

import os  # Add this line
import json
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import yaml


# Load params
with open("params.yaml") as f:
    params = yaml.safe_load(f)

test_size = params["train"]["test_size"]
random_state = params["train"]["random_state"]

# Load data
data = pd.read_csv("data/raw/train.csv")
data.dropna(subset=["Survived"], inplace=True)
X = data[["Pclass", "SibSp", "Parch"]]
y = data["Survived"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

# Load model
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# ✅ Save result to evaluation/metrics.json
os.makedirs("evaluation", exist_ok=True)
with open("evaluation/metrics.json", "w") as f:
    json.dump({"accuracy": accuracy}, f)

print(f"✅ Model evaluation complete. Accuracy: {accuracy:.4f}")
