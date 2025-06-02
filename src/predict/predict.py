import pickle
import pandas as pd
import os

# Load test data
test_path = os.path.join("data", "raw", "test.csv")
test_data = pd.read_csv(test_path)

# Simple feature selection (same as training)
X_test = test_data[["Pclass", "SibSp", "Parch"]]

# Load trained model
model_path = os.path.join("models", "model.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Predict
predictions = model.predict(X_test)

# Save predictions
output = pd.DataFrame({
    "PassengerId": test_data["PassengerId"],
    "Survived": predictions
})
os.makedirs("outputs", exist_ok=True)
output.to_csv("outputs/predictions.csv", index=False)

print("✅ Predictions saved to outputs/predictions.csv")
