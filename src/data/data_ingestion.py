import pandas as pd
import os

# Create raw data directory
data_path = os.path.join("data", "raw")
os.makedirs(data_path, exist_ok=True)

# URLs for Titanic datasets (you can replace with your own source if needed)
train_url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
test_url = "https://raw.githubusercontent.com/agconti/kaggle-titanic/master/data/test.csv"

# Download datasets
train_data = pd.read_csv(train_url)
test_data = pd.read_csv(test_url)

# Save locally to data/raw/
train_data.to_csv(os.path.join(data_path, "train.csv"), index=False)
test_data.to_csv(os.path.join(data_path, "test.csv"), index=False)

print("✅ Data ingestion complete.")
