
import pandas as pd
import os
import json
from sklearn.model_selection import train_test_split

# --------------------------------------------------
# 1. Load dataset from repository data folder
# --------------------------------------------------
DATA_PATH = "data/tourism.csv"

df = pd.read_csv(DATA_PATH)

print("Original dataset shape:", df.shape)

# --------------------------------------------------
# 2. Remove unnecessary columns
# --------------------------------------------------
# Unnamed: 0 is an automatically generated index column.
# CustomerID is a unique identifier and should not be used
# as a predictive feature.

columns_to_remove = ["Unnamed: 0", "CustomerID"]

df = df.drop(columns=columns_to_remove, errors="ignore")

print("Shape after removing unnecessary columns:", df.shape)

# --------------------------------------------------
# 3. Separate target and features
# --------------------------------------------------
X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]

# --------------------------------------------------
# 4. Convert categorical variables to numerical form
# --------------------------------------------------
categorical_columns = X.select_dtypes(
    include=["object"]
).columns.tolist()

X = pd.get_dummies(
    X,
    columns=categorical_columns,
    drop_first=True,
    dtype=int
)

print("Feature shape after encoding:", X.shape)

# --------------------------------------------------
# 5. Train-test split
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

# --------------------------------------------------
# 6. Save the feature column names
# --------------------------------------------------
FEATURE_COLUMNS_PATH = "model_building/feature_columns.json"

with open(FEATURE_COLUMNS_PATH, "w") as f:
    json.dump(X_train.columns.tolist(), f)

# --------------------------------------------------
# 7. Save train and test data
# --------------------------------------------------
X_train.to_csv(
    "model_building/Xtrain.csv",
    index=False
)

X_test.to_csv(
    "model_building/Xtest.csv",
    index=False
)

y_train.to_csv(
    "model_building/ytrain.csv",
    index=False
)

y_test.to_csv(
    "model_building/ytest.csv",
    index=False
)

print("\nData preparation completed successfully.")
print("Training and testing files saved.")
print("Feature column information saved.")
