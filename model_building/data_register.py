
import pandas as pd
import os

DATA_PATH = "tourism_project/data/tourism.csv"

EXPECTED_COLUMNS = [
    "CustomerID",
    "ProdTaken",
    "Age",
    "TypeofContact",
    "CityTier",
    "Occupation",
    "Gender",
    "NumberOfPersonVisiting",
    "PreferredPropertyStar",
    "MaritalStatus",
    "NumberOfTrips",
    "Passport",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "Designation",
    "MonthlyIncome",
    "PitchSatisfactionScore",
    "ProductPitched",
    "NumberOfFollowups",
    "DurationOfPitch"
]

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

print("Dataset registered successfully.")
print("Dataset shape:", df.shape)

missing_columns = [
    col for col in EXPECTED_COLUMNS
    if col not in df.columns
]

if missing_columns:
    raise ValueError(f"Missing expected columns: {missing_columns}")

print("\nAll expected columns are present.")

extra_columns = [
    col for col in df.columns
    if col not in EXPECTED_COLUMNS
]

if extra_columns:
    print("Additional columns found:", extra_columns)

print("\nDataset summary:")
print(df.describe(include="all").T)

print("\nMissing values:")
print(df.isnull().sum())

print("\nTarget variable distribution:")
print(df["ProdTaken"].value_counts())
