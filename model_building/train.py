
import os
import json
import joblib
import mlflow
import mlflow.sklearn

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)


# --------------------------------------------------
# 1. Load train and test data
# --------------------------------------------------

BASE_PATH = "model_building"

X_train = pd.read_csv(os.path.join(BASE_PATH, "Xtrain.csv"))
X_test = pd.read_csv(os.path.join(BASE_PATH, "Xtest.csv"))

y_train = pd.read_csv(
    os.path.join(BASE_PATH, "ytrain.csv")
).squeeze()

y_test = pd.read_csv(
    os.path.join(BASE_PATH, "ytest.csv")
).squeeze()

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)


# --------------------------------------------------
# 2. Define MLflow tracking
# --------------------------------------------------

mlflow.set_tracking_uri(
    os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")
)

mlflow.set_experiment("Visit-with-Us-Tourism")


# --------------------------------------------------
# 3. Define model
# --------------------------------------------------

model = RandomForestClassifier(
    random_state=42,
    n_jobs=-1
)


# --------------------------------------------------
# 4. Define hyperparameter grid
# --------------------------------------------------

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}


# --------------------------------------------------
# 5. Hyperparameter tuning
# --------------------------------------------------

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=3,
    scoring="f1",
    n_jobs=-1,
    verbose=1
)

with mlflow.start_run(run_name="RandomForest_Hyperparameter_Tuning"):

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    print("\nBest parameters:")
    print(best_params)

    # Log tuned parameters
    mlflow.log_params(best_params)

    # --------------------------------------------------
    # 6. Evaluate best model
    # --------------------------------------------------

    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)

    # Log evaluation metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", roc_auc)

    mlflow.sklearn.log_model(
        best_model,
        "random_forest_model"
    )

    print("\nModel Evaluation:")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


# --------------------------------------------------
# 7. Save best model for deployment
# --------------------------------------------------

DEPLOYMENT_PATH = "deployment"

os.makedirs(DEPLOYMENT_PATH, exist_ok=True)

MODEL_PATH = os.path.join(
    DEPLOYMENT_PATH,
    "best_model.pkl"
)

joblib.dump(best_model, MODEL_PATH)


# Save feature columns for consistent deployment preprocessing
FEATURE_PATH = os.path.join(
    BASE_PATH,
    "feature_columns.json"
)

if os.path.exists(FEATURE_PATH):

    with open(FEATURE_PATH, "r") as f:
        feature_columns = json.load(f)

    DEPLOYMENT_FEATURE_PATH = os.path.join(
        DEPLOYMENT_PATH,
        "feature_columns.json"
    )

    with open(DEPLOYMENT_FEATURE_PATH, "w") as f:
        json.dump(feature_columns, f)

print("\nBest model saved to:")
print(MODEL_PATH)

print("\nModel training and experimentation completed successfully.")
