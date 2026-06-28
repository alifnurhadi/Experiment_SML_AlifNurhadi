import os

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

if __name__ == "__main__":
    DAGSHUB_USERNAME = "<DagsHub_Username>"
    REPO_NAME = "<Repository_Name>"

    mlflow.set_tracking_uri(
        f"https://dagshub.com/{DAGSHUB_USERNAME}/{REPO_NAME}.mlflow"
    )
    mlflow.set_experiment("Student_Burnout_Prediction_Tuning")

    print("Membaca data preprocessed...")
    train_df = pd.read_csv("data/processed/train_processed.csv")
    test_df = pd.read_csv("data/processed/test_processed.csv")

    X_train = train_df.drop("Burnout_Risk_Level", axis=1)
    y_train = train_df["Burnout_Risk_Level"]
    X_test = test_df.drop("Burnout_Risk_Level", axis=1)
    y_test = test_df["Burnout_Risk_Level"]

    with mlflow.start_run(run_name="Manual_Log_LR_Tuned"):
        print("Melatih model Logistic Regression dengan Manual Logging...")

        # Hyperparameters (misal kita ubah parameter C dan solver)
        params = {"max_iter": 2000, "C": 0.5, "solver": "lbfgs", "random_state": 42}

        model = LogisticRegression(**params)
        model.fit(X_train, y_train)

        # Prediksi
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        mlflow.log_params(params)
        mlflow.log_metrics({"accuracy": acc, "f1_score": f1})

        report = classification_report(y_test, y_pred)
        with open("classification_report.txt", "w") as f:
            f.write(report)
        mlflow.log_artifact("classification_report.txt")

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")

        mlflow.sklearn.log_model(model, "logistic_regression_model")

        print(f"Akurasi: {acc:.4f} | F1 Score: {f1:.4f}")
        print("Semua metrik dan artefak berhasil di-log secara manual ke DagsHub!")
