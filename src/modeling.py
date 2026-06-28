import os

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Student_Burnout_Prediction")

    mlflow.sklearn.autolog()

    print("Membaca data preprocessed...")
    train_df = pd.read_csv("data/processed/train_processed.csv")
    test_df = pd.read_csv("data/processed/test_processed.csv")

    X_train = train_df.drop("Burnout_Risk_Level", axis=1)
    y_train = train_df["Burnout_Risk_Level"]

    X_test = test_df.drop("Burnout_Risk_Level", axis=1)
    y_test = test_df["Burnout_Risk_Level"]

    # 3. Memulai MLflow Run
    with mlflow.start_run(run_name="Logistic_Regression_Baseline"):
        print("Melatih model Logistic Regression...")

        # Model tanpa hyperparameter tuning kompleks
        model = LogisticRegression(max_iter=1000, random_state=42)

        # Saat fungsi .fit() dipanggil, MLflow Autolog akan otomatis mencatat:
        # Parameter, metrik training, dan model artifact.
        model.fit(X_train, y_train)

        # Evaluasi pada data uji
        print("Mengevaluasi model pada test set...")
        y_pred = model.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)

        # Anda juga bisa menambahkan log manual jika diperlukan
        mlflow.log_metric("test_accuracy", test_acc)

        print(f"Akurasi Model pada Test Set: {test_acc:.4f}")
        print("Selesai! Model dan metrik telah dicatat oleh MLflow di folder ./mlruns")
