import time

import mlflow.sklearn
import pandas as pd
import psutil
from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app
from pydantic import BaseModel

app = FastAPI()

# --- 10 METRIKS PROMETHEUS (Kriteria Advance) ---
REQUEST_COUNT = Counter(
    "api_requests_total", "Total API requests", ["method", "endpoint"]
)
ERROR_COUNT = Counter("api_errors_total", "Total API errors")
LATENCY = Histogram("api_latency_seconds", "API latency in seconds")
CPU_USAGE = Gauge("system_cpu_usage_percent", "CPU usage percent")
MEM_USAGE = Gauge("system_memory_usage_bytes", "Memory usage in bytes")
ACTIVE_REQUESTS = Gauge("api_active_requests", "Number of active requests")
PREDICTION_HIGH = Counter(
    "model_predictions_high_risk_total", "Total High Risk Predictions"
)
PREDICTION_LOW_MED = Counter(
    "model_predictions_low_med_risk_total", "Total Low/Med Risk Predictions"
)
INPUT_GENAI_HOURS = Histogram(
    "model_input_genai_hours", "Distribution of GenAI hours input"
)
INPUT_GPA = Histogram("model_input_gpa", "Distribution of GPA input")

# Mount Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Load Model (Ganti URI dengan format yang sesuai jika menggunakan MLflow/lokal)
# model = mlflow.sklearn.load_model("runs:/<RUN_ID>/logistic_regression_model")


class StudentData(BaseModel):
    Weekly_GenAI_Hours: float
    Pre_Semester_GPA: float
    # Tambahkan fitur lain sesuai X_train...


@app.post("/predict")
def predict(data: StudentData):
    ACTIVE_REQUESTS.inc()
    start_time = time.time()
    REQUEST_COUNT.labels(method="POST", endpoint="/predict").inc()

    try:
        # Update System Metrics
        CPU_USAGE.set(psutil.cpu_percent())
        MEM_USAGE.set(psutil.virtual_memory().used)

        # Track Input Distributions
        INPUT_GENAI_HOURS.observe(data.Weekly_GenAI_Hours)
        INPUT_GPA.observe(data.Pre_Semester_GPA)

        # Prediksi (Mockup - sesuaikan dengan input model sebenarnya)
        # input_df = pd.DataFrame([data.dict()])
        # pred = model.predict(input_df)[0]
        pred = "High" if data.Weekly_GenAI_Hours > 10 else "Low"  # Mockup logic

        if pred == "High":
            PREDICTION_HIGH.inc()
        else:
            PREDICTION_LOW_MED.inc()

        LATENCY.observe(time.time() - start_time)
        ACTIVE_REQUESTS.dec()
        return {"prediction": pred}

    except Exception as e:
        ERROR_COUNT.inc()
        ACTIVE_REQUESTS.dec()
        raise HTTPException(status_code=500, detail=str(e))
