import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from src.data.load_data import clean_data, create_features, load_raw_data

FEATURES = [
    "hora", "dia_semana", "mes", "es_fin_semana",
    "pm25_lag1", "pm25_lag3", "pm25_lag6", "pm25_lag24",
    "pm25_media3h", "pm25_media6h", "estacion",
]
TARGET = "pm25_siguiente"

print("Cargando datos...")
df = load_raw_data("data/raw/Datos_SIATA_Aire_pm25.json")
df = clean_data(df)
df = create_features(df)
X = df[FEATURES]
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("prediccion-pm25-siata")

print("Entrenando XGBoost...")
with mlflow.start_run(run_name="xgboost"):
    mlflow.log_params({
        "modelo": "XGBRegressor",
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
    })
    modelo = XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
    )
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mlflow.log_metrics({"mae": mae, "rmse": rmse, "r2": r2})
    print(f"  MAE: {mae:.2f} | RMSE: {rmse:.2f} | R2: {r2:.3f}")

print("XGBoost registrado en MLflow")