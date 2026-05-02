import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from prefect import flow, task
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.data.load_data import clean_data, create_features, load_raw_data

FEATURES = [
    "hora", "dia_semana", "mes", "es_fin_semana",
    "pm25_lag1", "pm25_lag3", "pm25_lag6", "pm25_lag24",
    "pm25_media3h", "pm25_media6h", "estacion",
]
TARGET = "pm25_siguiente"


@task(name="Cargar datos")
def task_cargar_datos(path: str) -> pd.DataFrame:
    df = load_raw_data(path)
    print(f"  Datos cargados: {df.shape}")
    return df


@task(name="Limpiar datos")
def task_limpiar_datos(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = clean_data(df)
    print(f"  Datos limpios: {df_clean.shape}")
    return df_clean


@task(name="Crear features")
def task_crear_features(df: pd.DataFrame) -> pd.DataFrame:
    df_features = create_features(df)
    print(f"  Features creadas: {df_features.shape}")
    return df_features


@task(name="Entrenar modelo")
def task_entrenar_modelo(df: pd.DataFrame) -> str:
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("prediccion-pm25-siata")

    with mlflow.start_run(run_name="pipeline-random-forest") as run:
        params = {
            "modelo": "RandomForestRegressor",
            "n_estimators": 200,
            "max_depth": 15,
            "min_samples_split": 5,
            "random_state": 42,
        }
        mlflow.log_params(params)

        modelo = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1,
        )
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        mlflow.log_metrics({"mae": mae, "rmse": rmse, "r2": r2})
        mlflow.sklearn.log_model(modelo, "modelo")

        print(f"  MAE: {mae:.2f} | RMSE: {rmse:.2f} | R2: {r2:.3f}")
        return run.info.run_id


@flow(name="Pipeline PM25 SIATA")
def pipeline_pm25(data_path: str = "data/raw/Datos_SIATA_Aire_pm25.json"):
    print("🚀 Iniciando pipeline...")
    df_raw = task_cargar_datos(data_path)
    df_clean = task_limpiar_datos(df_raw)
    df_features = task_crear_features(df_clean)
    run_id = task_entrenar_modelo(df_features)
    print(f"✅ Pipeline completado! Run ID: {run_id}")


if __name__ == "__main__":
    pipeline_pm25()