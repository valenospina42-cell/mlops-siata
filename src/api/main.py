import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="API Predicción PM2.5 - SIATA",
    description="Predice el nivel de PM2.5 de la próxima hora",
    version="1.0.0",
)

# Cargar modelo al iniciar la API
import pickle
modelo = pickle.load(open("mlartifacts/1/models/m-0ade34a865da49ab8184eb2f915cc689/artifacts/model.pkl", "rb"))


class DatosEntrada(BaseModel):
    estacion: int
    hora: int
    dia_semana: int
    mes: int
    es_fin_semana: int
    pm25_lag1: float
    pm25_lag3: float
    pm25_lag6: float
    pm25_lag24: float
    pm25_media3h: float
    pm25_media6h: float


class Prediccion(BaseModel):
    pm25_predicho: float
    categoria: str
    mensaje: str


def categorizar_pm25(valor: float) -> tuple[str, str]:
    if valor <= 12:
        return "Bueno", " Calidad del aire buena"
    elif valor <= 25:
        return "Moderado", "🟡 Calidad del aire moderada"
    elif valor <= 50:
        return "Dañino para grupos sensibles", "🟠 Precaución para grupos sensibles"
    else:
        return "Dañino", "🔴 Calidad del aire dañina"


@app.get("/")
def root():
    return {"mensaje": "API PM2.5 SIATA funcionando "}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predecir", response_model=Prediccion)
def predecir(datos: DatosEntrada):
    FEATURES = [
        "hora", "dia_semana", "mes", "es_fin_semana",
        "pm25_lag1", "pm25_lag3", "pm25_lag6", "pm25_lag24",
        "pm25_media3h", "pm25_media6h", "estacion",
    ]
    df = pd.DataFrame([datos.model_dump()])[FEATURES]
    pm25_predicho = float(modelo.predict(df)[0])
    categoria, mensaje = categorizar_pm25(pm25_predicho)
    return Prediccion(
        pm25_predicho=round(pm25_predicho, 2),
        categoria=categoria,
        mensaje=mensaje,
    )