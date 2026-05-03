import json

import numpy as np
import pandas as pd


def load_raw_data(path: str) -> pd.DataFrame:
    """Carga el JSON del SIATA y lo convierte a DataFrame"""
    with open(path) as f:
        data = json.load(f)

    registros = []
    for estacion in data:
        for dato in estacion["datos"]:
            registros.append({
                "estacion": estacion["codigoSerial"],
                "nombre": estacion["nombre"],
                "latitud": estacion["latitud"],
                "longitud": estacion["longitud"],
                "fecha": dato["fecha"],
                "pm25": dato["valor"],
                "calidad": dato["calidad"],
            })

    df = pd.DataFrame(registros)
    df["fecha"] = pd.to_datetime(df["fecha"])
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia valores inválidos e interpola"""
    df_clean = df.copy()
    df_clean.loc[df_clean["pm25"] < 0, "pm25"] = np.nan
    df_clean.loc[df_clean["pm25"] > 500, "pm25"] = np.nan
    df_clean = df_clean.sort_values(["estacion", "fecha"]).reset_index(drop=True)
    df_clean["pm25"] = df_clean.groupby("estacion")["pm25"].transform(
        lambda x: x.interpolate(method="linear", limit=3)
    )
    df_clean["pm25"] = df_clean.groupby("estacion")["pm25"].transform(
        lambda x: x.fillna(x.median())
    )
    return df_clean


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crea features temporales y de serie de tiempo"""
    df_f = df.copy()
    df_f["hora"] = df_f["fecha"].dt.hour
    df_f["dia_semana"] = df_f["fecha"].dt.dayofweek
    df_f["mes"] = df_f["fecha"].dt.month
    df_f["es_fin_semana"] = (df_f["dia_semana"] >= 5).astype(int)
    df_f["pm25_lag1"] = df_f.groupby("estacion")["pm25"].shift(1)
    df_f["pm25_lag3"] = df_f.groupby("estacion")["pm25"].shift(3)
    df_f["pm25_lag6"] = df_f.groupby("estacion")["pm25"].shift(6)
    df_f["pm25_lag24"] = df_f.groupby("estacion")["pm25"].shift(24)
    df_f["pm25_media3h"] = df_f.groupby("estacion")["pm25"].transform(
        lambda x: x.shift(1).rolling(3).mean()
    )
    df_f["pm25_media6h"] = df_f.groupby("estacion")["pm25"].transform(
        lambda x: x.shift(1).rolling(6).mean()
    )
    df_f["pm25_siguiente"] = df_f.groupby("estacion")["pm25"].shift(-1)
    df_f = df_f.dropna().reset_index(drop=True)
    return df_f
def validate_data(df: pd.DataFrame) -> bool:
    """Valida que el dataframe tenga la estructura esperada"""
    columnas_requeridas = [
        "estacion", "nombre", "latitud", "longitud", "fecha", "pm25", "calidad"
    ]
    
    for col in columnas_requeridas:
        if col not in df.columns:
            raise ValueError(f"Columna requerida no encontrada: {col}")
    
    if df.empty:
        raise ValueError("El dataframe está vacío")
    
    if df["pm25"].isna().all():
        raise ValueError("Todos los valores de PM2.5 son nulos")
    
    if not pd.api.types.is_datetime64_any_dtype(df["fecha"]):
        raise ValueError("La columna fecha no es de tipo datetime")
    
    print(f"✅ Validación exitosa: {len(df):,} registros, {len(df.columns)} columnas")
    return True