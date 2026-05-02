import pytest
import pandas as pd
import numpy as np
from src.data.load_data import clean_data, create_features


def test_clean_data_elimina_negativos():
    """Los valores negativos deben convertirse en NaN y luego interpolarse"""
    df = pd.DataFrame({
        "estacion": [1, 1, 1],
        "fecha": pd.to_datetime(["2019-01-01", "2019-01-02", "2019-01-03"]),
        "pm25": [10.0, -9999.0, 20.0],
        "nombre": ["Test", "Test", "Test"],
        "latitud": [6.0, 6.0, 6.0],
        "longitud": [-75.0, -75.0, -75.0],
        "calidad": [1.0, 1.0, 1.0],
    })
    df_clean = clean_data(df)
    assert df_clean["pm25"].isna().sum() == 0
    assert (df_clean["pm25"] >= 0).all()


def test_clean_data_elimina_mayores_500():
    """Los valores mayores a 500 deben eliminarse"""
    df = pd.DataFrame({
        "estacion": [1, 1, 1],
        "fecha": pd.to_datetime(["2019-01-01", "2019-01-02", "2019-01-03"]),
        "pm25": [10.0, 99999.0, 20.0],
        "nombre": ["Test", "Test", "Test"],
        "latitud": [6.0, 6.0, 6.0],
        "longitud": [-75.0, -75.0, -75.0],
        "calidad": [1.0, 1.0, 1.0],
    })
    df_clean = clean_data(df)
    assert (df_clean["pm25"] <= 500).all()


def test_create_features_columnas():
    """El dataframe debe tener las columnas esperadas"""
    df = pd.DataFrame({
        "estacion": [1] * 30,
        "fecha": pd.date_range("2019-01-01", periods=30, freq="h"),
        "pm25": np.random.uniform(10, 50, 30),
        "nombre": ["Test"] * 30,
        "latitud": [6.0] * 30,
        "longitud": [-75.0] * 30,
        "calidad": [1.0] * 30,
    })
    df_features = create_features(df)
    columnas_esperadas = [
        "hora", "dia_semana", "mes", "es_fin_semana",
        "pm25_lag1", "pm25_lag3", "pm25_lag6", "pm25_lag24",
        "pm25_media3h", "pm25_media6h", "pm25_siguiente"
    ]
    for col in columnas_esperadas:
        assert col in df_features.columns, f"Falta columna: {col}"


def test_create_features_no_nulos():
    """No deben quedar nulos en el dataframe de features"""
    df = pd.DataFrame({
        "estacion": [1] * 30,
        "fecha": pd.date_range("2019-01-01", periods=30, freq="h"),
        "pm25": np.random.uniform(10, 50, 30),
        "nombre": ["Test"] * 30,
        "latitud": [6.0] * 30,
        "longitud": [-75.0] * 30,
        "calidad": [1.0] * 30,
    })
    df_features = create_features(df)
    assert df_features.isnull().sum().sum() == 0