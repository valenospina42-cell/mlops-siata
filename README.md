# MLOps SIATA - Predicción de Calidad del Aire en Medellín

Proyecto final del curso MLOps - Universidad de Medellín.

## Descripción

Modelo de machine learning que predice el nivel de PM2.5 (partículas finas)
de la próxima hora en las estaciones de monitoreo del SIATA en el Valle de Aburrá.

**Problema de negocio:** Las autoridades ambientales pueden anticipar episodios
de mala calidad del aire con 1 hora de anticipación y tomar medidas preventivas.

## Dataset

- **Fuente:** SIATA (Sistema de Alerta Temprana de Medellín y el Valle de Aburrá)
- **Período:** Agosto 2018 - Agosto 2019
- **Estaciones:** 21 estaciones en el Valle de Aburrá
- **Registros:** ~184,000 mediciones horarias de PM2.5

## Resultados del modelo

| Modelo | MAE | RMSE | R2 |
|---|---|---|---|
| Regresión Lineal | 6.12 | 8.94 | 0.511 |
| Random Forest base | 5.59 | 8.11 | 0.597 |
| **Random Forest optimizado** | **5.49** | **8.00** | **0.608** |

## Estructura del proyecto
mlops-siata/
├── data/
│   ├── raw/                  # Datos originales del SIATA
│   └── processed/            # Datos procesados con features
├── src/
│   ├── data/load_data.py     # Carga, limpieza y features
│   ├── models/train.py       # Pipeline de entrenamiento (Prefect)
│   └── api/main.py           # API de predicción (FastAPI)
├── notebooks/
│   ├── 01_eda.ipynb          # Análisis exploratorio
│   ├── 02_preprocessing.ipynb
│   └── 03_experiments.ipynb  # Experimentos con MLflow
├── tests/                    # Unit tests
├── Dockerfile
└── README.md




