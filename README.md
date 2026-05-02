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
- **Descarga:** [Datos SIATA PM2.5](https://www.siata.gov.co)

> ⚠️ Los datos crudos no están incluidos en el repositorio por su tamaño.
> Descarga el archivo JSON y colócalo en `data/raw/Datos_SIATA_Aire_pm25.json` antes de ejecutar el pipeline.

## Resultados del modelo

| Modelo | MAE | RMSE | R2 |
|---|---|---|---|
| Regresión Lineal | 6.12 | 8.94 | 0.511 |
| Random Forest base | 5.59 | 8.11 | 0.597 |
| Random Forest optimizado | 5.49 | 8.00 | 0.608 |
| **XGBoost** | **5.29** | **7.72** | **0.635** |

## Inicio rápido

Sigue estos pasos en orden para reproducir el proyecto completo:

### 1. Clonar e instalar

```bash
git clone https://github.com/valenospina42-cell/mlops-siata.git
cd mlops-siata
uv sync
```

### 2. Agregar los datos

Coloca el archivo `Datos_SIATA_Aire_pm25.json` en la carpeta `data/raw/`.

### 3. Ejecutar el pipeline de entrenamiento

```bash
# Terminal 1: arrancar MLflow
uv run mlflow ui

# Terminal 2: ejecutar el pipeline (entrena el modelo y lo registra en MLflow)
uv run python -m src.models.train
```

Esto genera el modelo entrenado y lo registra en MLflow en `mlartifacts/`.

### 4. Levantar la API

```bash
uv run uvicorn src.api.main:app --port 8000
```

La API estará disponible en `http://localhost:8000/docs`

### 5. Ejecutar con Docker

```bash
docker build -t pm25-siata-api .
docker run -p 8000:8000 pm25-siata-api
```

### 6. Ejecutar tests

```bash
uv run pytest tests/ -v
```

## Estructura del proyecto

```
mlops-siata/
├── data/
│   ├── raw/                  # Datos originales del SIATA (no incluidos)
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
```

## Tecnologías usadas

- **ML:** scikit-learn (Random Forest), XGBoost
- **Experiment Tracking:** MLflow
- **Pipeline:** Prefect
- **API:** FastAPI + Uvicorn
- **Containerización:** Docker
- **Testing:** pytest
- **Linter:** ruff
- **CI/CD:** GitHub Actions

## Monitoreo (propuesta)

- Registrar las predicciones vs valores reales en una base de datos
- Dashboard en Grafana con métricas de drift del modelo
- Alertas cuando el MAE supere un umbral definido
