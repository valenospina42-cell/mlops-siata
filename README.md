# MLOps SIATA - Predicción de Calidad del Aire en Medellín

Proyecto final del curso MLOps - Universidad de Medellín.

## Descripción

Modelo de machine learning que predice el nivel de PM2.5 (partículas finas)
de la próxima hora en las estaciones de monitoreo del SIATA en el Valle de Aburrá.

**Problema de negocio:** Las autoridades ambientales pueden anticipar episodios
de mala calidad del aire con 1 hora de anticipación y tomar medidas preventivas.

## Alcance del proyecto

|**COMPONENTE**|**Descripción**|
|-|-|
|MVP|Pipeline de entrenamiento + API de predicción local|
|Funcionalidad completa|API dockerizada + CI/CD + monitoreo propuesto|



## Timeline

|FASE|TAREA|TIEMPO ESTIMADO|RESPONSABLE|
|-|-|-|-|
|1|Setup del entorno y EDA|2 horas|Valentina Ospina|
|2|Preprocesamiento y features|1 hora|Valentina Ospina|
|3|Experimentos con MLflow|2 horas|Valentina Ospina|
|4|Pipeline con Prefect|1 hora|Valentina Ospina|
|<br />5|API con FastAPI|1 hora|Valentina Ospina|
|6|Docker|1 horas|Valentina Ospina|
|7|Tests y documentación|1 horas|Valentina Ospina|
|8|CI/CD con GitHub Actions|30 minutos|Valentina Ospina|



## Dataset

**## Dataset**



**- \*\*Fuente:\*\* SIATA (Sistema de Alerta Temprana de Medellín y el Valle de Aburrá)**

**- \*\*Período:\*\* Agosto 2018 - Agosto 2019**

**- \*\*Estaciones:\*\* 21 estaciones en el Valle de Aburrá**

**- \*\*Registros:\*\* \~184,000 mediciones horarias de PM2.5**

**- \*\*Descarga datos crudos:\*\* \[Datos\_SIATA\_Aire\_pm25.json](https://drive.google.com/file/d/1crae2SE-R8-m2FtlmTXZCfEFkdy2cqaC/view?usp=drive\_link)**

**- \*\*Descarga datos procesados:\*\* \[siata\_features.csv](https://drive.google.com/file/d/15vLD3NMS2eGyd3pcbjE2qWEzox77g94y/view?usp=drive\_link)**



**⚠️ Coloca los archivos así:**

**- data/raw/Datos\_SIATA\_Aire\_pm25.json**

**- data/processed/siata\_features.csv**

## Resultados del modelo

|Modelo|MAE|RMSE|R2|
|-|-|-|-|
|Regresión Lineal|6.12|8.94|0.511|
|Random Forest base|5.59|8.11|0.597|
|Random Forest optimizado|5.49|8.00|0.608|
|**XGBoost**|**5.29**|**7.72**|**0.635**|

## Inicio rápido

Sigue estos pasos en orden para reproducir el proyecto completo:

### 1\. Clonar e instalar

```bash
git clone https://github.com/valenospina42-cell/mlops-siata.git
cd mlops-siata
uv sync
```

### 2\. Agregar los datos

Coloca el archivo `Datos\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_SIATA\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_Aire\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_pm25.json` en la carpeta `data/raw/`.

### 3\. Preprocesar los datos

Ejecuta el notebook `notebooks/02\\\\\\\_preprocessing.ipynb` completo para generar

el archivo `data/processed/siata\\\\\\\_features.csv`.



O ejecuta directamente desde la terminal:

```python

uv run python -c "

import json, pandas as pd, numpy as np

from src.data.load\\\\\\\_data import load\\\\\\\_raw\\\\\\\_data, clean\\\\\\\_data, create\\\\\\\_features

df = load\\\\\\\_raw\\\\\\\_data('data/raw/Datos\\\\\\\_SIATA\\\\\\\_Aire\\\\\\\_pm25.json')

df = clean\\\\\\\_data(df)

df = create\\\\\\\_features(df)

df.to\\\\\\\_csv('data/processed/siata\\\\\\\_features.csv', index=False)

print(' Datos procesados guardados en data/processed/siata\\\\\\\_features.csv')

"

```

### 4\. Ejecutar el pipeline de entrenamiento

```bash

\\\\# Terminal 1: arrancar MLflow

uv run mlflow ui



\\\\# Terminal 2: ejecutar el pipeline

uv run python -m src.models.train

### ```

### 5\\\\. Ejecutar con Docker

```bash
docker build -t pm25-siata-api .
docker run -p 8000:8000 pm25-siata-api
```

### 6\. Ejecutar tests

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
│   ├── data/load\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_data.py     # Carga, limpieza y features
│   ├── models/train.py       # Pipeline de entrenamiento (Prefect)
│   └── api/main.py           # API de predicción (FastAPI)
├── notebooks/
│   ├── 01\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_eda.ipynb          # Análisis exploratorio
│   ├── 02\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_preprocessing.ipynb
│   └── 03\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_experiments.ipynb  # Experimentos con MLflow
├── tests/                    # Unit tests
├── Dockerfile
└── README.md
```

## Tecnologías usadas

* **ML:** scikit-learn (Random Forest), XGBoost
* **Experiment Tracking:** MLflow
* **Pipeline:** Prefect
* **API:** FastAPI + Uvicorn
* **Containerización:** Docker
* **Testing:** pytest
* **Linter:** ruff
* **CI/CD:** GitHub Actions

## Monitoreo (propuesta)

* Registrar las predicciones vs valores reales en una base de datos
* Dashboard en Grafana con métricas de drift del modelo
* Alertas cuando el MAE supere un umbral definido

