FROM python:3.13-slim

WORKDIR /app

# Copiar archivos del proyecto
COPY pyproject.toml .
COPY src/ ./src/
COPY mlartifacts/ ./mlartifacts/
COPY mlflow.db ./mlflow.db

# Instalar dependencias
RUN pip install fastapi uvicorn mlflow scikit-learn pandas numpy

# Exponer el puerto
EXPOSE 8000

# Variables de entorno para MLflow
ENV MLFLOW_TRACKING_URI=sqlite:///mlflow.db

# Comando para arrancar la API
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]