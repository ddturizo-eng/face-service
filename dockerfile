# ABIS Web - Face Service
# Imagen CPU-only, pensada para hardware limitado (sin GPU dedicada)

FROM python:3.11-slim

# Dependencias de sistema necesarias para OpenCV (libgl1 es requerido por
# opencv-python-headless para decodificación de imágenes, aunque sea "headless")
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Variables de entorno agrupadas juntas
ENV DEEPFACE_HOME=/app/.deepface
ENV TF_CPP_MIN_LOG_LEVEL=2

# Copiar solo requirements primero (aprovecha el cache de capas de Docker:
# si el código cambia pero requirements.txt no, no reinstala dependencias)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código de la aplicación y el script de warm-up
COPY app/ ./app/
COPY warmup.py .

RUN mkdir -p /app/.deepface

# Forzar descarga de modelos durante el BUILD, no en el arranque del
# contenedor. Hace la imagen más pesada pero el arranque en producción
# es inmediato y no depende de conectividad a internet.
RUN python warmup.py

EXPOSE 8001

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]