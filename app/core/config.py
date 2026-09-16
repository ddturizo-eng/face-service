"""
Averyn - Face Service
Configuracion centralizada via variables de entorno.
Todos los umbrales son valores de partida -- se recalibran con datos reales.

Uso: FACE_SIMILARITY_THRESHOLD=0.70 uvicorn app.main:app --port 8001
"""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Configuracion del servicio, cargada de variables de entorno con prefijo FACE_."""

    # Modelos
    MODEL_NAME: str = "ArcFace"
    DETECTOR_BACKEND: str = "yunet"
    DISTANCE_METRIC: str = "cosine"

    # Calidad de imagen (Capa 1)
    BRILLO_MIN: int = 60
    BRILLO_MAX: int = 200
    NITIDEZ_MIN: float = 20.0
    ANCHO_MIN: int = 200
    ALTO_MIN: int = 200

    # Calidad de rostro (Capa 2)
    ROSTRO_AREA_MIN_RATIO: float = 0.05
    ROSTRO_ANCHO_MIN_PX: int = 50

    # Verificacion 1:1
    # Valor documentado por DeepFace para ArcFace + cosine.
    # NO calibrado -- recalibrar con datos reales (Fase 3).
    SIMILARITY_THRESHOLD: float = 0.68

    # Concorrencia y servidor
    # IMPORTANTE: los modelos de DeepFace no son thread-safe. La inferencia
    # se serializa con un lock (pipeline.py). El escalado real se logra con
    # procesos (uvicorn --workers N), no con mas concurrencia por proceso.
    # Valor honesto por worker: 1.
    MAX_CONCURRENT_REQUESTS: int = 1
    HOST: str = "0.0.0.0"
    PORT: int = 8001

    model_config = {"env_prefix": "FACE_"}


settings = Settings()
