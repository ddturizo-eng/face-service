"""
Configuración centralizada del Face Service.
Todos los umbrales están marcados como valores de partida —
se recalibran con datos reales en el Módulo 9.
"""

MODEL_NAME = "ArcFace"
DETECTOR_BACKEND = "yunet"
DISTANCE_METRIC = "cosine"

# Umbrales de calidad de imagen (Capa 1)
BRILLO_MIN = 60
BRILLO_MAX = 200
NITIDEZ_MIN = 20.0
ANCHO_MIN = 200
ALTO_MIN = 200

# Umbrales de calidad de rostro (Capa 2)
ROSTRO_AREA_MIN_RATIO = 0.05
ROSTRO_ANCHO_MIN_PX = 50

# Umbral de similitud para verificación 1:1
# Valor documentado por DeepFace para ArcFace + cosine, NO calibrado con UPC
SIMILARITY_THRESHOLD = 0.68