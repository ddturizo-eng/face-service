"""Fase 4: Metricas Prometheus para el pipeline de face service."""
from prometheus_client import Counter, Histogram

# Histogramas por etapa del pipeline (milisegundos)
PIPELINE_STAGES: dict[str, Histogram] = {
    "resize": Histogram(
        "face_pipeline_resize_ms",
        "Resize stage duration in ms",
        buckets=[1, 5, 10, 25, 50, 100, 250],
    ),
    "calidad_imagen": Histogram(
        "face_pipeline_calidad_imagen_ms",
        "Image quality check duration in ms",
        buckets=[1, 5, 10, 25, 50, 100],
    ),
    "deteccion_y_liveness": Histogram(
        "face_pipeline_deteccion_liveness_ms",
        "Detection + liveness duration in ms",
        buckets=[50, 100, 200, 500, 1000, 2000, 5000],
    ),
    "calidad_rostro": Histogram(
        "face_pipeline_calidad_rostro_ms",
        "Face quality check duration in ms",
        buckets=[1, 5, 10, 25, 50],
    ),
    "embedding": Histogram(
        "face_pipeline_embedding_ms",
        "Embedding generation duration in ms",
        buckets=[50, 100, 200, 500, 1000, 2000],
    ),
}

# Contadores
FACE_REQUESTS = Counter(
    "face_requests_total",
    "Total face requests processed",
    ["endpoint", "result"],
)

FACE_CONCURRENCY_REJECTED = Counter(
    "face_concurrency_rejected_total",
    "Requests rejected by concurrency semaphore (503)",
    ["endpoint"],
)

FACE_QUALITY_ISSUES = Counter(
    "face_quality_issues_total",
    "Quality rejection reasons by type",
    ["reason"],
)
