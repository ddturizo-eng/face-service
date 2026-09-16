"""
Averyn - Face Service
Microservicio FastAPI para deteccion, calidad, liveness y verificacion facial.

Fase 1: Thread limits antes de importar librerias pesadas.
Fase 2: Endpoint /enroll, config via env vars.
Fase 4: Metricas Prometheus, logging JSON, healthchecks, codigos HTTP correctos.
"""
import os

# ---------------------------------------------------------------------------
# Fase 1: Thread limits ANTES de importar librerias pesadas para que
# OpenCV, TensorFlow y PyTorch no compitan por todos los cores.
# ---------------------------------------------------------------------------
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import asyncio
import json
import time
import logging
from contextlib import asynccontextmanager

import numpy as np
import cv2
import torch
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from prometheus_fastapi_instrumentator import Instrumentator
from deepface import DeepFace

from app.core.config import settings
from app.core.logging_config import setup_logging
from app.core.metrics import (
    PIPELINE_STAGES,
    FACE_REQUESTS,
    FACE_CONCURRENCY_REJECTED,
    FACE_QUALITY_ISSUES,
)
from app.services.pipeline import procesar_verificacion
from app.schemas.face import (
    VerifyResponse,
    VerifyResponseData,
    VerifyResponseMeta,
    EnrollResponse,
    EnrollResponseData,
)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
setup_logging()
logger = logging.getLogger("averyn.face")

# Estado global del lifespan
_semaphore: asyncio.Semaphore | None = None
_models_ready: bool = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm-up de modelos + inicializacion de semaforo."""
    global _semaphore, _models_ready

    _semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_REQUESTS)

    # Fase 1: limitar threads post-import
    cv2.setNumThreads(1)
    torch.set_num_threads(1)

    # Warm-up: forzar carga de modelos en memoria antes del primer request
    dummy = np.zeros((300, 300, 3), dtype=np.uint8)
    try:
        DeepFace.extract_faces(
            img_path=dummy,
            detector_backend=settings.DETECTOR_BACKEND,
            anti_spoofing=True,
            enforce_detection=False,
        )
        DeepFace.represent(
            img_path=dummy,
            model_name=settings.MODEL_NAME,
            detector_backend="skip",
            enforce_detection=False,
        )
        _models_ready = True
    except Exception:
        _models_ready = False
        logger.warning("Warm-up de modelos fallo; se cargaran bajo demanda.")

    logger.info(
        "Face Service listo",
        extra={
            "models_ready": _models_ready,
            "max_concurrent": settings.MAX_CONCURRENT_REQUESTS,
        },
    )
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Averyn Face Service",
    version="0.2.0",
    description="Microservicio de reconocimiento facial: calidad, liveness y verificacion 1:1.",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Fase 4: metricas HTTP automaticas (request count, latency, etc.)
Instrumentator().instrument(app).expose(app)


# ---------------------------------------------------------------------------
# Fase 4: manejo consistente de errores tecnicos
# ---------------------------------------------------------------------------
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Convierte cualquier excepcion no manejada en un 500 JSON consistente.

    Importante: el 500 es un ERROR TECNICO (no un rechazo de negocio).
    El adapter NestJS debe reintentar con backoff y escalar si persiste.
    """
    logger.error(
        "Error no manejado",
        extra={"endpoint": request.url.path, "error_type": type(exc).__name__},
        exc_info=True,
    )
    return _error_response(
        500,
        f"Error interno: {type(exc).__name__}. Consulte los logs para detalles.",
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Convierte errores de validacion de FastAPI en 422 con body consistente."""
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "data": None,
            "meta": {
                "model": settings.MODEL_NAME,
                "detector": settings.DETECTOR_BACKEND,
                "processingTimeMs": 0,
            },
            "error": str(exc.errors()),
        },
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _leer_imagen(file_bytes: bytes) -> np.ndarray:
    array = np.frombuffer(file_bytes, dtype=np.uint8)
    return cv2.imdecode(array, cv2.IMREAD_COLOR)


def _error_response(status_code: int, message: str) -> JSONResponse:
    """Respuesta de error consistente para todos los endpoints."""
    return JSONResponse(
        status_code=status_code,
        content={
            "success": False,
            "data": None,
            "meta": {
                "model": settings.MODEL_NAME,
                "detector": settings.DETECTOR_BACKEND,
                "processingTimeMs": 0,
            },
            "error": message,
        },
    )


def _observe_pipeline_metrics(tiempos: dict) -> None:
    """Exporta metricas por etapa del pipeline a Prometheus."""
    for stage, ms in tiempos.items():
        if stage in PIPELINE_STAGES:
            PIPELINE_STAGES[stage].observe(ms)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    """Healthcheck basico: el proceso esta vivo."""
    return {"status": "ok"}


@app.get("/api/v1/face/health")
def detailed_health():
    """Healthcheck detallado: verifica que los modelos estan cargados."""
    return {
        "status": "healthy" if _models_ready else "degraded",
        "models_loaded": _models_ready,
        "concurrent_max": settings.MAX_CONCURRENT_REQUESTS,
    }


@app.post(
    "/api/v1/face/verify",
    response_model=VerifyResponse,
    tags=["verificacion"],
    summary="Verificacion facial 1:1",
    description=(
        "Compara la foto recibida contra un embedding previamente registrado. "
        "Devuelve similarity (coseno) y verified (true si similarity >= umbral). "
        "Si no se proporciona embedding_registrado, solo valida calidad + liveness."
    ),
    responses={
        400: {"description": "embedding_registrado no es JSON valido"},
        503: {"description": "Capacidad maxima alcanzada (backpressure)"},
    },
)
async def verify(
    imagen: UploadFile = File(..., description="Foto a analizar"),
    embedding_registrado: str = Form(
        None,
        description='JSON string: lista de 512 floats del embedding registrado',
    ),
):
    """Verificacion facial 1:1."""
    # --- Fase 1: Backpressure via semaforo ---
    if _semaphore.locked():
        FACE_CONCURRENCY_REJECTED.labels(endpoint="verify").inc()
        return _error_response(503, "Capacidad maxima alcanzada. Intente de nuevo.")

    async with _semaphore:
        t_start = time.perf_counter()

        contenido = await imagen.read()
        img = _leer_imagen(contenido)

        emb_registrado = None
        if embedding_registrado:
            try:
                emb_registrado = json.loads(embedding_registrado)
            except json.JSONDecodeError:
                return _error_response(400, "embedding_registrado no es un JSON valido.")

        resultado, _, tiempos = await run_in_threadpool(
            procesar_verificacion, img, emb_registrado
        )

        total_ms = round((time.perf_counter() - t_start) * 1000, 2)

        # Fase 4: metricas
        _observe_pipeline_metrics(tiempos)
        FACE_REQUESTS.labels(endpoint="verify", result="ok").inc()
        for issue in resultado.get("qualityIssues", []):
            reason = issue.split(" ")[0] if issue else "unknown"
            FACE_QUALITY_ISSUES.labels(reason=reason).inc()

        logger.info(
            "verify completado",
            extra={
                "endpoint": "verify",
                "processing_time_ms": total_ms,
                "result": "ok",
            },
        )

        return VerifyResponse(
            success=True,
            data=VerifyResponseData(
                **{k: v for k, v in resultado.items() if k != "embedding"}
            ),
            meta=VerifyResponseMeta(
                model=settings.MODEL_NAME,
                detector=settings.DETECTOR_BACKEND,
                processingTimeMs=total_ms,
            ),
        )


@app.post(
    "/api/v1/face/enroll",
    response_model=EnrollResponse,
    tags=["enrolamiento"],
    summary="Enrolamiento facial",
    description=(
        "Procesa la foto pasando por calidad + liveness y devuelve el embedding "
        "de 512 dimensiones. El caller debe almacenar el embedding (ej. en "
        "Supabase/pgvector). Endpoint interno -- no exponer a internet."
    ),
    responses={
        503: {"description": "Capacidad maxima alcanzada (backpressure)"},
    },
)
async def enroll(
    imagen: UploadFile = File(..., description="Foto para generar embedding"),
):
    """Enrolamiento facial: devuelve embedding de 512 floats."""
    if _semaphore.locked():
        FACE_CONCURRENCY_REJECTED.labels(endpoint="enroll").inc()
        return _error_response(503, "Capacidad maxima alcanzada. Intente de nuevo.")

    async with _semaphore:
        t_start = time.perf_counter()

        contenido = await imagen.read()
        img = _leer_imagen(contenido)

        resultado, _, tiempos = await run_in_threadpool(
            procesar_verificacion, img, None  # None = modo enrolamiento
        )

        total_ms = round((time.perf_counter() - t_start) * 1000, 2)

        # Fase 4: metricas
        _observe_pipeline_metrics(tiempos)
        FACE_REQUESTS.labels(endpoint="enroll", result="ok").inc()
        for issue in resultado.get("qualityIssues", []):
            reason = issue.split(" ")[0] if issue else "unknown"
            FACE_QUALITY_ISSUES.labels(reason=reason).inc()

        logger.info(
            "enroll completado",
            extra={
                "endpoint": "enroll",
                "processing_time_ms": total_ms,
                "result": "ok",
            },
        )

        return EnrollResponse(
            success=True,
            data=EnrollResponseData(
                **{k: v for k, v in resultado.items()
                   if k not in ("similarity", "verified")}
            ),
            meta=VerifyResponseMeta(
                model=settings.MODEL_NAME,
                detector=settings.DETECTOR_BACKEND,
                processingTimeMs=total_ms,
            ),
        )
