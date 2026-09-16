"""
ABIS Web - Face Service
Microservicio FastAPI para detección, calidad, liveness y verificación facial.
"""

from contextlib import asynccontextmanager
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, Form
from deepface import DeepFace

from app.core.config import MODEL_NAME, DETECTOR_BACKEND
from app.services.pipeline import procesar_verificacion
from app.schemas.face import VerifyResponse, VerifyResponseData, VerifyResponseMeta


@asynccontextmanager
async def lifespan(app: FastAPI):
    # "Warm-up": corremos una inferencia dummy al arrancar para forzar
    # la carga de TODOS los modelos (ArcFace, RetinaFace, MiniFAS) en
    # memoria ANTES de recibir el primer request real. Sin esto, el
    # PRIMER usuario que llegue pagaría el costo de carga (~10-15s extra).
    dummy = np.zeros((300, 300, 3), dtype=np.uint8)
    try:
        DeepFace.extract_faces(img_path=dummy, detector_backend=DETECTOR_BACKEND,
                                anti_spoofing=True, enforce_detection=False)
        DeepFace.represent(img_path=dummy, model_name=MODEL_NAME,
                            detector_backend="skip", enforce_detection=False)
    except Exception:
        pass  # el dummy no tiene rostro real, es normal que falle silenciosamente
    print("Modelos precargados. Face Service listo.")
    yield


app = FastAPI(title="ABIS Face Service", lifespan=lifespan)


def leer_imagen(file_bytes: bytes) -> np.ndarray:
    array = np.frombuffer(file_bytes, dtype=np.uint8)
    return cv2.imdecode(array, cv2.IMREAD_COLOR)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/api/v1/face/verify", response_model=VerifyResponse)
async def verify(
    imagen: UploadFile = File(...),
    embedding_registrado: str = Form(None),  # JSON string de una lista de floats
):
    import json
    import time

    contenido = await imagen.read()
    img = leer_imagen(contenido)

    emb_registrado = json.loads(embedding_registrado) if embedding_registrado else None

    resultado, tiempo_ms = procesar_verificacion(img, emb_registrado)

    return VerifyResponse(
        success=True,
        data=VerifyResponseData(**{k: v for k, v in resultado.items() if k != "embedding"}),
        meta=VerifyResponseMeta(
            model=MODEL_NAME, detector=DETECTOR_BACKEND, processingTimeMs=tiempo_ms,
        ),
    )