"""
Orquestación del pipeline completo: validar resolución (sobre la imagen
ORIGINAL) -> redimensionar (optimización interna) -> calidad ->
detección+liveness (1 sola pasada) -> embedding (reutilizando el
rostro ya detectado, sin re-detectar).
"""

import time
import cv2
import numpy as np
from deepface import DeepFace

from app.core.config import (
    MODEL_NAME,
    DETECTOR_BACKEND,
    SIMILARITY_THRESHOLD,
    ANCHO_MIN,
    ALTO_MIN,
)
from app.services.quality import validar_calidad_imagen, validar_calidad_rostro


def redimensionar_si_es_necesario(img: np.ndarray, lado_max: int = 800) -> np.ndarray:
    """
    Reduce la imagen si su lado mayor excede lado_max, manteniendo aspect
    ratio. No afecta precisión del modelo — RetinaFace/ArcFace ya reescalan
    internamente a un tamaño fijo antes de procesar. Esto solo reduce el
    costo computacional de la búsqueda de rostro en la imagen de entrada.
    """
    alto, ancho = img.shape[:2]
    lado_mayor = max(alto, ancho)

    if lado_mayor <= lado_max:
        return img  # ya es suficientemente pequeña, no tocar

    escala = lado_max / lado_mayor
    nuevo_ancho = int(ancho * escala)
    nuevo_alto = int(alto * escala)
    return cv2.resize(img, (nuevo_ancho, nuevo_alto), interpolation=cv2.INTER_AREA)


def cosine_similarity(emb1: list, emb2: list) -> float:
    a, b = np.array(emb1), np.array(emb2)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def procesar_verificacion(img: np.ndarray, embedding_registrado: list | None):
    """
    Pipeline completo en UNA sola imagen de entrada.
    Si embedding_registrado es None, solo hace detección+calidad+liveness+
    genera embedding (caso de ENROLAMIENTO, no de verificación).
    """
    start = time.perf_counter()
    tiempos = {}

    alto_original, ancho_original = img.shape[:2]
    if ancho_original < ANCHO_MIN or alto_original < ALTO_MIN:
        return _respuesta_temprana(
            [f"resolucion_insuficiente ({ancho_original}x{alto_original})"],
            start,
        )

    t0 = time.perf_counter()
    img = redimensionar_si_es_necesario(img, lado_max=800)
    tiempos["resize"] = round((time.perf_counter() - t0) * 1000, 2)

    t0 = time.perf_counter()
    calidad_img = validar_calidad_imagen(img)
    tiempos["calidad_imagen"] = round((time.perf_counter() - t0) * 1000, 2)
    if not calidad_img.aceptable:
        return _respuesta_temprana(calidad_img.razones, start)

    # --- Detección + liveness en UNA sola pasada (evita detectar 2 veces) ---
    t0 = time.perf_counter()
    try:
        faces = DeepFace.extract_faces(
            img_path=img,
            detector_backend=DETECTOR_BACKEND,
            anti_spoofing=True,
            enforce_detection=True,
        )
    except ValueError:
        return _respuesta_sin_rostro(start)
    tiempos["deteccion_y_liveness"] = round((time.perf_counter() - t0) * 1000, 2)

    t0 = time.perf_counter()
    calidad_rostro = validar_calidad_rostro(img, faces)
    tiempos["calidad_rostro"] = round((time.perf_counter() - t0) * 1000, 2)
    if not calidad_rostro.aceptable:
        return _respuesta_calidad_rostro(calidad_rostro, faces, start)

    rostro = faces[0]
    is_live = bool(rostro["is_real"])
    liveness_confidence = round(float(rostro["antispoof_score"]), 4)

    if not is_live:
        return _respuesta_liveness_fallido(is_live, liveness_confidence, start)

    t0 = time.perf_counter()
    face_crop = rostro["face"]
    rep = DeepFace.represent(
        img_path=face_crop,
        model_name=MODEL_NAME,
        detector_backend="skip",
        enforce_detection=False,
    )
    embedding_nuevo = rep[0]["embedding"]
    tiempos["embedding"] = round((time.perf_counter() - t0) * 1000, 2)

    resultado = {
        "faceDetected": True,
        "faceCount": len(faces),
        "qualityOk": True,
        "qualityIssues": [],
        "isLive": is_live,
        "livenessConfidence": liveness_confidence,
        "similarity": None,
        "verified": None,
    }

    if embedding_registrado is not None:
        similarity = cosine_similarity(embedding_nuevo, embedding_registrado)
        resultado["similarity"] = round(similarity, 4)
        resultado["verified"] = similarity >= SIMILARITY_THRESHOLD
    else:
        resultado["embedding"] = embedding_nuevo

    elapsed_ms = (time.perf_counter() - start) * 1000
    return resultado, round(elapsed_ms, 2)


def _respuesta_temprana(razones: list, start):
    elapsed_ms = (time.perf_counter() - start) * 1000
    return {
        "faceDetected": False,
        "faceCount": 0,
        "qualityOk": False,
        "qualityIssues": razones,
        "isLive": None,
        "livenessConfidence": None,
        "similarity": None,
        "verified": None,
    }, round(elapsed_ms, 2)


def _respuesta_sin_rostro(start):
    elapsed_ms = (time.perf_counter() - start) * 1000
    return {
        "faceDetected": False,
        "faceCount": 0,
        "qualityOk": True,
        "qualityIssues": ["ningun_rostro_detectado"],
        "isLive": None,
        "livenessConfidence": None,
        "similarity": None,
        "verified": None,
    }, round(elapsed_ms, 2)


def _respuesta_calidad_rostro(calidad_rostro, faces, start):
    elapsed_ms = (time.perf_counter() - start) * 1000
    return {
        "faceDetected": True,
        "faceCount": len(faces),
        "qualityOk": False,
        "qualityIssues": calidad_rostro.razones,
        "isLive": None,
        "livenessConfidence": None,
        "similarity": None,
        "verified": None,
    }, round(elapsed_ms, 2)


def _respuesta_liveness_fallido(is_live, confidence, start):
    elapsed_ms = (time.perf_counter() - start) * 1000
    return {
        "faceDetected": True,
        "faceCount": 1,
        "qualityOk": True,
        "qualityIssues": [],
        "isLive": is_live,
        "livenessConfidence": confidence,
        "similarity": None,
        "verified": None,
    }, round(elapsed_ms, 2)