"""
Orquestacion del pipeline completo: validar resolucion (sobre la imagen
ORIGINAL) -> redimensionar (optimizacion interna) -> calidad ->
deteccion+liveness (1 sola pasada) -> embedding (reutilizando el
rostro ya detectado, sin re-detectar).

Fase 2: config via settings. Fase 4: retorna tiempos por etapa.
"""
import threading
import time
import cv2
import numpy as np
from deepface import DeepFace

from app.core.config import settings
from app.services.quality import validar_calidad_imagen, validar_calidad_rostro

# ---------------------------------------------------------------------------
# Los modelos de DeepFace (YuNet/MiniFAS/ArcFace) NO son thread-safe para
# inferencia concurrente (race condition en cv::dnn::Net::Impl::forwardGraph).
# Este lock serializa el acceso a los modelos -> es el verdadero cuello de
# botella (~250ms/request). El escalado real se logra con procesos
# (uvicorn --workers N), no con mas threads.
# ---------------------------------------------------------------------------
INFERENCE_LOCK = threading.Lock()


def redimensionar_si_es_necesario(img: np.ndarray, lado_max: int = 800) -> np.ndarray:
    """Reduce la imagen si su lado mayor excede lado_max, manteniendo aspect ratio."""
    alto, ancho = img.shape[:2]
    lado_mayor = max(alto, ancho)
    if lado_mayor <= lado_max:
        return img
    escala = lado_max / lado_mayor
    return cv2.resize(
        img,
        (int(ancho * escala), int(alto * escala)),
        interpolation=cv2.INTER_AREA,
    )


def cosine_similarity(emb1: list, emb2: list) -> float:
    a, b = np.array(emb1), np.array(emb2)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def procesar_verificacion(
    img: np.ndarray,
    embedding_registrado: list | None,
):
    """
    Pipeline completo en UNA sola imagen de entrada.

    Retorna: (resultado_dict, elapsed_ms, tiempos_dict)

    Si embedding_registrado es None, solo hace deteccion+calidad+liveness+
    genera embedding (caso de ENROLAMIENTO, no de verificacion).
    """
    start = time.perf_counter()
    tiempos: dict[str, float] = {}

    # --- Capa 0: Resolucion minima ---
    alto_original, ancho_original = img.shape[:2]
    if ancho_original < settings.ANCHO_MIN or alto_original < settings.ALTO_MIN:
        return _respuesta_temprana(
            [f"resolucion_insuficiente ({ancho_original}x{alto_original})"],
            start,
            tiempos,
        )

    # --- Capa 1a: Redimensionado ---
    t0 = time.perf_counter()
    img = redimensionar_si_es_necesario(img, lado_max=800)
    tiempos["resize"] = round((time.perf_counter() - t0) * 1000, 2)

    # --- Capa 1b: Calidad de imagen ---
    t0 = time.perf_counter()
    calidad_img = validar_calidad_imagen(img)
    tiempos["calidad_imagen"] = round((time.perf_counter() - t0) * 1000, 2)
    if not calidad_img.aceptable:
        return _respuesta_temprana(calidad_img.razones, start, tiempos)

    # --- Capa 2a: Deteccion + liveness (1 sola pasada) ---
    t0 = time.perf_counter()
    try:
        with INFERENCE_LOCK:
            faces = DeepFace.extract_faces(
                img_path=img,
                detector_backend=settings.DETECTOR_BACKEND,
                anti_spoofing=True,
                enforce_detection=True,
            )
    except ValueError:
        return _respuesta_sin_rostro(start, tiempos)
    tiempos["deteccion_y_liveness"] = round((time.perf_counter() - t0) * 1000, 2)

    # --- Capa 2b: Calidad de rostro ---
    t0 = time.perf_counter()
    calidad_rostro = validar_calidad_rostro(img, faces)
    tiempos["calidad_rostro"] = round((time.perf_counter() - t0) * 1000, 2)
    if not calidad_rostro.aceptable:
        return _respuesta_calidad_rostro(calidad_rostro, faces, start, tiempos)

    # --- Capa 3: Liveness ---
    rostro = faces[0]
    is_live = bool(rostro["is_real"])
    liveness_confidence = round(float(rostro["antispoof_score"]), 4)
    if not is_live:
        return _respuesta_liveness_fallido(is_live, liveness_confidence, start, tiempos)

    # --- Capa 4: Embedding (sin re-detectar) ---
    t0 = time.perf_counter()
    face_crop = rostro["face"]
    with INFERENCE_LOCK:
        rep = DeepFace.represent(
            img_path=face_crop,
            model_name=settings.MODEL_NAME,
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

    # --- Capa 5: Comparacion (solo si hay embedding registrado) ---
    if embedding_registrado is not None:
        similarity = cosine_similarity(embedding_nuevo, embedding_registrado)
        resultado["similarity"] = round(similarity, 4)
        resultado["verified"] = similarity >= settings.SIMILARITY_THRESHOLD
    else:
        resultado["embedding"] = embedding_nuevo

    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
    return resultado, elapsed_ms, tiempos


# ---------------------------------------------------------------------------
# Respuestas tempranas (todos con verified: None)
# ---------------------------------------------------------------------------

def _respuesta_temprana(razones: list, start, tiempos: dict):
    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
    return {
        "faceDetected": False, "faceCount": 0, "qualityOk": False,
        "qualityIssues": razones, "isLive": None, "livenessConfidence": None,
        "similarity": None, "verified": None,
    }, elapsed_ms, tiempos


def _respuesta_sin_rostro(start, tiempos: dict):
    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
    return {
        "faceDetected": False, "faceCount": 0, "qualityOk": True,
        "qualityIssues": ["ningun_rostro_detectado"], "isLive": None,
        "livenessConfidence": None, "similarity": None, "verified": None,
    }, elapsed_ms, tiempos


def _respuesta_calidad_rostro(calidad_rostro, faces, start, tiempos: dict):
    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
    return {
        "faceDetected": True, "faceCount": len(faces), "qualityOk": False,
        "qualityIssues": calidad_rostro.razones, "isLive": None,
        "livenessConfidence": None, "similarity": None, "verified": None,
    }, elapsed_ms, tiempos


def _respuesta_liveness_fallido(is_live, confidence, start, tiempos: dict):
    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
    return {
        "faceDetected": True, "faceCount": 1, "qualityOk": True,
        "qualityIssues": [], "isLive": is_live, "livenessConfidence": confidence,
        "similarity": None, "verified": None,
    }, elapsed_ms, tiempos
