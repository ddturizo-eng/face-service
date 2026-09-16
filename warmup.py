"""
Script de warm-up para Docker build.
Fuerza la descarga de los modelos (YuNet, ArcFace, MiniFAS) durante la
construcción de la imagen, para que el contenedor arranque sin depender
de internet ni de un primer request lento.
"""
import numpy as np
from deepface import DeepFace

dummy = np.zeros((300, 300, 3), dtype=np.uint8)

try:
    DeepFace.extract_faces(
        img_path=dummy,
        detector_backend="yunet",
        anti_spoofing=True,
        enforce_detection=False,
    )
    DeepFace.represent(
        img_path=dummy,
        model_name="ArcFace",
        detector_backend="skip",
        enforce_detection=False,
    )
except Exception:
    pass  # el dummy no tiene rostro real, es esperado que falle silenciosamente

print("Modelos descargados y cacheados en la imagen.")