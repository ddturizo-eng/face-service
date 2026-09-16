"""
Averyn - Validacion de calidad de captura.
"""
from dataclasses import dataclass, field

import cv2
import numpy as np

from app.core.config import settings


@dataclass
class ResultadoCalidad:
    aceptable: bool
    razones: list = field(default_factory=list)
    metricas: dict = field(default_factory=dict)


def validar_calidad_imagen(img: np.ndarray) -> ResultadoCalidad:
    """Valida brillo y nitidez de la imagen YA CARGADA (np.ndarray)."""
    razones = []
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    brillo = float(np.mean(gris))
    nitidez = float(cv2.Laplacian(gris, cv2.CV_64F).var())

    if brillo < settings.BRILLO_MIN:
        razones.append(f"imagen_muy_oscura (brillo={brillo:.1f})")
    elif brillo > settings.BRILLO_MAX:
        razones.append(f"imagen_sobreexpuesta (brillo={brillo:.1f})")
    if nitidez < settings.NITIDEZ_MIN:
        razones.append(f"imagen_borrosa (nitidez={nitidez:.1f})")

    return ResultadoCalidad(
        aceptable=len(razones) == 0,
        razones=razones,
        metricas={"brillo": round(brillo, 1), "nitidez": round(nitidez, 1)},
    )


def validar_calidad_rostro(img: np.ndarray, face_data: list) -> ResultadoCalidad:
    """Evalua calidad del rostro ya detectado (no detecta de nuevo)."""
    razones = []

    if len(face_data) == 0:
        return ResultadoCalidad(aceptable=False, razones=["ningun_rostro_detectado"])
    if len(face_data) > 1:
        razones.append(f"multiples_rostros_detectados ({len(face_data)})")

    alto_img, ancho_img = img.shape[:2]
    area_imagen = alto_img * ancho_img

    area_facial = face_data[0]["facial_area"]
    area_rostro = area_facial["w"] * area_facial["h"]
    ratio = area_rostro / area_imagen

    if ratio < settings.ROSTRO_AREA_MIN_RATIO:
        razones.append(f"rostro_muy_pequeno (ratio={ratio:.3f})")
    if area_facial["w"] < settings.ROSTRO_ANCHO_MIN_PX:
        razones.append(f"rostro_muy_pequeno_en_px (ancho={area_facial['w']}px)")

    return ResultadoCalidad(
        aceptable=len(razones) == 0,
        razones=razones,
        metricas={
            "cantidad_rostros": len(face_data),
            "ratio_area_rostro": round(ratio, 3),
            "ancho_rostro_px": area_facial["w"],
        },
    )
