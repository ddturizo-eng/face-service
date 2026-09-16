"""
ABIS Web - Validación de calidad de captura.
Ver docstring original en el POC (Módulo 2) para el detalle de cada check.
"""

from dataclasses import dataclass, field
import cv2
import numpy as np

from app.core.config import (
    BRILLO_MIN,
    BRILLO_MAX,
    NITIDEZ_MIN,
    ROSTRO_AREA_MIN_RATIO,
    ROSTRO_ANCHO_MIN_PX,
)


@dataclass
class ResultadoCalidad:
    aceptable: bool
    razones: list = field(default_factory=list)
    metricas: dict = field(default_factory=dict)


def validar_calidad_imagen(img: np.ndarray) -> ResultadoCalidad:
    """Recibe la imagen YA CARGADA (np.ndarray), no una ruta de archivo.
    Cambio respecto al POC: en el servicio evitamos leer el archivo dos
    veces (una para calidad, otra para detección).
    El chequeo de resolución mínima NO va aquí: se hace por separado en
    pipeline.py sobre la imagen ORIGINAL, antes del redimensionado."""
    razones = []
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    brillo = float(np.mean(gris))
    nitidez = float(cv2.Laplacian(gris, cv2.CV_64F).var())

    if brillo < BRILLO_MIN:
        razones.append(f"imagen_muy_oscura (brillo={brillo:.1f})")
    elif brillo > BRILLO_MAX:
        razones.append(f"imagen_sobreexpuesta (brillo={brillo:.1f})")
    if nitidez < NITIDEZ_MIN:
        razones.append(f"imagen_borrosa (nitidez={nitidez:.1f})")

    return ResultadoCalidad(
        aceptable=len(razones) == 0,
        razones=razones,
        metricas={"brillo": round(brillo, 1), "nitidez": round(nitidez, 1)},
    )


def validar_calidad_rostro(img: np.ndarray, face_data: list) -> ResultadoCalidad:
    """face_data viene ya calculado por la detección hecha en pipeline.py —
    aquí NO se vuelve a detectar nada, solo se evalúan los datos."""
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

    if ratio < ROSTRO_AREA_MIN_RATIO:
        razones.append(f"rostro_muy_pequeno (ratio={ratio:.3f})")
    if area_facial["w"] < ROSTRO_ANCHO_MIN_PX:
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
