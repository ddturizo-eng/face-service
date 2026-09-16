"""
Módulo 4b - Comparación empírica de detectores (RetinaFace vs YuNet)
------------------------------------------------------------------------
Objetivo: determinar si el cambio de detector (por rendimiento) afecta
la decisión final de verificación (verified) o la similitud calculada,
usando el mismo pipeline real, no solo literatura académica.

Genera N pares aleatorios de LFW (mezcla de misma persona / personas
distintas), corre cada par con AMBOS detectores, y compara resultados.
"""

import os
import random
import time
import numpy as np
from deepface import DeepFace

LFW_ROOT = "test_images/lfw_funneled"
MODEL_NAME = "ArcFace"
SIMILARITY_THRESHOLD = 0.68
DETECTORES = ["retinaface", "yunet"]
CANTIDAD_PARES = 8  # ajustable; más pares = comparación más robusta


def cosine_similarity(emb1, emb2):
    a, b = np.array(emb1), np.array(emb2)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def generar_pares(lfw_root: str, cantidad: int):
    """Genera una mezcla de pares 'misma persona' y 'personas distintas'."""
    personas = [
        p for p in os.listdir(lfw_root) if os.path.isdir(os.path.join(lfw_root, p))
    ]

    con_dos_fotos = []
    for p in personas:
        fotos = sorted(
            f
            for f in os.listdir(os.path.join(lfw_root, p))
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        if len(fotos) >= 2:
            con_dos_fotos.append((p, fotos))

    pares = []
    mitad = cantidad // 2

    # Mitad "misma persona"
    for persona, fotos in random.sample(con_dos_fotos, min(mitad, len(con_dos_fotos))):
        img1 = os.path.join(lfw_root, persona, fotos[0])
        img2 = os.path.join(lfw_root, persona, fotos[1])
        pares.append((img1, img2, "misma_persona"))

    # Mitad "personas distintas"
    for _ in range(cantidad - len(pares)):
        p1, p2 = random.sample(personas, 2)
        fotos1 = [
            f
            for f in os.listdir(os.path.join(lfw_root, p1))
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        fotos2 = [
            f
            for f in os.listdir(os.path.join(lfw_root, p2))
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        img1 = os.path.join(lfw_root, p1, fotos1[0])
        img2 = os.path.join(lfw_root, p2, fotos2[0])
        pares.append((img1, img2, "personas_distintas"))

    random.shuffle(pares)
    return pares


def generar_embedding(image_path: str, detector: str):
    rep = DeepFace.represent(
        img_path=image_path,
        model_name=MODEL_NAME,
        detector_backend=detector,
        enforce_detection=True,
    )
    return rep[0]["embedding"]


def evaluar_par(img1: str, img2: str, detector: str):
    start = time.perf_counter()
    try:
        emb1 = generar_embedding(img1, detector)
        emb2 = generar_embedding(img2, detector)
        similarity = cosine_similarity(emb1, emb2)
        verified = similarity >= SIMILARITY_THRESHOLD
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {
            "similarity": round(similarity, 4),
            "verified": verified,
            "tiempo_ms": round(elapsed_ms, 2),
            "error": None,
        }
    except ValueError as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {
            "similarity": None,
            "verified": None,
            "tiempo_ms": round(elapsed_ms, 2),
            "error": str(e),
        }


if __name__ == "__main__":
    random.seed(42)  # reproducible: mismos pares en cada corrida
    pares = generar_pares(LFW_ROOT, CANTIDAD_PARES)

    print("=" * 90)
    print(f"Comparación RetinaFace vs YuNet sobre {len(pares)} pares de LFW")
    print("=" * 90)

    coincidencias = 0
    discrepancias = []
    tiempos_por_detector = {d: [] for d in DETECTORES}

    for i, (img1, img2, tipo_esperado) in enumerate(pares, 1):
        print(
            f"\nPar {i} ({tipo_esperado}): {os.path.basename(img1)} vs {os.path.basename(img2)}"
        )

        resultados = {}
        for detector in DETECTORES:
            r = evaluar_par(img1, img2, detector)
            resultados[detector] = r
            if r["error"] is None:
                tiempos_por_detector[detector].append(r["tiempo_ms"])
            print(
                f"  [{detector:12s}] verified={r['verified']} | "
                f"similarity={r['similarity']} | tiempo={r['tiempo_ms']}ms | "
                f"error={r['error']}"
            )

        v_retina = resultados["retinaface"]["verified"]
        v_yunet = resultados["yunet"]["verified"]

        if v_retina == v_yunet:
            coincidencias += 1
        else:
            discrepancias.append((i, img1, img2, v_retina, v_yunet))

    print("\n" + "=" * 90)
    print("RESUMEN")
    print("=" * 90)
    print(f"Pares evaluados: {len(pares)}")
    print(
        f"Coincidencias en la decisión final (verified): {coincidencias}/{len(pares)}"
    )

    if discrepancias:
        print(f"\n⚠️  DISCREPANCIAS ENCONTRADAS ({len(discrepancias)}):")
        for idx, img1, img2, vr, vy in discrepancias:
            print(f"  Par {idx}: RetinaFace={vr} | YuNet={vy} -> {img1} vs {img2}")
    else:
        print(
            "\n✅ Sin discrepancias: ambos detectores coincidieron en TODAS las decisiones."
        )

    for detector in DETECTORES:
        tiempos = tiempos_por_detector[detector]
        if tiempos:
            promedio = sum(tiempos) / len(tiempos)
            print(
                f"\nTiempo promedio por par [{detector}]: {round(promedio, 2)}ms "
                f"(min={min(tiempos)}ms, max={max(tiempos)}ms)"
            )
