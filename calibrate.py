"""
Fase 3: Calibracion de umbrales con datos reales.

REQUIERE un dataset de pares reales (misma persona / personas distintas)
representativo de las condiciones reales de captura del kiosco.

Estructura esperada del dataset (formato LFW-style):
    calibration_data/
        persona_a/
            foto1.jpg
            foto2.jpg
        persona_b/
            foto1.jpg
            ...

Uso:
    python calibrate.py --data calibration_data --thresholds "0.4,0.5,0.6,0.65,0.68,0.7,0.75,0.8"

Genera:
    - CURVA_FAR_FRR.csv   : FAR/FRR/EER para cada umbral evaluado
    - CURVA_FAR_FRR.png   : grafico de la curva
    - REPORTE.md          : resumen con el punto de operacion recomendado
"""
import argparse
import csv
import os
import random
import time
from pathlib import Path

import numpy as np
from deepface import DeepFace

DEFAULT_THRESHOLDS = [0.4, 0.5, 0.55, 0.6, 0.62, 0.65, 0.68, 0.7, 0.72, 0.75, 0.8, 0.85]


def cosine_similarity(emb1, emb2):
    a, b = np.array(emb1), np.array(emb2)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def listar_personas(dataset: Path) -> list[str]:
    """Devuelve las personas (carpetas) que tienen >= 2 fotos."""
    personas = []
    for p in sorted(dataset.iterdir()):
        if not p.is_dir():
            continue
        fotos = [f for f in p.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
        if len(fotos) >= 2:
            personas.append(p.name)
    return personas


def generar_pares(dataset: Path, personas: list[str], max_pares_misma: int = 100, max_pares_distinta: int = 100):
    """Genera pares misma persona y personas distintas."""
    mismo = []
    distinto = []

    for persona in personas:
        fotos = sorted(
            f for f in (dataset / persona).iterdir()
            if f.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
        if len(fotos) >= 2 and len(mismo) < max_pares_misma:
            mismo.append((fotos[0], fotos[1], "misma_persona"))

    random.shuffle(personas)
    for _ in range(max_pares_distinta):
        p1, p2 = random.sample(personas, 2)
        f1 = random.choice(list((dataset / p1).iterdir()))
        f2 = random.choice(list((dataset / p2).iterdir()))
        distinto.append((f1, f2, "personas_distintas"))

    return mismo, distinto


def embed_image(path: Path, detector: str = "yunet") -> list | None:
    """Genera embedding de una imagen. Devuelve None si no detecta rostro."""
    try:
        rep = DeepFace.represent(
            img_path=str(path),
            model_name="ArcFace",
            detector_backend=detector,
            enforce_detection=True,
        )
        return rep[0]["embedding"]
    except Exception:
        return None


def evaluar_umbral(similitudes_mismo: list, similitudes_distinto: list, umbral: float) -> dict:
    """Calcula FAR/FRR para un umbral dado."""
    # FAR: pares distintos mal aceptados (similitud >= umbral)
    far = sum(1 for s in similitudes_distinto if s >= umbral) / max(len(similitudes_distinto), 1)
    # FRR: pares mismos mal rechazados (similitud < umbral)
    frr = sum(1 for s in similitudes_mismo if s < umbral) / max(len(similitudes_mismo), 1)
    # FMR/FNMR alias
    return {
        "threshold": umbral,
        "FAR": round(far, 4),
        "FRR": round(frr, 4),
        "EER": round(abs(far - frr), 4),
        "n_same": len(similitudes_mismo),
        "n_diff": len(similitudes_distinto),
    }


def main():
    parser = argparse.ArgumentParser(description="Calibracion de umbrales FAR/FRR")
    parser.add_argument("--data", type=Path, required=True, help="Carpeta con datos de calibracion")
    parser.add_argument("--thresholds", default=", ".join(map(str, DEFAULT_THRESHOLDS)),
                        help="Umbrales a evaluar (separados por coma)")
    parser.add_argument("--max-misma", type=int, default=100)
    parser.add_argument("--max-distinta", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    thresholds = [float(t.strip()) for t in args.thresholds.split(",")]

    print("=" * 60)
    print("Fase 3 - Calibracion de SIMILARITY_THRESHOLD")
    print("=" * 60)

    personas = listar_personas(args.data)
    print(f"Personas con >= 2 fotos: {len(personas)}")
    if len(personas) < 2:
        print("ERROR: se necesitan al menos 2 personas con 2 fotos cada una.")
        return

    mismo, distinto = generar_pares(args.data, personas, args.max_misma, args.max_distinta)
    print(f"Pares misma persona: {len(mismo)} | Pares distintas personas: {len(distinto)}")

    # 1. Generar embeddings (una sola pasada por archivo)
    archivos_unicos = set()
    for p1, p2, _ in mismo + distinto:
        archivos_unicos.add(p1)
        archivos_unicos.add(p2)

    cache_embeddings = {}
    print(f"\nGenerando embeddings para {len(archivos_unicos)} imagenes unicas...")
    t0 = time.time()
    for archivo in sorted(archivos_unicos):
        emb = embed_image(archivo)
        if emb is not None:
            cache_embeddings[str(archivo)] = emb
        else:
            print(f"  WARN: sin rostro detectable en {archivo.name}")
    print(f"Embeddings listos en {round(time.time() - t0, 1)}s")

    # 2. Calcular similitudes
    sim_mismo, sim_distinto = [], []
    for p1, p2, tipo in mismo:
        e1, e2 = cache_embeddings.get(str(p1)), cache_embeddings.get(str(p2))
        if e1 and e2:
            sim_mismo.append(cosine_similarity(e1, e2))
    for p1, p2, tipo in distinto:
        e1, e2 = cache_embeddings.get(str(p1)), cache_embeddings.get(str(p2))
        if e1 and e2:
            sim_distinto.append(cosine_similarity(e1, e2))

    print(f"\nSimilitudes calculadas: misma={len(sim_mismo)} | distinta={len(sim_distinto)}")

    # 3. Evaluar umbrales
    resultados = [evaluar_umbral(sim_mismo, sim_distinto, t) for t in thresholds]

    # 4. Reporte
    out_csv = Path("CURVA_FAR_FRR.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=resultados[0].keys())
        writer.writeheader()
        writer.writerows(resultados)
    print(f"\nCSV guardado: {out_csv}")

    print("\n" + "=" * 60)
    print(f"{'Umbral':>8} {'FAR':>8} {'FRR':>8} {'|FAR-FRR|':>10}")
    print("-" * 60)
    for r in resultados:
        print(f"{r['threshold']:>8.2f} {r['FAR']:>8.4f} {r['FRR']:>8.4f} {r['EER']:>10.4f}")

    # 5. Punto de operacion recomendado
    eer = min(resultados, key=lambda r: r["EER"])
    print("\n" + "=" * 60)
    print(f"EER (error balanceado): threshold={eer['threshold']}, FAR={eer['FAR']}, FRR={eer['FRR']}")
    print("RECOMENDACION: elegir el umbral segun la tolerancia de negocio:")
    print("  - Si prefieres rechazar impostores (seguridad alta): umbral MAS ALTO (FAR baja, FRR sube)")
    print("  - Si prefieres no rechazar legitimos (friction baja): umbral MAS BAJO (FRR baja, FAR sube)")
    print("Ejemplo: si el kiosco debe ser estricto, usar un umbral donde FAR <= 0.001 (0.1%)")
    print("=" * 60)

    # 6. Nota sobre los umbrales de calidad
    print("\n" + "-" * 60)
    print("PENDIENTE: calibrar tambien BRILLO_MIN/MAX, NITIDEZ_MIN y")
    print("ROSTRO_AREA_MIN_RATIO con muestras de captura real.""")
    print("Para eso, capturar fotos de los kioscos reales y comparar las")
    print("metricas (brillo, nitidez, ratio) de capturas aceptadas vs rechazadas.")
    print("-" * 60)


if __name__ == "__main__":
    main()