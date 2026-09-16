"""
Fase 0: Load test para Averyn Face Service.
Uso:
    1. Arrancar el servicio: uvicorn app.main:app --port 8001
    2. Ejecutar: locust -f locustfile.py --host http://127.0.0.1:8001
    3. Abrir http://127.0.0.1:8089 para configurar VUs y lanzar pruebas.

Escenarios recomendados:
    - 1 VU, 50 requests  --> baseline单线程
    - 10 VUs, 200 requests --> concurrencia moderada
    - 50 VUs, 500 requests --> punto de saturacion estimado
    - 100 VUs, 1000 requests --> stress test

Medir: throughput (req/s), P50/P95/P99 latencia, tasa de errores,
       uso de RAM/CPU via `docker stats` en paralelo.
"""
import os
import json
from pathlib import Path

from locust import HttpUser, task, between, events


IMAGES_DIR = Path(__file__).parent / "test_images"
IMAGE_FILES: list[bytes] = []


def _load_test_images():
    """Carga imagenes de prueba al inicio del test."""
    global IMAGE_FILES
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        for p in IMAGES_DIR.glob(ext):
            IMAGE_FILES.append(p.read_bytes())
    if not IMAGE_FILES:
        # Fallback: crear una imagen dummy de 400x400 con un rectangulo
        # (no tendra rostro real, pero permite medir latencia sin error de imagen)
        import numpy as np
        import cv2
        dummy = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
        # Dibujar un rectangulo claro para simular una zona de "rostro"
        cv2.rectangle(dummy, (100, 100), (300, 300), (200, 200, 200), -1)
        _, buf = cv2.imencode(".jpg", dummy)
        IMAGE_FILES.append(buf.tobytes())
        print("[locustfile] No se encontraron imagenes reales, usando imagen dummy.")


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    _load_test_images()
    print(f"[locustfile] {len(IMAGE_FILES)} imagenes cargadas para el test.")


class FaceServiceUser(HttpUser):
    """Simula un usuario enviando fotos al face service."""

    wait_time = between(0.1, 0.3)  # tiempo entre requests por usuario (pensamiento corto)

    def on_start(self):
        self._img_idx = 0

    def _next_image(self) -> bytes:
        img = IMAGE_FILES[self._img_idx % len(IMAGE_FILES)]
        self._img_idx += 1
        return img

    @task(10)
    def verify_without_embedding(self):
        """POST /api/v1/face/verify sin embedding_registrado (modo calidad+liveness)."""
        img_data = self._next_image()
        with self.client.post(
            "/api/v1/face/verify",
            files={"imagen": ("test.jpg", img_data, "image/jpeg")},
            name="/api/v1/face/verify [sin embedding]",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                body = response.json()
                if body.get("success"):
                    response.success()
                else:
                    response.failure(f"success=false: {body.get('error')}")
            elif response.status_code == 503:
                response.success()  # 503 es comportamiento esperado bajo carga
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(5)
    def verify_with_embedding(self):
        """POST /api/v1/face/verify con embedding ficticio (simula verificacion)."""
        img_data = self._next_image()
        fake_embedding = json.dumps([0.001] * 512)
        with self.client.post(
            "/api/v1/face/verify",
            files={"imagen": ("test.jpg", img_data, "image/jpeg")},
            data={"embedding_registrado": fake_embedding},
            name="/api/v1/face/verify [con embedding]",
            catch_response=True,
        ) as response:
            if response.status_code in (200, 503):
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(3)
    def enroll(self):
        """POST /api/v1/face/enroll (enrolamiento)."""
        img_data = self._next_image()
        with self.client.post(
            "/api/v1/face/enroll",
            files={"imagen": ("test.jpg", img_data, "image/jpeg")},
            name="/api/v1/face/enroll",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                body = response.json()
                if body.get("success") and body.get("data", {}).get("embedding"):
                    response.success()
                else:
                    response.failure(f"Sin embedding: {body}")
            elif response.status_code == 503:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(2)
    def health(self):
        """GET /health (baseline, sin modelos)."""
        self.client.get("/health", name="/health")

    @task(1)
    def detailed_health(self):
        """GET /api/v1/face/health (con verificacion de modelos)."""
        self.client.get("/api/v1/face/health", name="/api/v1/face/health")
