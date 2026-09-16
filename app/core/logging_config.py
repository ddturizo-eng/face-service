"""
Fase 4: Configuracion de logging estructurado (JSON).
"""
import json
import logging
import sys
from datetime import datetime, timezone


class JSONFormatter(logging.Formatter):
    """Formatter que produce una linea JSON por log record."""

    def format(self, record: logging.LogRecord) -> str:
        log_record: dict = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        # Adjuntar campos extra si existen
        for key in ("request_id", "endpoint", "processing_time_ms", "result",
                     "models_ready", "max_concurrent"):
            val = getattr(record, key, None)
            if val is not None:
                log_record[key] = val
        if record.exc_info and record.exc_info[0]:
            log_record["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_record, default=str, ensure_ascii=False)


def setup_logging() -> None:
    """Configura el root logger con JSON a stdout."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)

    # Silenciar librerias ruidosas
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("deepface").setLevel(logging.WARNING)
    logging.getLogger("tf_keras").setLevel(logging.WARNING)
