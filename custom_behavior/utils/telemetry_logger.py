import json
import logging
import time
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = PROJECT_ROOT / "logs" / "logs.txt"

class TelemetryLogger:
    def __init__(
        self,
        log_path: str,
        name: str = "telemetry",
        enabled: bool = True,
    ):
        self.enabled = enabled
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        if not enabled:
            self.logger.addHandler(logging.NullHandler())
            return

        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(message)s"))
        self.logger.addHandler(handler)

    def emit(self, event: str, **fields: Any) -> None:
        """
        Записывает одно телеметрическое событие (1 строка JSON)
        """
        if not self.enabled:
            return

        payload: Dict[str, Any] = {
            "ts": time.time(),
            "event": event,
            **fields,
        }

        self.logger.info(json.dumps(payload, ensure_ascii=False))

telemetry = TelemetryLogger(
    log_path=LOGS_DIR,
    enabled=True
)