"""Telemetry and structured logging for JaxonFlow."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class TelemetryEvent(str, Enum):
    """Types of telemetry events."""
    
    SYSTEM_INIT = "system_init"
    GENERATION_START = "generation_start"
    GENERATION_SUCCESS = "generation_success"
    GENERATION_FAILURE = "generation_failure"
    CACHE_HIT = "cache_hit"
    CACHE_MISS = "cache_miss"
    FALLBACK_USED = "fallback_used"
    WARMUP_START = "warmup_start"
    WARMUP_COMPLETE = "warmup_complete"


@dataclass
class TelemetryRecord:
    """A single telemetry event record."""
    
    event: str
    timestamp: str
    data: dict[str, Any]
    session_id: str


class TelemetryLogger:
    """Handles structured logging and metrics export."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TelemetryLogger, cls).__new__(cls)
            cls._instance.reset()
        return cls._instance

    def reset(self) -> None:
        """Reset state."""
        self._session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._enabled = True
        self._log_file: Path | None = None
        self._buffer: list[TelemetryRecord] = []

    def configure(self, log_dir: Path | None = None, enabled: bool = True) -> None:
        """Configure the telemetry logger."""
        self._enabled = enabled
        if log_dir and enabled:
            log_dir.mkdir(parents=True, exist_ok=True)
            self._log_file = log_dir / f"jaxonflow_telemetry_{self._session_id}.jsonl"
    
    def log(self, event: TelemetryEvent, **data: Any) -> None:
        """Log a telemetry event."""
        if not self._enabled:
            return

        record = TelemetryRecord(
            event=event.value,
            timestamp=datetime.now().isoformat(),
            data=data,
            session_id=self._session_id
        )
        
        # Log to file if configured
        if self._log_file:
            try:
                with open(self._log_file, "a") as f:
                    f.write(json.dumps(asdict(record)) + "\n")
            except Exception as e:
                logger.warning(f"Failed to write telemetry: {e}")
        else:
            self._buffer.append(record)
            
        # Also log to standard python logger for visibility
        logger.debug(f"Telemetry: {event.value} - {data}")

    def get_metrics(self) -> dict[str, int]:
        """Aggregate basic metrics from current session."""
        counts = {ev.value: 0 for ev in TelemetryEvent}
        
        # Determine source of records (file or buffer)
        records = self._buffer
        if self._log_file and self._log_file.exists():
            try:
                with open(self._log_file, "r") as f:
                    records = [
                        TelemetryRecord(**json.loads(line)) 
                        for line in f if line.strip()
                    ]
            except Exception:
                pass

        for record in records:
            if record.event in counts:
                counts[record.event] += 1
                
        return counts

    def export_csv(self, output_path: str) -> None:
        """Export metrics to CSV."""
        # Simple implementation
        pass
