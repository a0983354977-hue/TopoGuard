"""
Read-Only SIEM Integration for TopoGuard
-----------------------------------------
Emits structured alerts to enterprise SIEM systems without modifying the host.

Supported sinks:
- ECS JSON (Elastic Common Schema): one JSON object per line, append to a file or stdout
- Syslog RFC 5424: UDP datagrams to a remote syslog collector

TopoGuard stays a read-only intelligence probe: it only observes and emits.
"""

from __future__ import annotations

import json
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

ECS_SEVERITY_MAP = {
    "CRITICAL": 2,
    "HIGH": 2,
    "MEDIUM": 4,
    "LOW": 6,
    "NONE": 6,
    "UNKNOWN": 6,
    "ERROR": 3,
}

SYSLOG_SEVERITY_MAP = {
    "CRITICAL": 2,
    "HIGH": 2,
    "MEDIUM": 4,
    "LOW": 6,
    "NONE": 6,
    "UNKNOWN": 6,
    "ERROR": 3,
}

SYSLOG_FACILITY_USER = 8


class SiemEmitter:
    """Base class for read-only SIEM emitters."""

    def emit(self, alert: dict[str, Any]) -> None:
        raise NotImplementedError

    def close(self) -> None:
        return None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class EcsJsonEmitter(SiemEmitter):
    """Append ECS-formatted alert JSON (one object per line) to a file or stdout."""

    def __init__(self, output: str | Path = "-"):
        self._output = output
        self._stream = None
        if output == "-":
            self._stream = None  # write to stdout per emit
        else:
            self._stream = Path(output).open("a", encoding="utf-8")

    def emit(self, alert: dict[str, Any]) -> None:
        line = json.dumps(alert, ensure_ascii=False, allow_nan=False)
        if self._stream is None:
            print(line)
        else:
            self._stream.write(line + "\n")
            self._stream.flush()

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None


class SyslogEmitter(SiemEmitter):
    """Send RFC 5424 syslog datagrams over UDP to a remote collector (read-only)."""

    def __init__(
        self,
        host: str,
        port: int = 514,
        facility: int = SYSLOG_FACILITY_USER,
        appname: str = "topoguard",
    ):
        self._host = host
        self._port = int(port)
        self._facility = int(facility)
        self._appname = appname
        self._hostname = socket.gethostname()
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def emit(self, alert: dict[str, Any]) -> None:
        try:
            message = self._format_rfc5424(alert)
            self._sock.sendto(message.encode("utf-8"), (self._host, self._port))
        except OSError:
            # Best-effort SIEM output; never crash the monitor on delivery failure
            pass

    def close(self) -> None:
        try:
            self._sock.close()
        except OSError:
            pass

    @staticmethod
    def _syslog_severity(alert: dict[str, Any]) -> int:
        sev = str(alert.get("alert", {}).get("severity", "UNKNOWN")).upper()
        return SYSLOG_SEVERITY_MAP.get(sev, 6)

    def _format_rfc5424(self, alert: dict[str, Any]) -> str:
        pri = self._facility * 8 + self._syslog_severity(alert)
        timestamp = alert.get("@timestamp") or datetime.now(timezone.utc).isoformat()
        msg = alert.get("message", "topoguard alert")
        # Escape MSG content: keep it single-line
        safe_msg = msg.replace("\n", " ").replace("\r", " ")
        structured = (
            f'[topoguard@12345 alert_severity="{alert.get("alert", {}).get("severity", "UNKNOWN")}" '
            f'source="{alert.get("source", {}).get("name", "unknown")}" '
            f'anomaly_score="{alert.get("topoguard", {}).get("anomaly_score", 0.0)}"]'
        )
        return (
            f"<{pri}>1 {timestamp} {self._hostname} {self._appname} - - "
            f"{structured} {safe_msg}"
        )


def build_alert(
    *,
    source: str,
    tick: int,
    result: dict[str, Any],
    message: str,
) -> dict[str, Any]:
    """Build a SIEM alert dict (ECS + TopoGuard extensions) from an inspection result."""
    context = result.get("context") if isinstance(result.get("context"), dict) else {}
    z3_result = result.get("z3_result")
    score = float(result.get("anomaly_score", context.get("anomaly_score", 0.0)))
    severity = str(
        context.get("severity", result.get("severity", "UNKNOWN"))
    ).upper()

    return {
        "@timestamp": datetime.now(timezone.utc).isoformat(),
        "event": {
            "module": "topoguard",
            "dataset": "topoguard.alert",
            "action": "anomaly_detected" if result.get("is_anomaly") else "inspection",
            "outcome": "alert" if result.get("is_anomaly") else "ok",
            "severity": ECS_SEVERITY_MAP.get(severity, 6),
        },
        "alert": {
            "severity": severity,
            "rule": {"id": "topoguard", "description": message},
        },
        "host": {"hostname": socket.gethostname()},
        "source": {"name": source},
        "topoguard": {
            "tick": tick,
            "anomaly_score": score,
            "is_anomaly": bool(result.get("is_anomaly")),
            "context": context,
            "z3_result": z3_result,
        },
        "message": message,
    }


def siem_emitter_from_spec(spec: str) -> SiemEmitter:
    """Factory: 'ecs-json:/path/to/file' or 'ecs-json:-' or 'syslog://host[:port]'."""
    spec = spec.strip()
    if spec.lower().startswith("ecs-json:"):
        output = spec[len("ecs-json:"):]
        return EcsJsonEmitter(output)
    if spec.lower().startswith("syslog://"):
        parsed = urlsplit(spec)
        host = parsed.hostname
        if not host:
            raise ValueError("Syslog specifier requires a host: syslog://host[:port]")
        port = parsed.port or 514
        return SyslogEmitter(host, port)
    raise ValueError(
        "Unsupported SIEM spec; use 'ecs-json:<path>' or 'syslog://host[:port]'"
    )
