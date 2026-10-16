"""
Time-Window Multi-Source Event Correlator
------------------------------------------
Joins heterogeneous telemetry (eBPF execve, Prometheus CPU spikes, Zeek
connections) that land in the same time window into a single Attack Context.

Read-only intelligence probe: observes and correlates, never modifies the host.

Responsibilities:
- Bucket ingested events by a fixed time window
- Per-source aggregation within each bucket (Polars group_by)
- Cross-source inner correlation on the bucket key
- Emit an Attack Context dict consumable by the Webhook notifier / SIEM output
"""

from __future__ import annotations

import time
from typing import Any

import polars as pl

# Detection thresholds for a correlated attack window
DEFAULT_WINDOW_SECONDS = 60.0
EBPF_EXECVE_THRESHOLD = 0          # any execve in the window counts
PROMETHEUS_CPU_SPIKE_THRESHOLD = 80.0  # cpu_percent > 80%
ZECK_MIN_CONNECTIONS = 1           # at least one Zeek connection event


class TimeWindowCorrelator:
    """Correlate multi-source events within fixed time windows."""

    def __init__(
        self,
        window_seconds: float = DEFAULT_WINDOW_SECONDS,
        ebpf_execve_threshold: int = EBPF_EXECVE_THRESHOLD,
        prometheus_cpu_spike_threshold: float = PROMETHEUS_CPU_SPIKE_THRESHOLD,
        zeek_min_connections: int = ZECK_MIN_CONNECTIONS,
    ):
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        if prometheus_cpu_spike_threshold <= 0:
            raise ValueError("prometheus_cpu_spike_threshold must be positive")
        self.window_seconds = float(window_seconds)
        self.ebpf_execve_threshold = ebpf_execve_threshold
        self.prometheus_cpu_spike_threshold = prometheus_cpu_spike_threshold
        self.zeek_min_connections = zeek_min_connections
        self._events: list[dict[str, Any]] = []
        self._last_context: dict[str, Any] | None = None

    def ingest(
        self,
        source: str,
        timestamp: float | None = None,
        **event: Any,
    ) -> None:
        """Ingest a single event from a source at a given wall-clock timestamp."""
        ts = float(time.time() if timestamp is None else timestamp)
        row: dict[str, Any] = {"source": source, "timestamp": ts}
        row.update(event)
        self._events.append(row)

    def ingest_batch(
        self,
        source: str,
        rows: list[dict[str, Any]],
        timestamp: float | None = None,
    ) -> None:
        """Ingest many rows from one source sharing one timestamp."""
        ts = float(time.time() if timestamp is None else timestamp)
        for row in rows:
            self.ingest(source, ts, **row)

    def _empty_context(self) -> dict[str, Any]:
        return {
            "correlated": False,
            "window_seconds": self.window_seconds,
            "buckets": [],
            "bucket_count": 0,
            "severity": "NONE",
            "summary": "no correlated events",
        }

    def correlate(self) -> dict[str, Any]:
        """Bucket, aggregate per source, and cross-correlate into Attack Contexts."""
        if not self._events:
            self._last_context = self._empty_context()
            return self._last_context

        df = pl.DataFrame(self._events)
        reserved = {"source", "timestamp"}
        df = df.with_columns(
            (pl.col("timestamp") // self.window_seconds)
            .cast(pl.Int64)
            .alias("bucket")
        )
        numeric_cols = [c for c in df.columns if c not in reserved | {"bucket"}]

        per_source = (
            df.group_by(["bucket", "source"])
            .agg(
                [
                    pl.col("timestamp").min().alias("t_start"),
                    pl.col("timestamp").max().alias("t_end"),
                    pl.len().alias("n_events"),
                    *[pl.col(c).max().alias(c) for c in numeric_cols],
                ]
            )
            .sort("bucket")
        )

        buckets = per_source.group_by("bucket").agg(
            [
                pl.col("source").alias("sources"),
                pl.col("t_start").min().alias("window_start"),
                pl.col("t_end").max().alias("window_end"),
                pl.col("n_events").sum().alias("total_events"),
            ]
        )

        contexts: list[dict[str, Any]] = []
        for row in buckets.iter_rows(named=True):
            bucket = int(row["bucket"])
            src_rows = {
                r["source"]: r
                for r in per_source.filter(pl.col("bucket") == bucket)
                .iter_rows(named=True)
            }
            sources = [str(s) for s in row["sources"]]

            ebpf = src_rows.get("ebpf", {})
            prom = src_rows.get("prometheus", {})
            zeek = src_rows.get("zeek", {})

            execve_count = float(ebpf.get("execve_count") or 0)
            cpu_percent = float(prom.get("cpu_percent") or 0.0)
            zeek_conn = float(zeek.get("connection_count") or 0)

            ebpf_hit = execve_count > self.ebpf_execve_threshold
            cpu_hit = cpu_percent > self.prometheus_cpu_spike_threshold
            zeek_hit = zeek_conn >= self.zeek_min_connections

            matched_sources = [s for s in ("ebpf", "prometheus", "zeek") if s in sources]
            correlated = ebpf_hit and cpu_hit and zeek_hit and len(matched_sources) == 3

            if not correlated:
                continue

            if cpu_percent > 95:
                severity = "CRITICAL"
            elif cpu_percent > 85:
                severity = "HIGH"
            else:
                severity = "MEDIUM"

            contexts.append(
                {
                    "bucket": bucket,
                    "window_start": float(row["window_start"]),
                    "window_end": float(row["window_end"]),
                    "sources": matched_sources,
                    "ebpf": {
                        "execve_count": execve_count,
                        "unique_processes": float(ebpf.get("unique_processes") or 0),
                        "connection_count": float(ebpf.get("connection_count") or 0),
                        "network_bytes": float(ebpf.get("network_bytes") or 0),
                    },
                    "prometheus": {
                        "cpu_percent": cpu_percent,
                        "memory_percent": float(prom.get("memory_percent") or 0.0),
                    },
                    "zeek": {
                        "connection_count": zeek_conn,
                        "orig_bytes": float(zeek.get("orig_bytes") or 0.0),
                        "resp_bytes": float(zeek.get("resp_bytes") or 0.0),
                    },
                    "severity": severity,
                }
            )

        self._last_context = {
            "correlated": len(contexts) > 0,
            "window_seconds": self.window_seconds,
            "buckets": contexts,
            "bucket_count": len(contexts),
            "severity": (
                max((c["severity"] for c in contexts), key=_SEVERITY_RANK.get)
                if contexts
                else "NONE"
            ),
        }
        self._last_context["summary"] = self._summary(self._last_context)
        return self._last_context

    def correlate_latest(self) -> dict[str, Any] | None:
        """Return the most recently computed Attack Context, or None."""
        return self._last_context

    def reset(self) -> None:
        """Clear ingested events and the last computed context."""
        self._events.clear()
        self._last_context = None

    @staticmethod
    def _summary(ctx: dict[str, Any]) -> str:
        if not ctx["correlated"]:
            return "no cross-source correlation"
        parts = [
            f"{b['severity']} attack window @{b['window_start']:.0f}s "
            f"sources={','.join(b['sources'])} "
            f"execve={b['ebpf']['execve_count']} "
            f"cpu={b['prometheus']['cpu_percent']:.1f}% "
            f"zeek_conn={b['zeek']['connection_count']}"
            for b in ctx["buckets"]
        ]
        return "; ".join(part for part in parts)


_SEVERITY_RANK = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
