"""Bounded, explicit data sources for live monitoring and historical replay."""

from __future__ import annotations

import json
import logging
import math
import platform
import re
import time
from collections import deque
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import psutil

logger = logging.getLogger(__name__)


class TransientDataSourceError(RuntimeError):
    """A retryable collection failure that does not invalidate the source schema."""


_PROMETHEUS_SAMPLE = re.compile(
    r'^\s*([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(.*)\})?\s+([^\s]+)(?:\s+[0-9]+)?\s*$'
)
_PROMETHEUS_LABEL = re.compile(
    r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*"((?:\\.|[^"\\])*)"\s*'
)


class SystemMetricsSource:
    """Collect real host CPU, memory, disk, network, and process metrics."""

    name = "system"
    snapshot = True
    finished = False
    min_samples = 10

    def __init__(self, window_size: int = 1000, cpu_interval: float = 0.1):
        if window_size < 1:
            raise ValueError("window_size must be positive")
        if cpu_interval < 0:
            raise ValueError("cpu_interval cannot be negative")
        self._history: deque[dict[str, float]] = deque(maxlen=window_size)
        self.min_samples = min(10, window_size)
        self._cpu_interval = cpu_interval
        self._previous_time = time.monotonic()
        self._previous_disk = psutil.disk_io_counters()
        self._previous_network = psutil.net_io_counters()
        psutil.cpu_percent(interval=None)

    @staticmethod
    def _rate(current: int, previous: int | None, elapsed: float) -> float:
        if previous is None or elapsed <= 0:
            return 0.0
        return max(0.0, (current - previous) / elapsed)

    def collect(self) -> list[dict[str, float]]:
        now = time.monotonic()
        elapsed = max(now - self._previous_time, 1e-6)
        disk = psutil.disk_io_counters()
        network = psutil.net_io_counters()
        memory = psutil.virtual_memory()

        row = {
            "cpu_percent": float(psutil.cpu_percent(interval=self._cpu_interval)),
            "memory_percent": float(memory.percent),
            "memory_available_bytes": float(memory.available),
            "disk_read_bytes_per_second": self._rate(
                disk.read_bytes if disk else 0,
                self._previous_disk.read_bytes if self._previous_disk else None,
                elapsed,
            ),
            "disk_write_bytes_per_second": self._rate(
                disk.write_bytes if disk else 0,
                self._previous_disk.write_bytes if self._previous_disk else None,
                elapsed,
            ),
            "network_sent_bytes_per_second": self._rate(
                network.bytes_sent if network else 0,
                self._previous_network.bytes_sent if self._previous_network else None,
                elapsed,
            ),
            "network_recv_bytes_per_second": self._rate(
                network.bytes_recv if network else 0,
                self._previous_network.bytes_recv if self._previous_network else None,
                elapsed,
            ),
            "process_count": float(len(psutil.pids())),
        }
        if hasattr(psutil, "getloadavg") and platform.system() != "Windows":
            load = psutil.getloadavg()
            row["load_average_1m"] = float(load[0])

        self._history.append(row)
        self._previous_time = now
        self._previous_disk = disk
        self._previous_network = network
        return list(self._history)

    def stop(self) -> None:
        return None


class EBPFSystemCallSource:
    """Aggregate real Linux eBPF execve events into numeric inspection rows."""

    name = "ebpf"
    snapshot = True
    finished = False
    min_samples = 10

    def __init__(self, window_size: int = 1000, poll_interval: float = 1.0):
        if window_size < 1:
            raise ValueError("window_size must be positive")
        if poll_interval < 0:
            raise ValueError("poll_interval cannot be negative")
        from ebpf_collector import EBPFCollector

        self._history: deque[dict[str, float]] = deque(maxlen=window_size)
        self.min_samples = min(10, window_size)
        self._collector = EBPFCollector(require_real=True)
        self._fallback: SystemMetricsSource | None = None
        self._poll_interval = max(0.0, poll_interval)
        self._collector.start()

    def collect(self) -> list[dict[str, float]]:
        if self._fallback is not None:
            return self._fallback.collect()
        deadline = time.monotonic() + self._poll_interval
        try:
            events = self._collector.poll()
            while not events and time.monotonic() < deadline:
                time.sleep(min(0.05, deadline - time.monotonic()))
                events = self._collector.poll()
        except RuntimeError as exc:
            self._collector.stop()
            self._fallback = SystemMetricsSource(window_size=self._history.maxlen or 1000)
            self.name = "system-fallback"
            logger.warning("eBPF collection failed; continuing with real psutil metrics: %s", exc)
            return self._fallback.collect()

        if not events:
            return []

        self._history.append({
            "event_count": float(len(events)),
            "unique_processes": float(len({event["pid"] for event in events})),
            "execve_count": float(sum(event["event_type"] == "execve" for event in events)),
            "connection_count": float(sum(event["event_type"] == "connect" for event in events)),
            "tcp_sendmsg_count": float(sum(event["event_type"] == "tcp_sendmsg" for event in events)),
            "network_bytes": float(sum(event.get("bytes", 0) for event in events)),
        })
        return list(self._history)

    def stop(self) -> None:
        self._collector.stop()
        if self._fallback is not None:
            self._fallback.stop()


class PrometheusSource:
    """Scrape bounded numeric samples from a Prometheus text exposition endpoint."""

    name = "prometheus"
    snapshot = True
    finished = False

    def __init__(
        self,
        endpoint: str,
        window_size: int = 1000,
        timeout: float = 5.0,
        max_response_bytes: int = 4 * 1024 * 1024,
        max_series: int = 1000,
    ):
        parsed = urlsplit(endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("Prometheus endpoint must be an absolute http:// or https:// URL")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("Credentials in Prometheus endpoint URLs are not supported")
        if window_size < 1 or timeout <= 0 or max_response_bytes < 1 or max_series < 1:
            raise ValueError("window_size, timeout, and Prometheus response limits must be positive")

        self.endpoint = endpoint
        self._timeout = timeout
        self._max_response_bytes = max_response_bytes
        self._max_series = max_series
        self._history: deque[dict[str, float]] = deque(maxlen=window_size)
        self.min_samples = min(4, window_size)

    @staticmethod
    def _parse_labels(payload: str) -> dict[str, str]:
        labels: dict[str, str] = {}
        position = 0
        while position < len(payload):
            match = _PROMETHEUS_LABEL.match(payload, position)
            if match is None:
                raise ValueError("Invalid Prometheus label set")
            name, escaped_value = match.groups()
            if name in labels:
                raise ValueError(f"Duplicate Prometheus label: {name}")

            value = []
            index = 0
            while index < len(escaped_value):
                character = escaped_value[index]
                if character == "\\":
                    index += 1
                    if index == len(escaped_value):
                        raise ValueError("Invalid escape in Prometheus label value")
                    character = escaped_value[index]
                    value.append("\n" if character == "n" else character)
                else:
                    value.append(character)
                index += 1
            labels[name] = "".join(value)

            position = match.end()
            if position == len(payload):
                break
            if payload[position] != ",":
                raise ValueError("Invalid separator in Prometheus label set")
            position += 1

        return labels

    def _parse_response(self, payload: str) -> dict[str, float]:
        samples: dict[str, float] = {}
        for line_number, line in enumerate(payload.splitlines(), start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            match = _PROMETHEUS_SAMPLE.match(line)
            if match is None:
                raise ValueError(f"Invalid Prometheus sample at line {line_number}")

            metric_name, label_payload, raw_value = match.groups()
            labels = self._parse_labels(label_payload) if label_payload is not None else {}
            try:
                value = float(raw_value)
            except ValueError as exc:
                raise ValueError(f"Invalid Prometheus value at line {line_number}") from exc
            if not math.isfinite(value):
                raise ValueError(f"Non-finite Prometheus value at line {line_number}")

            if labels:
                label_text = ",".join(
                    f"{name}={json.dumps(labels[name], ensure_ascii=True)}"
                    for name in sorted(labels)
                )
                feature_name = f"{metric_name}{{{label_text}}}"
            else:
                feature_name = metric_name
            if feature_name in samples:
                raise ValueError(f"Duplicate Prometheus series at line {line_number}")
            samples[feature_name] = value
            if len(samples) > self._max_series:
                raise ValueError(
                    f"Prometheus response exceeds the {self._max_series}-series limit"
                )

        if not samples:
            raise ValueError("Prometheus endpoint returned no numeric samples")
        return samples

    def collect(self) -> list[dict[str, float]]:
        request = Request(
            self.endpoint,
            headers={"Accept": "text/plain; version=0.0.4"},
        )
        try:
            with urlopen(request, timeout=self._timeout) as response:
                payload = response.read(self._max_response_bytes + 1)
        except HTTPError as exc:
            if exc.code == 429 or exc.code >= 500:
                raise TransientDataSourceError(
                    f"Prometheus scrape temporarily failed with HTTP {exc.code}"
                ) from exc
            raise RuntimeError(
                f"Prometheus scrape failed with HTTP {exc.code}"
            ) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise TransientDataSourceError(
                f"Prometheus scrape temporarily failed: {exc}"
            ) from exc

        if len(payload) > self._max_response_bytes:
            raise ValueError(
                f"Prometheus response exceeds the {self._max_response_bytes}-byte limit"
            )
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("Prometheus response must be UTF-8 text") from exc

        self._history.append(self._parse_response(text))
        feature_names = sorted({name for row in self._history for name in row})
        return [
            {name: row.get(name, 0.0) for name in feature_names}
            for row in self._history
        ]

    def stop(self) -> None:
        return None


class WebLogSource:
    """Read and monitor a web access log, retaining only a bounded line window."""

    name = "web-log"
    snapshot = False
    finished = False
    min_samples = 4

    def __init__(self, path: str | Path, window_size: int = 1000):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Access log does not exist: {self.path}")
        if window_size < 1:
            raise ValueError("window_size must be positive")
        self._lines: deque[str] = deque(maxlen=window_size)
        self._offset = 0
        self._partial_line = ""
        self._file_identity = None
        from adapters.module_b_web_traffic import WebTrafficGuard

        self._adapter = WebTrafficGuard()

    def collect(self) -> list[dict[str, float]]:
        stat = self.path.stat()
        identity = (stat.st_dev, stat.st_ino)
        if stat.st_size < self._offset or (
            self._file_identity is not None and identity != self._file_identity
        ):
            self._offset = 0
            self._lines.clear()
            self._partial_line = ""

        with self.path.open("r", encoding="utf-8", errors="replace") as log_file:
            log_file.seek(self._offset)
            new_lines = []
            while line := log_file.readline():
                combined = self._partial_line + line
                if line.endswith(("\n", "\r")):
                    new_lines.append(combined)
                    self._partial_line = ""
                else:
                    self._partial_line = combined
            self._offset = log_file.tell()
            self._file_identity = identity

        new_lines = [line for line in new_lines if line.strip()]
        self._lines.extend(new_lines)
        if not self._lines:
            return []

        if not new_lines:
            return []
        try:
            new_feature_count = self._adapter.parse_access_log_lines(new_lines).shape[0]
        except ValueError as exc:
            raise ValueError(f"No valid log entries found in access log file {self.path}") from exc
        matrix = self._adapter.parse_access_log_lines(list(self._lines))
        rows_to_return = min(new_feature_count, matrix.shape[0])
        if rows_to_return == 0:
            return []
        return [
            {f"web_feature_{index}": float(value) for index, value in enumerate(row)}
            for row in matrix[-rows_to_return:]
        ]

    def stop(self) -> None:
        return None


class ReplaySource:
    """Replay numeric CSV/JSON records in bounded, non-repeating batches."""

    finished = False
    snapshot = True
    min_samples = 4

    def __init__(
        self,
        path: str | Path,
        batch_size: int = 1000,
        window_size: int = 1000,
        *,
        record_format: str = "auto",
    ):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Replay data file does not exist: {self.path}")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if window_size < 1:
            raise ValueError("window_size must be positive")
        self.name = "zeek" if record_format == "zeek" else self.path.suffix.lower().lstrip(".")
        self._batch_size = batch_size
        self._offset = 0
        self._history: deque[dict[str, float]] = deque(maxlen=window_size)
        self.finished = False
        self._pending = None
        self.invalid_records = 0
        if record_format == "zeek":
            if self.path.suffix.lower() not in {".log", ".json", ".jsonl"}:
                raise ValueError("Zeek JSON replay requires a .log, .json, or .jsonl file")
            self._reader = iter(self._json_records())
        elif self.path.suffix.lower() == ".csv":
            self._reader = pd.read_csv(self.path, chunksize=batch_size, encoding="utf-8-sig")
        elif self.path.suffix.lower() in {".json", ".jsonl"}:
            self._reader = iter(self._json_records())
        else:
            raise ValueError("Replay input must be a .csv, .json, or .jsonl file")

    def _json_records(self):
        with self.path.open("r", encoding="utf-8-sig") as source_file:
            first = ""
            while not first.strip():
                first = source_file.readline()
                if not first:
                    return

            if first.lstrip().startswith("["):
                yield from self._json_array_records(source_file, first)
                return

            try:
                first_record = json.loads(first)
            except json.JSONDecodeError:
                payload = json.loads(first + source_file.read())
                records = payload.get("data", payload) if isinstance(payload, dict) else payload
                if isinstance(records, list):
                    yield from records
                else:
                    yield records
                return

            if isinstance(first_record, dict) and isinstance(first_record.get("data"), list):
                yield from first_record["data"]
            elif isinstance(first_record, dict):
                yield first_record
                for line in source_file:
                    if line.strip():
                        yield json.loads(line)
            elif isinstance(first_record, list):
                yield from first_record
            else:
                yield {"value": first_record}

    @staticmethod
    def _json_array_records(source_file, first_line: str):
        decoder = json.JSONDecoder()
        buffer = first_line[first_line.index("[") + 1:]
        while True:
            buffer = buffer.lstrip()
            while buffer.startswith(","):
                buffer = buffer[1:].lstrip()
            if buffer.startswith("]"):
                return
            try:
                record, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                chunk = source_file.read(65536)
                if not chunk:
                    raise ValueError("Invalid or incomplete JSON array")
                buffer += chunk
                continue
            yield record
            buffer = buffer[end:]

    @classmethod
    def _flatten_numeric_fields(
        cls,
        record: Any,
        prefix: str = "",
    ) -> dict[str, float]:
        if isinstance(record, dict):
            flattened: dict[str, float] = {}
            for key, value in record.items():
                field = f"{prefix}.{key}" if prefix else str(key)
                flattened.update(cls._flatten_numeric_fields(value, field))
            return flattened

        if isinstance(record, (list, tuple)):
            return {}

        field_name = prefix.rsplit(".", 1)[-1].casefold().lstrip("@")
        if field_name in {"timestamp", "time", "epoch", "unix_time", "ts"}:
            return {}
        if isinstance(record, (int, float, np.integer, np.floating)) and not isinstance(record, (bool, np.bool_)):
            return {prefix or "value": float(record)}
        return {}

    def _numeric_records(self, records) -> list[dict[str, float]]:
        if isinstance(records, pd.DataFrame):
            raw_records = records.to_dict(orient="records")
        elif isinstance(records, dict):
            raw_records = [records]
        else:
            raw_records = records

        flattened = []
        for record in raw_records:
            numeric_row = self._flatten_numeric_fields(record)
            if any(not np.isfinite(value) for value in numeric_row.values()):
                self.invalid_records += 1
                logger.warning(
                    "Skipping replay row with non-finite numeric features "
                    "(rejected count=%s) in %s",
                    self.invalid_records,
                    self.path,
                )
                continue
            flattened.append(numeric_row)
        feature_names = sorted({
            name
            for row in flattened
            for name in row
        })
        if not feature_names:
            raise ValueError(f"No finite numeric feature rows found in {self.path}")

        return [
            {
                name: row.get(name, 0.0)
                for name in feature_names
            }
            for row in flattened
        ]

    def collect(self) -> list[dict[str, float]]:
        if self.finished:
            return []
        try:
            if self.name == "csv":
                records = self._next_records()
                self.finished = len(records) < self._batch_size
                if not self.finished:
                    try:
                        self._pending = next(self._reader)
                    except StopIteration:
                        self.finished = True
            else:
                records = []
                for _ in range(self._batch_size):
                    try:
                        records.append(self._next_records())
                    except StopIteration:
                        self.finished = True
                        break
                if not self.finished and len(records) == self._batch_size:
                    try:
                        self._pending = next(self._reader)
                    except StopIteration:
                        self.finished = True
                elif len(records) < self._batch_size:
                    self.finished = True
        except StopIteration:
            self.finished = True
            return []

        if records is None or (isinstance(records, list) and not records) or (
            isinstance(records, pd.DataFrame) and records.empty
        ):
            self.stop()
            return []
        new_rows = self._numeric_records(records)
        self._history.extend(new_rows)
        if self.finished:
            self.stop()
        feature_names = sorted({
            name
            for row in self._history
            for name in row
        })
        return [
            {name: row.get(name, 0.0) for name in feature_names}
            for row in self._history
        ]

    def _next_records(self):
        if self._pending is not None:
            records = self._pending
            self._pending = None
            return records
        return next(self._reader)

    def stop(self) -> None:
        close = getattr(self._reader, "close", None)
        if close is not None:
            close()


class ZeekJSONSource(ReplaySource):
    """Replay only recognized numeric fields from Zeek JSON logs."""

    _NUMERIC_FIELDS = frozenset({
        "duration",
        "orig_bytes",
        "resp_bytes",
        "missed_bytes",
        "orig_pkts",
        "orig_ip_bytes",
        "resp_pkts",
        "resp_ip_bytes",
        "id.orig_p",
        "id.resp_p",
        "trans_depth",
        "request_body_len",
        "response_body_len",
        "status_code",
    })

    def __init__(self, path: str | Path, batch_size: int = 1000, window_size: int = 1000):
        super().__init__(
            path,
            batch_size=batch_size,
            window_size=window_size,
            record_format="zeek",
        )

    @classmethod
    def _flatten_numeric_fields(
        cls,
        record: Any,
        prefix: str = "",
    ) -> dict[str, float]:
        if not isinstance(record, dict):
            return {}

        features = {}
        for name in cls._NUMERIC_FIELDS:
            value = record.get(name)
            if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
                value, (bool, np.bool_)
            ):
                numeric = float(value)
                if not np.isfinite(numeric):
                    raise ValueError(f"Zeek field {name} must be finite")
                features[f"zeek.{name}"] = numeric
        return features


class PcapSource:
    """Replay bounded numeric metadata from an offline PCAP/PCAPNG capture."""

    name = "pcap"
    snapshot = True
    min_samples = 4

    def __init__(
        self,
        path: str | Path,
        batch_size: int = 1000,
        window_size: int = 1000,
    ):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Packet capture does not exist: {self.path}")
        if batch_size < 1 or window_size < 1:
            raise ValueError("batch_size and window_size must be positive")
        try:
            from scapy.utils import PcapReader
        except ImportError as exc:
            raise RuntimeError("PCAP auditing requires Scapy: install scapy") from exc

        self._reader = PcapReader(str(self.path))
        self._batch_size = batch_size
        self._history: deque[dict[str, float]] = deque(maxlen=window_size)
        self._valid_packets = 0
        self.finished = False

    def collect(self) -> list[dict[str, float]]:
        if self.finished:
            return []
        from scapy.layers.inet import ICMP, IP, TCP, UDP

        batch = []
        try:
            while len(batch) < self._batch_size:
                packet = next(self._reader)
                if IP not in packet:
                    continue
                ip = packet[IP]
                transport = packet[TCP] if TCP in packet else packet[UDP] if UDP in packet else None
                batch.append({
                    "packet_length": float(len(packet)),
                    "ip_protocol": float(ip.proto),
                    "source_port": float(transport.sport if transport else 0),
                    "destination_port": float(transport.dport if transport else 0),
                    "ttl": float(ip.ttl),
                    "tcp_syn": float(bool(TCP in packet and packet[TCP].flags & 0x02)),
                    "icmp": float(ICMP in packet),
                })
        except StopIteration:
            self.finished = True

        if batch:
            self._valid_packets += len(batch)
            self._history.extend(batch)
        elif self.finished and not self._valid_packets:
            raise ValueError(f"No IPv4 packets found in capture: {self.path}")
        return list(self._history) if batch else []

    def stop(self) -> None:
        self._reader.close()
        self.finished = True


def create_data_source(
    source: str,
    input_path: str | None = None,
    window_size: int = 1000,
    batch_size: int = 1000,
    poll_interval: float = 1.0,
):
    """Create the requested source without silently substituting synthetic data."""
    if source == "system":
        return SystemMetricsSource(window_size=window_size)
    if source == "ebpf":
        try:
            return EBPFSystemCallSource(window_size=window_size, poll_interval=poll_interval)
        except RuntimeError as exc:
            fallback = SystemMetricsSource(window_size=window_size)
            fallback.name = "system-fallback"
            fallback.degraded_reason = str(exc)
            logger.warning("eBPF is unavailable; using real psutil system metrics: %s", exc)
            return fallback
    if source == "prometheus":
        if not input_path:
            raise ValueError("--input is required for the Prometheus source URL")
        return PrometheusSource(input_path, window_size=window_size)
    if source == "web-log":
        if not input_path:
            raise ValueError("--input is required for the web-log source")
        return WebLogSource(input_path, window_size=window_size)
    if source in {"csv", "json", "jsonl"}:
        if not input_path:
            raise ValueError(f"--input is required for the {source} replay source")
        expected_suffix = f".{source}"
        if Path(input_path).suffix.lower() != expected_suffix:
            raise ValueError(f"The {source} source requires a {expected_suffix} file")
        return ReplaySource(
            input_path,
            batch_size=batch_size,
            window_size=window_size,
        )
    if source == "zeek":
        if not input_path:
            raise ValueError("--input is required for the zeek source")
        if Path(input_path).suffix.lower() not in {".log", ".json", ".jsonl"}:
            raise ValueError("The zeek source requires a JSON .log, .json, or .jsonl file")
        return ZeekJSONSource(
            input_path,
            batch_size=batch_size,
            window_size=window_size,
        )
    if source == "pcap":
        if not input_path:
            raise ValueError("--input is required for the pcap source")
        if Path(input_path).suffix.lower() not in {".pcap", ".pcapng", ".cap"}:
            raise ValueError("The pcap source requires a .pcap, .pcapng, or .cap file")
        return PcapSource(
            input_path,
            batch_size=batch_size,
            window_size=window_size,
        )
    raise ValueError(f"Unsupported data source: {source}")
