"""Tests for real host, web-log, and historical data sources."""

import json
import threading
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd

from data_sources import (
    PrometheusSource,
    ReplaySource,
    SystemMetricsSource,
    TransientDataSourceError,
    WebLogSource,
    ZeekJSONSource,
    create_data_source,
)


def test_system_metrics_source_collects_bounded_numeric_samples():
    source = SystemMetricsSource(window_size=3, cpu_interval=0)

    first = source.collect()
    second = source.collect()

    assert len(first) == 1
    assert len(second) == 2
    assert set(second[-1]) >= {
        "cpu_percent",
        "memory_percent",
        "disk_read_bytes_per_second",
        "network_recv_bytes_per_second",
    }
    assert all(np.isfinite(value) for value in second[-1].values())
    assert source.snapshot


def test_csv_replay_batches_without_repeating_or_exceeding_window():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "metrics.csv"
        pd.DataFrame({
            "timestamp": [1, 2, 3, 4, 5],
            "cpu": [10, 20, np.nan, 40, 50],
            "memory": [30, 40, 50, 60, 70],
        }).to_csv(path, index=False)

        source = ReplaySource(path, batch_size=2, window_size=3)
        first = source.collect()
        second = source.collect()
        final = source.collect()
        source.stop()

        assert len(first) == 2
        assert len(second) == 3
        assert len(final) == 3
        assert source.invalid_records == 1
        assert source.finished
        assert source.collect() == []
        assert second[-1] == {"cpu": 40.0, "memory": 60.0}
        assert all(np.isfinite(value) for row in second for value in row.values())


def test_json_replay_and_source_validation():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "metrics.json"
        path.write_text(json.dumps([
            {"timestamp": "2026-01-01T00:00:00Z", "cpu": 10, "memory": 30},
            {"timestamp": "2026-01-01T00:00:01Z", "cpu": 20, "memory": 40},
        ]), encoding="utf-8-sig")

        source = create_data_source("json", str(path), batch_size=1)
        assert source.collect() == [{"cpu": 10.0, "memory": 30.0}]
        assert source.collect() == [
            {"cpu": 10.0, "memory": 30.0},
            {"cpu": 20.0, "memory": 40.0},
        ]
        assert source.finished
        source.stop()

        wrapped_path = Path(directory) / "wrapped.json"
        wrapped_path.write_text(
            json.dumps({"data": [{"cpu": 25, "memory": 50}]}),
            encoding="utf-8",
        )
        wrapped_source = create_data_source("json", str(wrapped_path), batch_size=1)
        assert wrapped_source.collect() == [{"cpu": 25.0, "memory": 50.0}]
        wrapped_source.stop()

        try:
            ReplaySource(wrapped_path, window_size=0)
        except ValueError as exc:
            assert "window_size" in str(exc)
        else:
            raise AssertionError("Replay windows must have a positive size")

        try:
            create_data_source("csv")
        except ValueError as exc:
            assert "--input" in str(exc)
        else:
            raise AssertionError("File-backed sources must require an input path")


def test_nested_jsonl_replay_preserves_numeric_field_names_and_schema():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "fluent-bit.jsonl"
        records = [
            {
                "timestamp": 1729050000,
                "source": "fluent-bit",
                "metrics": {
                    "cpu_usage": 15.5,
                    "memory_usage": 42.1,
                    "request_rate": 120.0,
                    "error_rate": 0.02,
                },
            },
            {
                "metrics": {
                    "error_rate": 0.03,
                    "request_rate": 125.0,
                    "memory_usage": 43.0,
                    "cpu_usage": 16.0,
                },
                "source": "fluent-bit",
                "timestamp": 1729050001,
            },
            {
                "timestamp": 1729050002,
                "metrics": {"cpu_usage": 17.0},
                "source": "fluent-bit",
            },
        ]
        path.write_text(
            "\n".join(json.dumps(record) for record in records) + "\n",
            encoding="utf-8",
        )

        source = create_data_source("jsonl", str(path), batch_size=1)
        first = source.collect()
        second = source.collect()

        assert first == [{
            "metrics.cpu_usage": 15.5,
            "metrics.error_rate": 0.02,
            "metrics.memory_usage": 42.1,
            "metrics.request_rate": 120.0,
        }]
        assert second == [
            first[0],
            {
                "metrics.cpu_usage": 16.0,
                "metrics.error_rate": 0.03,
                "metrics.memory_usage": 43.0,
                "metrics.request_rate": 125.0,
            },
        ]
        third = source.collect()
        assert third[-1] == {
            "metrics.cpu_usage": 17.0,
            "metrics.error_rate": 0.0,
            "metrics.memory_usage": 0.0,
            "metrics.request_rate": 0.0,
        }
        assert source.finished
        source.stop()


def test_jsonl_replay_rejects_records_without_numeric_features():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "metadata.jsonl"
        path.write_text(
            '{"timestamp": 1729050000, "source": "fluent-bit"}\n',
            encoding="utf-8",
        )
        source = create_data_source("jsonl", str(path), batch_size=1)
        try:
            source.collect()
        except ValueError as exc:
            assert "No finite numeric feature rows" in str(exc)
        else:
            raise AssertionError("Metadata-only records must not be treated as telemetry")
        finally:
            source.stop()


def test_prometheus_source_scrapes_real_format_and_keeps_series_schema():
    payloads = [
        b'# HELP cpu_usage Host CPU usage\n'
        b'# TYPE cpu_usage gauge\n'
        b'cpu_usage{zone="east",core="0"} 0.5\n'
        b'requests_total 10 1729050000000\n',
        b'cpu_usage{core="0",zone="east"} 0.75\n'
        b'queue_depth 3\n',
    ]

    class MetricsHandler(BaseHTTPRequestHandler):
        request_count = 0

        def do_GET(self):
            assert self.path == "/metrics"
            payload = payloads[min(self.request_count, len(payloads) - 1)]
            type(self).request_count += 1
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, _format, *_args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), MetricsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    source = create_data_source(
        "prometheus",
        f"http://127.0.0.1:{server.server_port}/metrics",
        window_size=4,
    )
    try:
        first = source.collect()
        second = source.collect()

        assert source.name == "prometheus"
        assert source.snapshot
        assert source.min_samples == 4
        assert first == [
            {
                'cpu_usage{core="0",zone="east"}': 0.5,
                "requests_total": 10.0,
            }
        ]
        assert second == [
            {
                'cpu_usage{core="0",zone="east"}': 0.5,
                "queue_depth": 0.0,
                "requests_total": 10.0,
            },
            {
                'cpu_usage{core="0",zone="east"}': 0.75,
                "queue_depth": 3.0,
                "requests_total": 0.0,
            },
        ]
    finally:
        source.stop()
        server.shutdown()
        server.server_close()
        thread.join()


def test_prometheus_source_fails_closed_for_invalid_metrics_and_limits():
    source = PrometheusSource("http://localhost:9090/metrics", max_series=1)
    for payload, expected_message in [
        ("# no samples\n", "no numeric samples"),
        ("metric NaN\n", "Non-finite"),
        ("metric 1\nother_metric 2\n", "series limit"),
        ('metric{label="unterminated} 1\n', "Invalid Prometheus label set"),
    ]:
        try:
            source._parse_response(payload)
        except ValueError as exc:
            assert expected_message.lower() in str(exc).lower()
        else:
            raise AssertionError(f"Expected invalid Prometheus payload to fail: {payload!r}")

    try:
        PrometheusSource("file:///metrics")
    except ValueError as exc:
        assert "http://" in str(exc)
    else:
        raise AssertionError("Only HTTP(S) Prometheus endpoints must be accepted")

    try:
        create_data_source("prometheus")
    except ValueError as exc:
        assert "--input" in str(exc)
    else:
        raise AssertionError("Prometheus source must require an endpoint URL")


def test_prometheus_source_enforces_response_size_limit():
    class LargeResponseHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            payload = b"metric_value 12345\n"
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, _format, *_args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), LargeResponseHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    source = PrometheusSource(
        f"http://127.0.0.1:{server.server_port}/metrics",
        max_response_bytes=4,
    )
    try:
        try:
            source.collect()
        except ValueError as exc:
            assert "byte limit" in str(exc)
        else:
            raise AssertionError("Oversized Prometheus responses must be rejected")
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_prometheus_retries_server_errors_as_transient_source_failures():
    class TemporaryFailureHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(503)
            self.end_headers()

        def log_message(self, _format, *_args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), TemporaryFailureHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    source = PrometheusSource(f"http://127.0.0.1:{server.server_port}/metrics")
    try:
        try:
            source.collect()
        except TransientDataSourceError as exc:
            assert "503" in str(exc)
        else:
            raise AssertionError("HTTP 5xx responses should be classified as transient")
    finally:
        source.stop()
        server.shutdown()
        server.server_close()
        thread.join()


def test_zeek_json_source_keeps_allowlisted_connection_and_http_features():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "conn.log"
        records = [
            {
                "ts": 1729050000.0,
                "uid": "C1",
                "id.orig_h": "192.0.2.1",
                "id.orig_p": 43210,
                "id.resp_p": 443,
                "duration": 0.25,
                "orig_bytes": 120,
                "resp_bytes": 900,
                "history": "ShADadFf",
            },
            {
                "ts": 1729050001.0,
                "uid": "C2",
                "id.orig_p": 43211,
                "id.resp_p": 80,
                "duration": 0.5,
                "orig_bytes": 80,
                "resp_bytes": 200,
            },
            {
                "ts": 1729050002.0,
                "uid": "C3",
                "id.orig_p": 43212,
                "id.resp_p": 80,
                "trans_depth": 1,
                "status_code": 200,
                "request_body_len": 0,
                "response_body_len": 1024,
                "method": "GET",
                "host": "example.test",
            },
            {
                "ts": 1729050003.0,
                "id.orig_p": 43213,
                "id.resp_p": 80,
                "trans_depth": 1,
                "status_code": 404,
                "response_body_len": 0,
            },
        ]
        path.write_text(
            "\n".join(json.dumps(record) for record in records) + "\n",
            encoding="utf-8",
        )

        source = create_data_source("zeek", str(path), batch_size=2, window_size=4)
        first = source.collect()
        second = source.collect()

        assert source.name == "zeek"
        assert first[-1]["zeek.duration"] == 0.5
        assert len(second) == 4
        assert second[-2]["zeek.status_code"] == 200.0
        assert second[-2]["zeek.response_body_len"] == 1024.0
        assert second[-1]["zeek.status_code"] == 404.0
        assert second[-1]["zeek.duration"] == 0.0
        assert all(
            not field.endswith(".ts") and ".uid" not in field and ".history" not in field
            for row in second
            for field in row
        )
        assert source.finished
        source.stop()


def test_zeek_source_rejects_unsupported_fields_and_invalid_extensions():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "metadata.log"
        path.write_text(
            '{"ts": 1729050000, "uid": "C1", "id.orig_h": "192.0.2.1"}\n',
            encoding="utf-8",
        )
        source = create_data_source("zeek", str(path), batch_size=1)
        try:
            source.collect()
        except ValueError as exc:
            assert "No finite numeric feature rows" in str(exc)
        else:
            raise AssertionError("Zeek metadata must not be treated as numeric telemetry")
        finally:
            source.stop()

        try:
            create_data_source("zeek", str(Path(directory) / "traffic.csv"))
        except ValueError as exc:
            assert "zeek source requires a JSON" in str(exc)
        else:
            raise AssertionError("Zeek source must reject unsupported log formats")


def test_web_log_source_reads_appends_and_ignores_no_new_data():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "access.log"
        lines = [
            f'192.0.2.{index} - - [16/Oct/2026:00:00:{index:02d} +0000] '
            f'"GET /page/{index} HTTP/1.1" 200 123\n'
            for index in range(1, 12)
        ]
        path.write_text("".join(lines), encoding="utf-8")
        source = WebLogSource(path, window_size=20)

        initial = source.collect()
        assert len(initial) == 11
        assert all(len(row) == 7 for row in initial)
        assert source.collect() == []

        with path.open("a", encoding="utf-8") as log_file:
            log_file.write(
                '198.51.100.2 - - [16/Oct/2026:00:01:00 +0000] '
                '"POST /login HTTP/1.1" 401 42\n'
            )
        appended = source.collect()
        assert len(appended) == 1
        assert appended[0]["web_feature_1"] == 401.0


def test_web_log_source_reports_invalid_input():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "invalid.log"
        path.write_text("not a valid access log\n", encoding="utf-8")
        source = WebLogSource(path)
        try:
            source.collect()
        except ValueError as exc:
            assert "No valid log entries" in str(exc)
        else:
            raise AssertionError("Invalid logs must not be replaced by synthetic data")


if __name__ == "__main__":
    tests = [
        test_system_metrics_source_collects_bounded_numeric_samples,
        test_csv_replay_batches_without_repeating_or_exceeding_window,
        test_json_replay_and_source_validation,
        test_nested_jsonl_replay_preserves_numeric_field_names_and_schema,
        test_jsonl_replay_rejects_records_without_numeric_features,
        test_prometheus_source_scrapes_real_format_and_keeps_series_schema,
        test_prometheus_source_fails_closed_for_invalid_metrics_and_limits,
        test_prometheus_source_enforces_response_size_limit,
        test_prometheus_retries_server_errors_as_transient_source_failures,
        test_zeek_json_source_keeps_allowlisted_connection_and_http_features,
        test_zeek_source_rejects_unsupported_fields_and_invalid_extensions,
        test_web_log_source_reads_appends_and_ignores_no_new_data,
        test_web_log_source_reports_invalid_input,
    ]
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")
