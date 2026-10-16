"""Focused tests for the production CLI and formal boundary verifier."""

import asyncio
import io
import json
import tempfile
from types import SimpleNamespace
from urllib.error import HTTPError
from unittest.mock import MagicMock, Mock, patch
from pathlib import Path

import numpy as np

from extension_plugins import Z3FormalVerificationPlugin
from data_sources import PcapSource
from notifications import WEBHOOK_ENV_VAR, WebhookNotifier
from topoguard_cli import _report, build_parser, main


def test_cli_report_colors_normal_and_alert_statuses():
    class TerminalOutput(io.StringIO):
        def isatty(self):
            return True

    for values, expected_color, expected_border in (
        ({"status": "SUCCESS", "severity": "LOW"}, "\033[32m", "-"),
        ({"status": "SUCCESS", "anomaly": True, "severity": "HIGH"}, "\033[1;31m", "#"),
    ):
        output = TerminalOutput()
        with (
            patch("topoguard_cli._console", return_value=None),
            patch("topoguard_cli.sys.stdout", output),
        ):
            _report("Monitor tick", values)
        assert expected_color in output.getvalue()
        assert f"+{expected_border}" in output.getvalue()


def test_cli_parses_production_commands_and_default_source():
    parser = build_parser()
    monitor = parser.parse_args(["monitor", "--ticks", "1"])
    audit = parser.parse_args(["audit", "--input", "access.log"])
    export = parser.parse_args(["export-tda"])
    prometheus_monitor = parser.parse_args([
        "monitor",
        "--source", "prometheus",
        "--input", "http://localhost:9090/metrics",
    ])
    prometheus_export = parser.parse_args([
        "export-tda",
        "--source", "prometheus",
        "--input", "http://localhost:9090/metrics",
    ])
    zeek_audit = parser.parse_args([
        "audit",
        "--source", "zeek",
        "--input", "conn.log",
    ])
    notify_monitor = parser.parse_args(["monitor", "--notify"])

    assert monitor.source == "system"
    assert monitor.ticks == 1
    assert monitor.tda_max_dim == 1
    assert audit.source == "auto"
    assert export.source == "system"
    assert prometheus_monitor.source == "prometheus"
    assert prometheus_export.source == "prometheus"
    assert zeek_audit.source == "zeek"
    assert notify_monitor.notify
    assert parser.parse_args(["monitor", "--tda-max-dim", "2"]).tda_max_dim == 2

    from topoguard_cli import _make_engine

    with patch("defensive.TopoGuardEngine") as engine_factory:
        _make_engine(parser.parse_args(["monitor", "--tda-max-dim", "2"]))
    assert engine_factory.call_args.kwargs["tda_max_dimension"] == 2


def test_cli_audits_zeek_json_connection_log():
    with tempfile.TemporaryDirectory() as directory:
        input_path = Path(directory) / "conn.log"
        input_path.write_text(
            "\n".join(
                json.dumps({
                    "ts": index,
                    "uid": f"C{index}",
                    "id.orig_p": 20000 + index,
                    "id.resp_p": 443,
                    "duration": index / 10,
                    "orig_bytes": 100 * index,
                    "resp_bytes": 200 * index,
                })
                for index in range(1, 5)
            ) + "\n",
            encoding="utf-8",
        )
        result = main([
            "audit",
            "--source", "zeek",
            "--input", str(input_path),
            "--window-size", "4",
        ])
        assert result in (0, 1)


def test_webhook_notifier_sends_data_minimized_discord_and_slack_alerts():
    for webhook_url, expected_key in [
        ("https://hooks.slack.com/services/example/token", "text"),
        ("https://discord.com/api/webhooks/123/secret", "content"),
    ]:
        notifier = WebhookNotifier(webhook_url)
        opener = Mock()
        response = MagicMock()
        response.status = 204
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=None)
        opener.open.return_value = response

        with patch("notifications.build_opener", return_value=opener):
            assert not notifier.send_alert(
                {"status": "SUCCESS", "is_anomaly": False},
                "system",
                1,
            )
            assert opener.open.call_count == 0
            assert notifier.send_alert(
                {
                    "status": "SUCCESS",
                    "is_anomaly": True,
                    "anomaly_score": 0.91,
                    "context": {"severity": "CRITICAL"},
                },
                "prometheus",
                7,
            )

        request = opener.open.call_args.args[0]
        payload = json.loads(request.data)
        assert request.get_method() == "POST"
        assert set(payload) == {expected_key}
        assert "score=0.9100" in payload[expected_key]
        assert "source=prometheus" in payload[expected_key]
        assert "raw" not in payload[expected_key]


def test_webhook_notifier_validates_secrets_and_surfaces_delivery_errors():
    for invalid_url in (
        "http://hooks.slack.com/services/example/token",
        "https://example.invalid/webhook",
        "https://user:password@hooks.slack.com/services/example/token",
    ):
        try:
            WebhookNotifier(invalid_url)
        except ValueError:
            pass
        else:
            raise AssertionError("Unsupported or unsafe webhook endpoints must be rejected")

    with patch.dict("os.environ", {WEBHOOK_ENV_VAR: ""}):
        try:
            WebhookNotifier.from_environment()
        except ValueError as exc:
            assert WEBHOOK_ENV_VAR in str(exc)
        else:
            raise AssertionError("Missing webhook secrets must fail explicitly")

    notifier = WebhookNotifier("https://hooks.slack.com/services/sensitive/token")
    opener = Mock()
    opener.open.side_effect = HTTPError(
        "https://hooks.slack.com/services/sensitive/token",
        403,
        "forbidden",
        {},
        None,
    )
    with patch("notifications.build_opener", return_value=opener):
        try:
            notifier.send_alert({"is_anomaly": True}, "system", 3)
        except RuntimeError as exc:
            assert "HTTP 403" in str(exc)
            assert "sensitive" not in str(exc)
        else:
            raise AssertionError("Webhook delivery failures must be surfaced")


def test_cli_monitor_wires_opt_in_anomaly_notifications():
    class Source:
        name = "test-source"
        finished = False
        snapshot = True

        def collect(self):
            return [{"cpu": 1.0}]

        def stop(self):
            return None

    class Engine:
        def stop(self):
            return None

    notifier = Mock()
    notifier.send_alert.return_value = True
    result = {
        "status": "SUCCESS",
        "is_anomaly": True,
        "anomaly_score": 0.91,
        "source": "test-source",
        "tick": 1,
    }
    inspector = Mock(return_value=result)
    with (
        patch("topoguard_cli._make_source", return_value=Source()),
        patch("topoguard_cli._make_engine", return_value=Engine()),
        patch("topoguard_cli._make_notifier", return_value=notifier),
        patch("topoguard_cli._inspect", inspector),
    ):
        assert main([
            "monitor",
            "--source", "system",
            "--ticks", "1",
            "--interval", "0",
            "--notify",
        ]) == 0

    notifier.send_alert.assert_called_once_with(result, "test-source", 1)
    assert inspector.call_args.args[2:] == (1, [{"cpu": 1.0}])
    assert build_parser().parse_args(["audit", "--source", "jsonl", "--input", "metrics.jsonl"]).source == "jsonl"


def test_cli_monitor_persists_trained_baseline_snapshot():
    from hybrid_engine import IsolationForestBaseline

    class Source:
        name = "test-source"
        finished = True
        snapshot = True

        def collect(self):
            return [{"cpu": 1.0}]

        def stop(self):
            return None

    class Engine:
        def __init__(self):
            detector = IsolationForestBaseline(random_state=19)
            detector.evaluate(
                np.random.default_rng(19).normal(0, 0.1, size=(64, 2)),
                ("cpu", "memory"),
            )
            self.hybrid_ai = Mock(isolation_forest=detector)

        def stop(self):
            return None

    with tempfile.TemporaryDirectory() as directory:
        baseline_path = Path(directory) / "monitor-baseline.json"
        with (
            patch("topoguard_cli._make_source", return_value=Source()),
            patch("topoguard_cli._make_engine", return_value=Engine()),
            patch("topoguard_cli._inspect", return_value={
                "status": "SUCCESS",
                "is_anomaly": False,
                "source": "test-source",
                "tick": 1,
                "context": {},
            }),
        ):
            assert main([
                "monitor",
                "--ticks", "1",
                "--interval", "0",
                "--baseline", str(baseline_path),
            ]) == 0

        assert baseline_path.is_file()
        from baseline import BaselineCalibrator

        snapshot = BaselineCalibrator.load(baseline_path)
        assert snapshot.model.get_status()["fitted"]
        assert snapshot.model.random_state == 19


def test_cli_monitor_applies_backpressure_without_dropping_snapshots():
    from topoguard_cli import _monitor_async

    class Source:
        name = "fast-snapshot-source"
        snapshot = True

        def __init__(self):
            self.count = 0
            self.finished = False

        def collect(self):
            self.count += 1
            self.finished = self.count == 30
            return [{"cpu": float(self.count)}]

    processed_samples = []

    def inspect_slowly(_engine, _source, _tick, rows):
        import time

        time.sleep(0.01)
        processed_samples.append(rows[0]["cpu"])
        return {
            "status": "SUCCESS",
            "is_anomaly": False,
            "context": {},
        }

    with (
        patch("topoguard_cli._MONITOR_QUEUE_SIZE", 1),
        patch("topoguard_cli._inspect", side_effect=inspect_slowly),
    ):
        exit_code = asyncio.run(_monitor_async(
            SimpleNamespace(ticks=0, interval=0, tda_max_dim=1),
            Source(),
            object(),
            None,
            None,
        ))

    assert exit_code == 0
    assert processed_samples == [float(index) for index in range(1, 31)]


def test_cli_monitor_opens_circuit_breaker_after_five_invalid_samples():
    from topoguard_cli import _monitor_async

    class Source:
        name = "unstable-source"
        snapshot = True
        finished = False

        def __init__(self):
            self.calls = 0

        def collect(self):
            self.calls += 1
            raise ValueError("invalid metric value")

    source = Source()
    notifier = Mock()
    siem = Mock()

    exit_code = asyncio.run(_monitor_async(
        SimpleNamespace(ticks=0, interval=0, tda_max_dim=1),
        source,
        object(),
        notifier,
        siem,
    ))

    assert exit_code == 2
    assert source.calls >= 5
    notifier.send_alert.assert_called_once()
    alert = notifier.send_alert.call_args.args[0]
    assert alert["context"]["severity"] == "CRITICAL"
    assert alert["context"]["consecutive_invalid_samples"] == 5
    siem.emit.assert_called_once()


def test_cli_monitor_resets_invalid_sample_streak_after_recovery():
    from topoguard_cli import _monitor_async

    class Source:
        name = "recovering-source"
        snapshot = True
        finished = False

        def __init__(self):
            self.samples = iter([
                ValueError("bad sample 1"),
                ValueError("bad sample 2"),
                [{"cpu": 1.0}],
                ValueError("bad sample 3"),
                ValueError("bad sample 4"),
                [{"cpu": 2.0}],
            ])
            self.calls = 0

        def collect(self):
            self.calls += 1
            sample = next(self.samples)
            if self.calls == 6:
                self.finished = True
            if isinstance(sample, Exception):
                raise sample
            return sample

    notifier = Mock()
    with patch("topoguard_cli._inspect", return_value={
        "status": "SUCCESS",
        "is_anomaly": False,
        "context": {},
    }):
        exit_code = asyncio.run(_monitor_async(
            SimpleNamespace(ticks=0, interval=0, tda_max_dim=1),
            Source(),
            object(),
            notifier,
            None,
        ))

    assert exit_code == 0
    assert all(
        call.args[0]["context"].get("severity") != "CRITICAL"
        for call in notifier.send_alert.call_args_list
    )


def test_cli_infers_jsonl_replay_source():
    with tempfile.TemporaryDirectory() as directory:
        input_path = Path(directory) / "metrics.jsonl"
        input_path.write_text(
            "\n".join(
                json.dumps({"timestamp": index, "metrics": {"cpu": index * 10.0}})
                for index in range(1, 5)
            ) + "\n",
            encoding="utf-8",
        )
        result = main(["audit", "--input", str(input_path), "--window-size", "4"])
        assert result in (0, 1)


def test_cli_rejects_mock_as_a_runtime_source():
    try:
        build_parser().parse_args(["monitor", "--source", "mock"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("Runtime CLI must not accept synthetic data sources")


def test_cli_rejects_unknown_input_type():
    with tempfile.TemporaryDirectory() as directory:
        invalid = Path(directory) / "traffic.bin"
        invalid.touch()
        try:
            main(["audit", "--input", str(invalid)])
        except SystemExit as exc:
            assert exc.code == 2
        else:
            raise AssertionError("Unknown file types must be rejected")


def test_z3_security_boundary_proves_safe_and_violated_states():
    safe = Z3FormalVerificationPlugin.verify_security_boundary({
        "cpu": 75.0,
        "sockets": 12,
        "su_attempts": 0,
    })
    violated = Z3FormalVerificationPlugin.verify_security_boundary({
        "cpu": 95.0,
        "sockets": 12,
        "su_attempts": 1,
    })

    assert safe["is_violated"] is False
    assert safe["solver_status"] == "sat"
    assert violated["is_violated"] is True
    assert violated["solver_status"] == "unsat"


def test_z3_security_boundary_fails_closed_for_missing_telemetry():
    result = Z3FormalVerificationPlugin.verify_security_boundary({"cpu": 50})

    assert result["is_violated"] is True
    assert result["solver_status"] == "invalid"


def test_engine_activates_z3_only_above_tda_threshold():
    from defensive import TopoGuardEngine

    class HybridStub:
        score = 0.85

        def evaluate(self, _matrix, return_details=False, feature_names=None):
            return {
                "anomaly_score": self.score,
                "is_anomaly": True,
                "tda_details": {"h0_count": 4, "h1_count": 60},
            }

    class Z3Stub:
        calls = []

        @classmethod
        def verify_topological_invariants(cls, **kwargs):
            cls.calls.append(kwargs)
            return {"is_violated": False, "reason": "verified_safe"}

    engine = TopoGuardEngine(
        use_ebpf=False,
        use_polars=False,
        use_hybrid=False,
        use_z3=False,
        window_size=4,
        stride=2,
    )
    engine.hybrid_ai = HybridStub()
    engine.z3_guard = Z3Stub
    raw_stream = [[1.0], [2.0], [3.0], [4.0]]

    engine.inspect_tick(raw_stream)
    assert Z3Stub.calls == []

    engine.hybrid_ai.score = 0.86
    engine.inspect_tick(raw_stream)
    assert len(Z3Stub.calls) == 1
    assert Z3Stub.calls[0]["tda_threshold"] == 0.85


def test_production_engine_uses_isolation_forest_without_untrained_gnn():
    from defensive import TopoGuardEngine

    engine = TopoGuardEngine(
        use_ebpf=False,
        use_polars=False,
        use_hybrid=True,
        use_z3=False,
        window_size=4,
        stride=2,
    )
    assert engine.hybrid_ai.use_gnn is False
    assert engine.hybrid_ai.isolation_forest is not None


def test_export_tda_writes_intervals_and_betti_curves():
    with tempfile.TemporaryDirectory() as directory:
        input_path = Path(directory) / "metrics.csv"
        output_path = Path(directory) / "topology.json"
        input_path.write_text(
            "timestamp,cpu,memory\n1,10,25\n2,20,40\n3,30,35\n4,40,50\n",
            encoding="utf-8",
        )
        result = main([
            "export-tda",
            "--source", "csv",
            "--input", str(input_path),
            "--window-size", "4",
            "--batch-size", "4",
            "--max-points", "4",
            "--output", str(output_path),
        ])
        payload = json.loads(output_path.read_text(encoding="utf-8"))

        assert result == 0
        assert payload["sample_count"] == 4
        assert set(payload["betti_curves"]) == {"betti_0", "betti_1"}
        assert payload["max_dimension"] == 1
        assert payload["analyzed_sample_count"] == 4
        assert len(payload["persistence_intervals"]) > 0

        extended_output_path = Path(directory) / "topology-h2.json"
        assert main([
            "export-tda",
            "--source", "csv",
            "--input", str(input_path),
            "--window-size", "4",
            "--batch-size", "4",
            "--max-points", "4",
            "--tda-max-dim", "2",
            "--output", str(extended_output_path),
        ]) == 0
        extended_payload = json.loads(
            extended_output_path.read_text(encoding="utf-8")
        )
        assert set(extended_payload["betti_curves"]) == {
            "betti_0",
            "betti_1",
            "betti_2",
        }
        assert extended_payload["max_dimension"] == 2


def test_pcap_source_reads_real_packet_metadata():
    from scapy.all import IP, TCP, Raw, wrpcap

    with tempfile.TemporaryDirectory() as directory:
        capture_path = Path(directory) / "traffic.pcap"
        packets = [
            IP(src="192.0.2.1", dst="198.51.100.1") / TCP(sport=1000 + index, dport=443, flags="S") / Raw(b"x")
            for index in range(3)
        ]
        wrpcap(str(capture_path), packets)
        source = PcapSource(capture_path, batch_size=2, window_size=3)
        first = source.collect()
        second = source.collect()
        source.stop()

        assert len(first) == 2
        assert len(second) == 3
        assert all(row["tcp_syn"] == 1 for row in second)
        assert all(row["destination_port"] == 443 for row in second)


def test_audit_runs_on_real_access_log_entries():
    with tempfile.TemporaryDirectory() as directory:
        log_path = Path(directory) / "access.log"
        lines = [
            f'192.0.2.{index} - - [16/Oct/2026:00:00:{index:02d} +0000] '
            f'"GET /resource/{index} HTTP/1.1" 200 123\n'
            for index in range(1, 9)
        ]
        log_path.write_text("".join(lines), encoding="utf-8")
        result = main([
            "audit",
            "--input", str(log_path),
            "--window-size", "10",
        ])
        assert result in (0, 1)


def run_all_tests():
    tests = [
        test_cli_report_colors_normal_and_alert_statuses,
        test_cli_parses_production_commands_and_default_source,
        test_cli_audits_zeek_json_connection_log,
        test_webhook_notifier_sends_data_minimized_discord_and_slack_alerts,
        test_webhook_notifier_validates_secrets_and_surfaces_delivery_errors,
        test_cli_monitor_wires_opt_in_anomaly_notifications,
        test_cli_monitor_persists_trained_baseline_snapshot,
        test_cli_monitor_applies_backpressure_without_dropping_snapshots,
        test_cli_monitor_opens_circuit_breaker_after_five_invalid_samples,
        test_cli_monitor_resets_invalid_sample_streak_after_recovery,
        test_cli_infers_jsonl_replay_source,
        test_cli_rejects_mock_as_a_runtime_source,
        test_cli_rejects_unknown_input_type,
        test_z3_security_boundary_proves_safe_and_violated_states,
        test_z3_security_boundary_fails_closed_for_missing_telemetry,
        test_engine_activates_z3_only_above_tda_threshold,
        test_production_engine_uses_isolation_forest_without_untrained_gnn,
        test_export_tda_writes_intervals_and_betti_curves,
        test_pcap_source_reads_real_packet_metadata,
        test_audit_runs_on_real_access_log_entries,
    ]
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")


if __name__ == "__main__":
    run_all_tests()
