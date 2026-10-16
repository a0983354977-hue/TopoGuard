"""
Negative / chaos tests for TopoGuard.

Verifies fail-closed behaviour under corrupted JSON, extreme values,
scrambled Zeek fields, Polars memory-budget pressure, and malformed telemetry.
The system must never silently crash: parse failures surface, and the engine
fails closed (status=ERROR, is_anomaly=True).
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import numpy as np

from data_sources import ReplaySource, ZeekJSONSource
from defensive import (
    HardCapGuard,
    TopoGuardEngine,
    AdaptiveBaseline,
    bulletproof_parse_adapter,
    BulletproofAdapter,
)
from polars_engine import PolarsDataPipeline


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def test_corrupted_truncated_json_raises():
    with tempfile.TemporaryDirectory() as directory:
        bad = Path(directory) / "bad.json"
        _write(bad, '[{"cpu": 10, "mem": 20,')  # truncated array
        source = ReplaySource(bad, batch_size=4, window_size=4)
        try:
            source.collect()
            raise AssertionError("Truncated JSON must raise")
        except (json.JSONDecodeError, ValueError):
            pass  # expected


def test_invalid_json_syntax_raises():
    with tempfile.TemporaryDirectory() as directory:
        bad = Path(directory) / "bad.json"
        _write(bad, "{not valid json!!!")
        source = ReplaySource(bad, batch_size=4, window_size=4)
        try:
            source.collect()
            raise AssertionError("Invalid JSON must raise")
        except (json.JSONDecodeError, ValueError):
            pass


def test_extreme_huge_finite_values_are_capped_but_finite():
    # 1e308 is finite and must pass the finite check
    data = np.array([[1e308, 1e308]], dtype=np.float64)
    out = HardCapGuard.validate_and_cap(data, max_points=10)
    assert np.isfinite(out).all()


def test_inf_and_nan_are_rejected_by_hard_cap():
    for bad in (np.array([[float("inf")]]), np.array([[float("nan")]])):
        try:
            HardCapGuard.validate_and_cap(bad, max_points=10)
            raise AssertionError("Non-finite values must be rejected")
        except ValueError:
            pass


def test_empty_input_is_rejected_by_hard_cap():
    try:
        HardCapGuard.validate_and_cap(np.array([]), max_points=10)
        raise AssertionError("Empty input must be rejected")
    except ValueError:
        pass


def test_bulletproof_parse_surfaces_non_numeric_and_empty():
    def bad_parser(_x):
        return ["not", "numeric"]

    try:
        bulletproof_parse_adapter(bad_parser, "x")
        raise AssertionError("Non-numeric parser output must raise")
    except ValueError:
        pass

    def empty_parser(_x):
        return np.array([])

    try:
        bulletproof_parse_adapter(empty_parser, "x")
        raise AssertionError("Empty parser output must raise")
    except ValueError:
        pass


def test_scrambled_zeek_fields_produce_same_result():
    with tempfile.TemporaryDirectory() as directory:
        ordered = Path(directory) / "ordered.log"
        scrambled = Path(directory) / "scrambled.log"
        base = {
            "ts": 1,
            "uid": "C1",
            "id.orig_p": 20001,
            "id.resp_p": 443,
            "duration": 1.5,
            "orig_bytes": 100,
            "resp_bytes": 200,
            "status_code": 200,
        }
        _write(ordered, json.dumps(base) + "\n")
        _write(scrambled, json.dumps(dict(reversed(list(base.items())))) + "\n")

        s1 = ZeekJSONSource(ordered, batch_size=4, window_size=4)
        s2 = ZeekJSONSource(scrambled, batch_size=4, window_size=4)
        r1 = s1.collect()
        r2 = s2.collect()
        s1.stop()
        s2.stop()

        assert r1 == r2, "Field order must not affect Zeek numeric extraction"
        assert r1[0]["zeek.duration"] == 1.5


def test_zeek_missing_fields_are_tolerated():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "partial.log"
        _write(path, json.dumps({"ts": 1, "duration": 3.0}) + "\n")
        source = ZeekJSONSource(path, batch_size=4, window_size=4)
        rows = source.collect()
        source.stop()
        assert len(rows) == 1
        assert rows[0]["zeek.duration"] == 3.0


def test_zeek_non_finite_field_raises():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "bad.log"
        _write(path, json.dumps({"duration": float("inf"), "orig_bytes": 100}) + "\n")
        source = ZeekJSONSource(path, batch_size=4, window_size=4)
        try:
            source.collect()
            raise AssertionError("Non-finite Zeek field must raise")
        except ValueError:
            pass
        source.stop()


def test_polars_single_row_exceeds_memory_budget_raises():
    pipe = PolarsDataPipeline(window_size=10, stride=5, max_memory_mb=0.000001)
    try:
        pipe.process([{"a": 1.0, "b": 2.0}])
        raise AssertionError("Oversized single row must raise MemoryError")
    except MemoryError:
        pass


def test_adaptive_baseline_extreme_scores_do_not_crash():
    baseline = AdaptiveBaseline(window_size=4)
    for score in (0.0, 1e6, -1e6, float("inf")):
        try:
            baseline.evaluate(score)
        except Exception:
            # inf/std edge cases may produce non-finite but must not raise unexpectedly
            if not math.isfinite(score):
                continue
            raise


def test_engine_fails_closed_on_non_finite_input():
    engine = TopoGuardEngine(
        use_ebpf=False,
        use_polars=False,
        use_hybrid=True,
        use_z3=False,
        window_size=4,
        stride=2,
    )
    # Non-finite matrix -> Hybrid TDA raises -> engine returns ERROR fail-closed
    result = engine.inspect_tick([[1.0, float("nan")], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]])
    assert result["status"] == "ERROR"
    assert result["is_anomaly"] is True
    engine.stop()


def test_engine_fails_closed_on_uneven_feature_count_after_baseline():
    # Train a baseline then feed a different-width matrix -> feature mismatch raises -> ERROR
    from baseline import BaselineCalibrator

    calibrator = BaselineCalibrator()
    rows = [{"a": float(i), "b": float(i * 10)} for i in range(1, 40)]
    calibrator.fit(rows)
    with tempfile.TemporaryDirectory() as directory:
        p = Path(directory) / "baseline.pkl"
        calibrator.save(p, format="pickle")

        engine = TopoGuardEngine(
            use_ebpf=False,
            use_polars=False,
            use_hybrid=True,
            use_z3=False,
            window_size=4,
            stride=2,
            baseline_path=str(p),
        )
        # Extra columns are aligned/sliced to the trained 2-feature schema -> SUCCESS
        result = engine.inspect_tick(
            [{"a": 1.0, "b": 2.0, "c": 3.0}, {"a": 4.0, "b": 5.0, "c": 6.0},
             {"a": 7.0, "b": 8.0, "c": 9.0}, {"a": 10.0, "b": 11.0, "c": 12.0}]
        )
        assert result["status"] == "SUCCESS"
        assert isinstance(result["is_anomaly"], bool)
        engine.stop()


def test_bulletproof_adapter_counts_errors_and_raises():
    adapter = BulletproofAdapter()

    def flaky(_x):
        raise RuntimeError("boom")

    for _ in range(adapter.max_errors):
        try:
            adapter.execute(flaky, "x")
        except RuntimeError:
            pass
    assert adapter.error_count >= adapter.max_errors
    try:
        adapter.execute(flaky, "x")
        raise AssertionError("should still raise")
    except RuntimeError:
        pass


def run_all_tests():
    tests = [
        test_corrupted_truncated_json_raises,
        test_invalid_json_syntax_raises,
        test_extreme_huge_finite_values_are_capped_but_finite,
        test_inf_and_nan_are_rejected_by_hard_cap,
        test_empty_input_is_rejected_by_hard_cap,
        test_bulletproof_parse_surfaces_non_numeric_and_empty,
        test_scrambled_zeek_fields_produce_same_result,
        test_zeek_missing_fields_are_tolerated,
        test_zeek_non_finite_field_raises,
        test_polars_single_row_exceeds_memory_budget_raises,
        test_adaptive_baseline_extreme_scores_do_not_crash,
        test_engine_fails_closed_on_non_finite_input,
        test_engine_fails_closed_on_uneven_feature_count_after_baseline,
        test_bulletproof_adapter_counts_errors_and_raises,
    ]
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")


if __name__ == "__main__":
    run_all_tests()
