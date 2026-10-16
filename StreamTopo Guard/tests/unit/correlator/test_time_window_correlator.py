"""Tests for the Time-Window Multi-Source Event Correlator."""

from correlator import TimeWindowCorrelator

BASE = 1_000_000.0  # fixed wall-clock anchor; bucket = ts // window_seconds


def _ebpf(ts, execve_count=5, unique_processes=3, connection_count=2, network_bytes=1024):
    return {
        "execve_count": execve_count,
        "unique_processes": unique_processes,
        "connection_count": connection_count,
        "network_bytes": network_bytes,
    }


def _prometheus(ts, cpu_percent=90.0, memory_percent=70.0):
    return {"cpu_percent": cpu_percent, "memory_percent": memory_percent}


def _zeek(ts, connection_count=10, orig_bytes=5000.0, resp_bytes=3000.0):
    return {"connection_count": connection_count, "orig_bytes": orig_bytes, "resp_bytes": resp_bytes}


def test_empty_correlator_returns_no_correlation():
    c = TimeWindowCorrelator()
    ctx = c.correlate()
    assert ctx["correlated"] is False
    assert ctx["bucket_count"] == 0
    assert ctx["severity"] == "NONE"
    assert c.correlate_latest() is ctx


def test_correlates_ebpf_execve_prometheus_cpu_and_zeek_in_one_window():
    c = TimeWindowCorrelator(window_seconds=60.0)
    c.ingest("ebpf", BASE + 5, **_ebpf(BASE + 5))
    c.ingest("prometheus", BASE + 10, **_prometheus(BASE + 10))
    c.ingest("zeek", BASE + 15, **_zeek(BASE + 15))

    ctx = c.correlate()
    assert ctx["correlated"] is True
    assert ctx["bucket_count"] == 1
    assert ctx["severity"] == "HIGH"  # cpu 90 > 85
    b = ctx["buckets"][0]
    assert set(b["sources"]) == {"ebpf", "prometheus", "zeek"}
    assert b["ebpf"]["execve_count"] == 5
    assert b["prometheus"]["cpu_percent"] == 90.0
    assert b["zeek"]["connection_count"] == 10
    assert "attack window" in ctx["summary"]
    assert c.correlate_latest() is ctx


def test_missing_one_source_is_not_correlated():
    c = TimeWindowCorrelator(window_seconds=60.0)
    c.ingest("ebpf", BASE + 5, **_ebpf(BASE + 5))
    c.ingest("prometheus", BASE + 10, **_prometheus(BASE + 10))
    # no zeek
    ctx = c.correlate()
    assert ctx["correlated"] is False
    assert ctx["bucket_count"] == 0


def test_cpu_below_threshold_is_not_correlated():
    c = TimeWindowCorrelator(window_seconds=60.0)
    c.ingest("ebpf", BASE + 5, **_ebpf(BASE + 5))
    c.ingest("prometheus", BASE + 10, **_prometheus(BASE + 10, cpu_percent=50.0))
    c.ingest("zeek", BASE + 15, **_zeek(BASE + 15))
    ctx = c.correlate()
    assert ctx["correlated"] is False


def test_execve_zero_is_not_correlated():
    c = TimeWindowCorrelator(window_seconds=60.0)
    c.ingest("ebpf", BASE + 5, **_ebpf(BASE + 5, execve_count=0))
    c.ingest("prometheus", BASE + 10, **_prometheus(BASE + 10))
    c.ingest("zeek", BASE + 15, **_zeek(BASE + 15))
    ctx = c.correlate()
    assert ctx["correlated"] is False


def test_events_in_different_windows_are_not_correlated():
    c = TimeWindowCorrelator(window_seconds=60.0)
    c.ingest("ebpf", BASE + 5, **_ebpf(BASE + 5))           # bucket BASE//60
    c.ingest("prometheus", BASE + 100, **_prometheus(BASE + 100))  # different bucket
    c.ingest("zeek", BASE + 110, **_zeek(BASE + 110))
    ctx = c.correlate()
    assert ctx["correlated"] is False


def test_cpu_above_95_is_critical():
    c = TimeWindowCorrelator(window_seconds=60.0)
    c.ingest("ebpf", BASE + 5, **_ebpf(BASE + 5))
    c.ingest("prometheus", BASE + 10, **_prometheus(BASE + 10, cpu_percent=98.0))
    c.ingest("zeek", BASE + 15, **_zeek(BASE + 15))
    ctx = c.correlate()
    assert ctx["correlated"] is True
    assert ctx["buckets"][0]["severity"] == "CRITICAL"


def test_ingest_batch_and_reset():
    c = TimeWindowCorrelator(window_seconds=60.0)
    c.ingest_batch(
        "ebpf",
        [_ebpf(BASE + 5), _ebpf(BASE + 6)],
        timestamp=BASE + 5,
    )
    c.ingest("prometheus", BASE + 10, **_prometheus(BASE + 10))
    c.ingest("zeek", BASE + 15, **_zeek(BASE + 15))
    ctx = c.correlate()
    assert ctx["correlated"] is True
    assert ctx["buckets"][0]["ebpf"]["execve_count"] == 5

    c.reset()
    assert c.correlate_latest() is None
    assert c.correlate()["correlated"] is False


def run_all_tests():
    tests = [
        test_empty_correlator_returns_no_correlation,
        test_correlates_ebpf_execve_prometheus_cpu_and_zeek_in_one_window,
        test_missing_one_source_is_not_correlated,
        test_cpu_below_threshold_is_not_correlated,
        test_execve_zero_is_not_correlated,
        test_events_in_different_windows_are_not_correlated,
        test_cpu_above_95_is_critical,
        test_ingest_batch_and_reset,
    ]
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")


if __name__ == "__main__":
    run_all_tests()
