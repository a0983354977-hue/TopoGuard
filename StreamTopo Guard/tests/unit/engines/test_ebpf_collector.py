"""Tests that kernel collectors report availability without fabricating telemetry."""

from ebpf_collector import EBPFCollector, NetworkPacketCollector


def test_ebpf_never_generates_synthetic_events():
    collector = EBPFCollector()
    status = collector.get_status()

    assert status["mode"] in {"real", "unavailable"}
    assert "mock_frequency" not in status
    assert "mock" not in status["mode"]

    if status["mode"] == "unavailable":
        try:
            collector.poll()
        except RuntimeError as exc:
            assert status["unavailable_reason"] in str(exc)
        else:
            raise AssertionError("Unavailable eBPF must report an error")
    else:
        assert collector.poll() == []


def test_require_real_fails_when_ebpf_is_unavailable():
    collector = EBPFCollector()
    if collector.get_status()["mode"] == "unavailable":
        try:
            EBPFCollector(require_real=True)
        except RuntimeError as exc:
            assert collector.get_status()["unavailable_reason"] in str(exc)
        else:
            raise AssertionError("require_real must reject unavailable eBPF")


def test_ebpf_event_buffer_is_bounded():
    collector = EBPFCollector()

    assert collector._event_buffer.maxlen == 10_000
    collector._event_buffer.append({"event_type": "execve"})
    assert collector._poll_real_events() == [{"event_type": "execve"}]
    assert collector._poll_real_events() == []


def test_live_packet_collector_reports_not_implemented():
    collector = NetworkPacketCollector()
    status = collector.get_status()

    assert status["mode"] == "unavailable"
    assert "offline PCAP audit" in status["unavailable_reason"]
    for operation in (collector.start, collector.poll):
        try:
            operation()
        except RuntimeError as exc:
            assert status["unavailable_reason"] in str(exc)
        else:
            raise AssertionError("Unimplemented packet capture must fail explicitly")


def run_all_tests():
    tests = [
        test_ebpf_never_generates_synthetic_events,
        test_require_real_fails_when_ebpf_is_unavailable,
        test_ebpf_event_buffer_is_bounded,
        test_live_packet_collector_reports_not_implemented,
    ]
    failures = []
    for test in tests:
        try:
            test()
            print(f"[PASS] {test.__name__}")
        except Exception as exc:
            failures.append((test.__name__, exc))
            print(f"[FAIL] {test.__name__}: {exc}")

    if failures:
        raise AssertionError(f"{len(failures)} eBPF collector test(s) failed")


if __name__ == "__main__":
    run_all_tests()
