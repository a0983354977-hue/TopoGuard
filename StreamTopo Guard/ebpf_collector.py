"""
eBPF (BCC) Telemetry Collector
--------------------------------
Optional plugin for real-time kernel-level data collection.
Responsibilities:
- Capture system calls and network packets via eBPF
- Push kernel events to Polars pipeline
- Graceful degradation on non-Linux or no-root environments

STRICT CONSTRAINTS:
- ONLY data collection (NO processing, NO analysis)
- Optional plugin (system works without it)
- Single-directional flow: eBPF -> Polars
- Never generate synthetic telemetry
"""

import platform
import logging
from collections import deque
from typing import List, Dict, Any, Optional
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EBPFCollector:
    """
    eBPF-based kernel telemetry collector.
    """

    def __init__(
        self,
        require_real: bool = False,
    ):
        """
        Initialize eBPF collector.

        Args:
            require_real: Raise during initialization if eBPF is unavailable
        """
        self.require_real = require_real
        self._is_linux = platform.system() == "Linux"
        self._has_root = self._check_root_permission()
        self._bcc_available = self._check_bcc_available()
        self._is_active = False
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._event_lock = threading.Lock()
        self._event_buffer: deque[Dict[str, Any]] = deque(maxlen=10_000)
        self._collector_error: Optional[str] = None

        unavailable_reason = None
        if not self._is_linux:
            unavailable_reason = "eBPF syscall collection requires Linux"
        elif not self._has_root:
            unavailable_reason = "eBPF syscall collection requires root privileges"
        elif not self._bcc_available:
            unavailable_reason = "eBPF syscall collection requires the BCC Python bindings"
        self._unavailable_reason = unavailable_reason
        if require_real and unavailable_reason:
            raise RuntimeError(unavailable_reason)

        self._actual_mode = "real" if unavailable_reason is None else "unavailable"
        if unavailable_reason:
            logger.warning("eBPF unavailable: %s", unavailable_reason)

    def _check_root_permission(self) -> bool:
        """Check if running with root privileges."""
        try:
            import os
            return os.geteuid() == 0 if hasattr(os, 'geteuid') else False
        except Exception:
            return False

    def _check_bcc_available(self) -> bool:
        """Check if BCC module is available."""
        try:
            import bcc
            return True
        except ImportError:
            return False

    def start(self) -> None:
        """Start eBPF collection thread."""
        if self._actual_mode == "unavailable":
            raise RuntimeError("eBPF is unavailable; use a live system metrics source instead")
        if self._is_active:
            logger.warning("eBPF collector already active")
            return

        self._stop_event.clear()
        self._is_active = True

        self._thread = threading.Thread(target=self._real_ebpf_loop, daemon=True)

        self._thread.start()
        logger.info(f"eBPF collector started in {self._actual_mode} mode")

    def stop(self) -> None:
        """Stop eBPF collection thread."""
        if not self._is_active and self._thread is None:
            return

        self._stop_event.set()
        self._is_active = False

        if self._thread:
            if self._thread is not threading.current_thread():
                self._thread.join(timeout=2.0)
            self._thread = None

        logger.info("eBPF collector stopped")

    def poll(self) -> List[Dict[str, Any]]:
        """
        Poll for collected events.
        Returns buffered events collected from the kernel.

        Returns:
            List of event dictionaries
        """
        if self._collector_error:
            raise RuntimeError(f"eBPF collector stopped: {self._collector_error}")
        if self._actual_mode == "unavailable":
            raise RuntimeError(f"eBPF is unavailable: {self._unavailable_reason}")
        return self._poll_real_events()

    def _real_ebpf_loop(self) -> None:
        """
        Real eBPF collection loop.
        Uses BCC to attach to kernel probes.
        """
        try:
            from bcc import BPF

            # Trace process execution and outbound connection system calls.
            bpf_code = """
            #include <uapi/linux/ptrace.h>
            #include <linux/sched.h>

            struct event_t {
                u32 pid;
                char comm[TASK_COMM_LEN];
                u64 timestamp;
                u64 bytes;
                u32 event_type;
            };

            BPF_PERF_OUTPUT(events);

            TRACEPOINT_PROBE(syscalls, sys_enter_execve) {
                struct event_t event = {};
                event.pid = bpf_get_current_pid_tgid() >> 32;
                bpf_get_current_comm(&event.comm, sizeof(event.comm));
                event.timestamp = bpf_ktime_get_ns();
                event.event_type = 1;
                events.perf_submit(args, &event, sizeof(event));
                return 0;
            }

            TRACEPOINT_PROBE(syscalls, sys_enter_connect) {
                struct event_t event = {};
                event.pid = bpf_get_current_pid_tgid() >> 32;
                bpf_get_current_comm(&event.comm, sizeof(event.comm));
                event.timestamp = bpf_ktime_get_ns();
                event.event_type = 2;
                events.perf_submit(args, &event, sizeof(event));
                return 0;
            }

            struct sock;
            struct msghdr;

            int trace_tcp_sendmsg(struct pt_regs *ctx, struct sock *sk,
                                  struct msghdr *msg, size_t size) {
                struct event_t event = {};
                event.pid = bpf_get_current_pid_tgid() >> 32;
                bpf_get_current_comm(&event.comm, sizeof(event.comm));
                event.timestamp = bpf_ktime_get_ns();
                event.bytes = size;
                event.event_type = 3;
                events.perf_submit(ctx, &event, sizeof(event));
                return 0;
            }
            """

            bpf = BPF(text=bpf_code)
            bpf.attach_kprobe(event="tcp_sendmsg", fn_name="trace_tcp_sendmsg")

            def handle_event(cpu, data, size):
                event = bpf["events"].event(data)
                record = {
                    'pid': event.pid,
                    'comm': event.comm.decode("utf-8", errors="replace").rstrip("\x00"),
                    'timestamp': event.timestamp,
                    'bytes': event.bytes,
                    'event_type': {
                        1: 'execve',
                        2: 'connect',
                        3: 'tcp_sendmsg',
                    }.get(event.event_type, 'unknown')
                }
                with self._event_lock:
                    self._event_buffer.append(record)

            bpf["events"].open_perf_buffer(handle_event)

            while not self._stop_event.is_set():
                try:
                    bpf.perf_buffer_poll(timeout=100)
                except Exception as e:
                    raise RuntimeError(f"eBPF perf-buffer poll failed: {e}") from e

        except Exception as e:
            self._collector_error = str(e)
            self._is_active = False
            logger.exception("Real eBPF collection failed; refusing synthetic fallback")

    def _poll_real_events(self) -> List[Dict[str, Any]]:
        """Poll real eBPF events from buffer."""
        with self._event_lock:
            events = list(self._event_buffer)
            self._event_buffer.clear()
        return events

    def get_status(self) -> Dict[str, Any]:
        """Get collector status."""
        return {
            'mode': self._actual_mode,
            'is_active': self._is_active,
            'is_linux': self._is_linux,
            'has_root': self._has_root,
            'bcc_available': self._bcc_available,
            'unavailable_reason': self._unavailable_reason,
            'collector_error': self._collector_error,
        }

    def reset(self) -> None:
        """Reset collector state."""
        if hasattr(self, '_event_buffer'):
            with self._event_lock:
                self._event_buffer.clear()
        logger.info("eBPF collector reset")


class NetworkPacketCollector:
    """
    Live packet capture is intentionally unavailable; use the offline PCAP source.
    """

    def __init__(self):
        self._is_linux = platform.system() == "Linux"
        self._actual_mode = "unavailable"
        self._unavailable_reason = (
            "Live packet capture is not implemented; use offline PCAP audit"
        )
        self._is_active = False

    def start(self) -> None:
        raise RuntimeError(self._unavailable_reason)

    def stop(self) -> None:
        self._is_active = False

    def poll(self) -> List[Dict[str, Any]]:
        raise RuntimeError(self._unavailable_reason)

    def get_status(self) -> Dict[str, Any]:
        """Get collector status."""
        return {
            'mode': self._actual_mode,
            'is_active': self._is_active,
            'is_linux': self._is_linux,
            'unavailable_reason': self._unavailable_reason,
        }
