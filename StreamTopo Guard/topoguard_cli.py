"""Production-oriented command line interface for TopoGuard."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("topoguard")
_MONITOR_QUEUE_SIZE = 8
_BASELINE_SNAPSHOT_INTERVAL = 60 * 60
_MONITOR_INVALID_SAMPLE_LIMIT = 5
_MONITOR_DONE = object()


def _console():
    try:
        from rich.console import Console
        return Console()
    except ImportError:
        return None


def _report(title: str, values: dict[str, Any]) -> None:
    status = str(values.get("status", "")).upper()
    severity = str(values.get("severity", "")).upper()
    is_alert = (
        bool(values.get("anomaly"))
        or status in {"ALERT", "ERROR"}
        or severity in {"ALERT", "HIGH", "CRITICAL"}
    )
    style = (
        "bold red"
        if is_alert
        else "yellow"
        if status in {"WARMING_UP", "NO_DATA"}
        else "green"
    )

    console = _console()
    if console is None:
        rows = [f"{key}: {value}" for key, value in values.items()]
        width = max([len(title), *(len(row) for row in rows)], default=len(title))
        border_char = "#" if is_alert else "-"
        border = f"+{border_char * (width + 2)}+"
        color_code = (
            "1;31"
            if is_alert
            else "33"
            if status in {"WARMING_UP", "NO_DATA"}
            else "32"
        )
        color = f"\033[{color_code}m" if sys.stdout.isatty() else ""
        reset = "\033[0m" if color else ""

        print(f"\n{color}{border}")
        print(f"| {title:<{width}} |")
        print(border)
        for row in rows:
            print(f"| {row:<{width}} |")
        print(f"{border}{reset}")
        return

    from rich import box
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    table = Table.grid(padding=(0, 1))
    table.add_column(style="cyan")
    table.add_column()
    for key, value in values.items():
        table.add_row(Text(str(key)), Text(str(value)))
    console.print(Panel(
        table,
        title=Text(title, style=style),
        border_style=style,
        box=box.HEAVY if is_alert else box.ROUNDED,
        padding=(0, 1),
    ))


def _make_source(args):
    from data_sources import create_data_source

    return create_data_source(
        args.source,
        input_path=getattr(args, "input", None),
        window_size=args.window_size,
        batch_size=args.batch_size,
        poll_interval=args.interval,
    )


def _make_notifier(args):
    if not getattr(args, "notify", False):
        return None
    from notifications import WebhookNotifier

    return WebhookNotifier.from_environment()


def _make_siem(args):
    spec = getattr(args, "siem", None)
    if not spec:
        return None
    from siem import siem_emitter_from_spec

    return siem_emitter_from_spec(spec)


def _is_alert(result: dict[str, Any]) -> bool:
    if result.get("is_anomaly"):
        return True
    if result.get("status") == "ERROR":
        return True
    context = result.get("context")
    if isinstance(context, dict):
        for key in ("z3_result", "security_boundary"):
            verification = context.get(key)
            if isinstance(verification, dict) and verification.get("is_violated"):
                return True
    return False


def _emit_siem(emitter, result, source, tick):
    if emitter is None or not _is_alert(result):
        return
    context = result.get("context")
    if not isinstance(context, dict):
        context = {}
    try:
        score = float(result.get("anomaly_score", context.get("anomaly_score", 0.0)))
    except (TypeError, ValueError):
        score = 0.0
    severity = str(context.get("severity", result.get("severity", "UNKNOWN")))
    message = (
        f"TopoGuard alert | source={source} tick={tick} "
        f"score={score:.4f} severity={severity}"
    )
    from siem import build_alert

    alert = build_alert(
        source=source,
        tick=tick,
        result=result,
        message=message,
    )
    try:
        emitter.emit(alert)
    except Exception as exc:
        logger.error("SIEM emission failed: %s", exc)



def _make_engine(args):
    from defensive import TopoGuardEngine

    return TopoGuardEngine(
        use_ebpf=False,
        use_polars=not args.no_polars,
        use_hybrid=True,
        use_z3=True,
        window_size=args.window_size,
        stride=max(1, args.window_size // 2),
        baseline_path=getattr(args, "baseline", None),
        tda_max_dimension=args.tda_max_dim,
    )


def _inspect(
    engine,
    source,
    tick: int,
    rows: list[dict[str, float]] | None = None,
) -> dict[str, Any]:
    if rows is None:
        rows = source.collect()
    if not isinstance(rows, list):
        raise RuntimeError(
            f"Data source {source.name} returned an invalid sample collection"
        )
    minimum = getattr(source, "min_samples", 1)
    if len(rows) < minimum:
        return {
            "status": "WARMING_UP" if rows else "NO_DATA",
            "is_anomaly": False,
            "data_points": len(rows),
            "message": f"Collecting samples ({len(rows)}/{minimum})" if rows else "No new data",
            "source": source.name,
            "tick": tick,
        }
    result = engine.inspect_tick(
        rows,
        replace_stream=getattr(source, "snapshot", False),
        source_name=source.name,
    )
    result.update({"source": source.name, "tick": tick})
    return result


def _monitor_baseline(engine):
    hybrid = getattr(engine, "hybrid_ai", None)
    detector = getattr(hybrid, "isolation_forest", None)
    if detector is None or not detector.get_status()["fitted"]:
        return None

    calibrator = getattr(engine, "baseline_calibrator", None)
    if calibrator is None or calibrator.model is not detector:
        from baseline import BaselineCalibrator

        calibrator = BaselineCalibrator.from_detector(detector)
    return calibrator


def _save_monitor_baseline(engine, path: str) -> float | None:
    calibrator = _monitor_baseline(engine)
    if calibrator is None:
        return None
    output_format = "json" if Path(path).suffix.lower() == ".json" else "pickle"
    calibrator.save(path, format=output_format)
    return calibrator.trained_at or time.time()


async def _monitor_async(args, source, engine, notifier, siem) -> int:
    from data_sources import TransientDataSourceError

    queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue(maxsize=_MONITOR_QUEUE_SIZE)
    consecutive_invalid_samples = 0

    async def enqueue(kind: str, payload: Any) -> None:
        await queue.put((kind, payload))

    async def record_invalid_sample(exc: Exception, alert_tick: int) -> bool:
        nonlocal consecutive_invalid_samples
        consecutive_invalid_samples += 1
        logger.warning(
            "Skipping invalid monitor sample (%s/%s consecutive): %s",
            consecutive_invalid_samples,
            _MONITOR_INVALID_SAMPLE_LIMIT,
            exc,
        )
        if consecutive_invalid_samples < _MONITOR_INVALID_SAMPLE_LIMIT:
            return False

        result = {
            "status": "ERROR",
            "is_anomaly": True,
            "anomaly_score": 1.0,
            "source": source.name,
            "tick": alert_tick,
            "message": "Circuit breaker opened after consecutive invalid samples",
            "context": {
                "severity": "CRITICAL",
                "reason": "invalid_data_circuit_breaker",
                "consecutive_invalid_samples": consecutive_invalid_samples,
            },
        }
        _report("Monitor circuit breaker", {
            "status": "ERROR",
            "severity": "CRITICAL",
            "consecutive invalid samples": consecutive_invalid_samples,
            "details": result["message"],
        })
        if notifier is not None:
            await asyncio.to_thread(
                notifier.send_alert, result, source.name, alert_tick
            )
        if siem is not None:
            await asyncio.to_thread(
                _emit_siem, siem, result, source.name, alert_tick
            )
        return True

    async def collect() -> None:
        try:
            while True:
                try:
                    rows = await asyncio.to_thread(source.collect)
                except (ValueError, OSError, TransientDataSourceError) as exc:
                    await enqueue("invalid", exc)
                    if getattr(source, "finished", False):
                        await enqueue("done", _MONITOR_DONE)
                        return
                    await asyncio.sleep(args.interval)
                    continue
                except Exception as exc:
                    await enqueue("error", exc)
                    return
                if not isinstance(rows, list):
                    await enqueue(
                        "error",
                        RuntimeError(
                            f"Data source {source.name} returned an invalid sample collection"
                        ),
                    )
                    return

                await enqueue("sample", rows)
                if getattr(source, "finished", False):
                    await enqueue("done", _MONITOR_DONE)
                    return
                await asyncio.sleep(args.interval)
        except asyncio.CancelledError:
            raise

    producer = asyncio.create_task(collect())
    tick = 0
    last_snapshot_at = 0.0
    last_snapshot_training_time = None
    try:
        while args.ticks == 0 or tick < args.ticks:
            kind, payload = await queue.get()
            queue.task_done()
            if kind == "done":
                break
            if kind == "error":
                raise payload
            if kind == "invalid":
                if await record_invalid_sample(payload, tick + 1):
                    return 2
                continue

            tick += 1
            try:
                result = await asyncio.to_thread(_inspect, engine, source, tick, payload)
            except ValueError as exc:
                if await record_invalid_sample(exc, tick):
                    return 2
                continue
            consecutive_invalid_samples = 0
            context = result.get("context")
            if not isinstance(context, dict):
                context = {}
            tda_details = context.get("tda_details")
            if not isinstance(tda_details, dict):
                tda_details = {}
            iforest_details = context.get("iforest_details")
            if not isinstance(iforest_details, dict):
                iforest_details = {}
            if iforest_details.get("status") == "warming_up":
                if result.get("status") == "SUCCESS":
                    result["status"] = "WARMING_UP"
                    result["message"] = (
                        "Isolation Forest baseline is warming up "
                        f"({context.get('baseline_samples', 0)}/"
                        f"{context.get('required_samples', 0)} samples)"
                    )

            _report(f"Monitor tick {tick}", {
                "source": result.get("source"),
                "status": result.get("status"),
                "anomaly": result.get("is_anomaly"),
                "severity": context.get("severity", "-"),
                "score": f"{result['anomaly_score']:.4f}" if "anomaly_score" in result else "-",
                "points": result.get("data_points", 0),
                "H0 features": tda_details.get("h0_count", "-"),
                "H1 features": tda_details.get("h1_count", "-"),
                "H2 features": (
                    tda_details.get("h2_count", "-")
                    if args.tda_max_dim >= 2
                    else "not computed"
                ),
                "Isolation Forest": (
                    iforest_details.get("status", "-")
                ),
                "IF score": (
                    f"{context['iforest_score']:.4f}"
                    if "iforest_score" in context else "-"
                ),
                "details": result.get("message", ""),
            })
            if notifier is not None:
                delivered = await asyncio.to_thread(
                    notifier.send_alert, result, source.name, tick
                )
                if delivered:
                    _report("Webhook notification", {"status": "delivered"})
            if siem is not None and _is_alert(result):
                await asyncio.to_thread(_emit_siem, siem, result, source.name, tick)

            baseline_path = getattr(args, "baseline", None)
            if baseline_path:
                trained_at = getattr(
                    getattr(getattr(engine, "hybrid_ai", None), "isolation_forest", None),
                    "_trained_at",
                    None,
                )
                now = time.monotonic()
                if trained_at is not None and (
                    last_snapshot_training_time != trained_at
                    or now - last_snapshot_at >= _BASELINE_SNAPSHOT_INTERVAL
                ):
                    saved_training_time = await asyncio.to_thread(
                        _save_monitor_baseline, engine, baseline_path
                    )
                    if saved_training_time is not None:
                        last_snapshot_at = now
                        last_snapshot_training_time = trained_at
                        _report("Baseline snapshot", {
                            "status": "saved",
                            "path": baseline_path,
                        })

            if result.get("status") == "ERROR":
                return 2
        return 0
    finally:
        producer.cancel()
        await asyncio.gather(producer, return_exceptions=True)


def _monitor(args) -> int:
    source = None
    engine = None
    notifier = None
    siem = None
    try:
        source = _make_source(args)
        engine = _make_engine(args)
        notifier = _make_notifier(args)
        siem = _make_siem(args)
        _report("TopoGuard monitor", {
            "source": source.name,
            "sampling interval": f"{args.interval:g}s",
            "window": args.window_size,
            "TDA max homology dimension": args.tda_max_dim,
            "mode": "continuous" if args.ticks == 0 else f"{args.ticks} ticks",
        })
        exit_code = asyncio.run(
            _monitor_async(args, source, engine, notifier, siem)
        )
        if exit_code == 0 and getattr(args, "baseline", None):
            saved_training_time = _save_monitor_baseline(engine, args.baseline)
            if saved_training_time is not None:
                _report("Baseline snapshot", {
                    "status": "saved",
                    "path": args.baseline,
                })
        return exit_code
    except (OSError, ImportError, RuntimeError, ValueError) as exc:
        logger.error("%s", exc)
        return 2
    except KeyboardInterrupt:
        if engine is not None and getattr(args, "baseline", None):
            saved_training_time = _save_monitor_baseline(engine, args.baseline)
            if saved_training_time is not None:
                _report("Baseline snapshot", {
                    "status": "saved",
                    "path": args.baseline,
                })
        return 130
    finally:
        if source is not None:
            source.stop()
        if engine is not None:
            engine.stop()
        if siem is not None:
            siem.close()


def _audit(args) -> int:
    source = None
    engine = None
    notifier = None
    siem = None
    try:
        source = _make_source(args)
        engine = _make_engine(args)
        notifier = _make_notifier(args)
        siem = _make_siem(args)
        final_result: dict[str, Any] = {"status": "NO_DATA", "is_anomaly": False}
        tick = 0
        while True:
            tick += 1
            current_result = _inspect(engine, source, tick)
            if tick > 1 and current_result.get("status") == "NO_DATA":
                break
            final_result = current_result
            if final_result.get("status") == "ERROR":
                _report("Audit failed closed", final_result)
                return 2
            if getattr(source, "finished", False):
                break
            if args.source == "web-log" or not final_result.get("data_points"):
                break

        if final_result.get("status") != "SUCCESS":
            _report("Audit incomplete", {
                "input": args.input,
                "source": source.name,
                "status": final_result.get("status"),
                "details": final_result.get("message", "Insufficient source data"),
            })
            return 2

        score = float(final_result.get("anomaly_score", 0.0))
        result_context = final_result.get("context")
        if not isinstance(result_context, dict):
            result_context = {}
        tda_details = result_context.get("tda_details")
        if not isinstance(tda_details, dict):
            tda_details = {}
        verification = "not triggered (TDA score <= 0.85)"
        if score > 0.85 and args.state_json:
            from extension_plugins import Z3FormalVerificationPlugin
            with Path(args.state_json).open("r", encoding="utf-8") as state_file:
                state = json.load(state_file)
            if not isinstance(state, dict):
                raise ValueError("--state-json must contain a JSON object")
            z3_result = Z3FormalVerificationPlugin.verify_security_boundary(state)
            verification = f"{z3_result['reason']} ({z3_result['solver_status']})"
            final_result["security_boundary"] = z3_result

        _report("TopoGuard audit", {
            "input": args.input,
            "source": source.name,
            "status": final_result.get("status"),
            "TDA max homology dimension": args.tda_max_dim,
            "H0 features": tda_details.get("h0_count", "-"),
            "H1 features": tda_details.get("h1_count", "-"),
            "H2 features": (
                tda_details.get("h2_count", "-")
                if args.tda_max_dim >= 2
                else "not computed"
            ),
            "TDA score": f"{score:.4f}",
            "anomaly": final_result.get("is_anomaly"),
            "Z3 boundary proof": verification,
            "Z3 topology result": (final_result.get("z3_result") or {}).get("reason", "not triggered"),
        })
        if notifier is not None and notifier.send_alert(final_result, source.name, tick):
            _report("Webhook notification", {"status": "delivered"})
        _emit_siem(siem, final_result, source.name, tick)
        return 1 if final_result.get("is_anomaly") else 0
    except (OSError, ImportError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        logger.error("Audit failed: %s", exc)
        return 2
    finally:
        if source is not None:
            source.stop()
        if engine is not None:
            engine.stop()
        if siem is not None:
            siem.close()


def _collect_all_rows(args) -> list[dict[str, Any]]:
    """Collect every available row from a source until it finishes or a deadline passes."""
    from data_sources import create_data_source

    source = create_data_source(
        args.source,
        input_path=args.input,
        window_size=args.window_size,
        batch_size=args.batch_size,
        poll_interval=args.interval,
    )
    rows: list[dict[str, Any]] = []
    deadline = None
    if args.duration_seconds and args.duration_seconds > 0:
        deadline = time.monotonic() + args.duration_seconds
    try:
        while True:
            batch = source.collect()
            if batch:
                rows.extend(batch)
            if getattr(source, "finished", False):
                break
            if deadline is not None and time.monotonic() >= deadline:
                break
            if not batch:
                # Live sources (system/prometheus) emit on a cadence; file sources are done.
                if args.source in {"system", "prometheus"}:
                    time.sleep(args.interval)
                else:
                    break
    finally:
        source.stop()
    return rows


def _train_baseline(args) -> int:
    from baseline import BaselineCalibrator

    try:
        rows = _collect_all_rows(args)
        from hybrid_engine import IsolationForestBaseline

        required = IsolationForestBaseline.MIN_BASELINE_SAMPLES + 1
        if len(rows) < required:
            raise ValueError(
                f"Need at least {required} normal samples for calibration; received {len(rows)}"
            )
        calibrator = BaselineCalibrator(
            contamination=args.contamination, random_state=args.random_state
        )
        summary = calibrator.fit(rows)
        calibrator.save(args.output, format=args.format)
        _report("Baseline trained", {
            "output": args.output,
            "format": args.format,
            "samples": summary["sample_count"],
            "features": summary["feature_count"],
            "contamination": summary["contamination"],
            "status": summary["status"],
        })
        return 0
    except (OSError, ImportError, RuntimeError, ValueError) as exc:
        logger.error("Baseline training failed: %s", exc)
        return 2


def _load_export_points(args) -> tuple[np.ndarray, str]:
    source = _make_source(args)
    try:
        points: list[dict[str, float]] = []
        while len(points) < args.max_points:
            batch = source.collect()
            if batch:
                points = batch[-args.max_points:]
            if getattr(source, "finished", False):
                break
            if not batch:
                if args.source in {"system", "prometheus"}:
                    time.sleep(args.interval)
                else:
                    break
        if len(points) < 4:
            raise ValueError(f"At least 4 numeric samples are required; received {len(points)}")
        matrix = np.asarray([list(row.values()) for row in points], dtype=np.float64)
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
        return matrix, source.name
    finally:
        source.stop()


def _export_tda(args) -> int:
    try:
        from tda_engine import TDAEngine

        matrix, source_name = _load_export_points(args)
        mean = matrix.mean(axis=0)
        scale = matrix.std(axis=0)
        normalized = (matrix - mean) / np.where(scale > 1e-12, scale, 1.0)
        analyzed_points, diagram = TDAEngine(
            max_dimension=args.tda_max_dim,
            max_edge_length=args.max_edge,
        ).compute_persistence_topology(
            normalized,
            max_edge=args.max_edge,
        )

        radii = np.linspace(0.0, args.max_edge, args.grid_points)
        curves = {
            f"betti_{dimension}": [
                int(sum(
                    dim == dimension and birth <= radius < death
                    for dim, (birth, death) in diagram
                ))
                for radius in radii
            ]
            for dimension in range(args.tda_max_dim + 1)
        }
        payload = {
            "source": source_name,
            "sample_count": int(matrix.shape[0]),
            "analyzed_sample_count": int(analyzed_points.shape[0]),
            "feature_count": int(matrix.shape[1]),
            "max_edge": args.max_edge,
            "max_dimension": args.tda_max_dim,
            "radii": radii.tolist(),
            "betti_curves": curves,
            "persistence_intervals": [
                {
                    "dimension": int(dimension),
                    "birth": float(birth),
                    "death": None if not np.isfinite(death) else float(death),
                }
                for dimension, (birth, death) in diagram
            ],
        }
        if args.format == "json":
            serialized = json.dumps(payload, indent=2, allow_nan=False)
            if args.output == "-":
                print(serialized)
            else:
                Path(args.output).write_text(serialized + "\n", encoding="utf-8")
        else:
            destination = sys.stdout if args.output == "-" else Path(args.output).open("w", newline="", encoding="utf-8")
            with (nullcontext(destination) if args.output == "-" else destination) as stream:
                writer = csv.writer(stream)
                dimensions = range(args.tda_max_dim + 1)
                writer.writerow(["radius", *(f"betti_{dimension}" for dimension in dimensions)])
                for index, radius in enumerate(radii):
                    writer.writerow([
                        radius,
                        *(curves[f"betti_{dimension}"][index] for dimension in range(args.tda_max_dim + 1)),
                    ])
        if args.output != "-":
            _report("TDA export complete", {
                "output": args.output,
                "samples": matrix.shape[0],
                "features": matrix.shape[1],
                "intervals": len(diagram),
            })
        return 0
    except (OSError, ImportError, RuntimeError, ValueError) as exc:
        logger.error("TDA export failed: %s", exc)
        return 2


def _add_common_source_args(
    parser: argparse.ArgumentParser,
    *,
    monitor: bool = False,
    include_system: bool = False,
) -> None:
    choices = (
        ["system", "ebpf", "prometheus"]
        if monitor
        else ["auto", "web-log", "csv", "json", "jsonl", "zeek", "pcap"]
    )
    if include_system and not monitor:
        choices[1:1] = ["system", "prometheus"]
    parser.add_argument("--source", choices=choices, default="system" if monitor or include_system else "auto")
    parser.add_argument(
        "--input",
        help="Input file, Zeek JSON log, or Prometheus http(s):// endpoint URL",
    )
    parser.add_argument("--window-size", type=int, default=500)
    parser.add_argument(
        "--tda-max-dim",
        type=int,
        choices=(1, 2),
        default=1,
        help=(
            "Maximum homology dimension: 1 (H0/H1, default) or 2 "
            "(also H2; substantially more expensive)"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--interval", type=float, default=0.1)
    parser.add_argument("--no-polars", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="topoguard-cli", description="Live TopoGuard monitoring and formal audit")
    commands = parser.add_subparsers(dest="command", required=True)

    monitor = commands.add_parser("monitor", help="Monitor live host metrics or Linux eBPF events")
    _add_common_source_args(monitor, monitor=True)
    monitor.add_argument("--ticks", type=int, default=0, help="Number of samples; 0 runs until interrupted")
    monitor.add_argument(
        "--notify",
        action="store_true",
        help="Send anomaly summaries to Slack/Discord using TOPOGUARD_WEBHOOK_URL",
    )
    monitor.add_argument(
        "--baseline",
        help=(
            "Baseline snapshot path: load it at startup when present, save after "
            "training, and checkpoint hourly"
        ),
    )
    monitor.add_argument(
        "--siem",
        help="SIEM sink: 'ecs-json:<path>' or 'syslog://host[:port]' (read-only output)",
    )
    monitor.set_defaults(handler=_monitor)

    audit = commands.add_parser("audit", help="Audit a web log, numeric replay file, or offline packet capture")
    _add_common_source_args(audit)
    audit.add_argument("--state-json", help="JSON object with cpu, sockets, and su_attempts for Z3 boundary proof")
    audit.add_argument(
        "--notify",
        action="store_true",
        help="Send anomaly summaries to Slack/Discord using TOPOGUARD_WEBHOOK_URL",
    )
    audit.add_argument(
        "--baseline",
        help="Path to a pre-trained baseline pickle/json (skips Isolation Forest warm-up)",
    )
    audit.add_argument(
        "--siem",
        help="SIEM sink: 'ecs-json:<path>' or 'syslog://host[:port]' (read-only output)",
    )
    audit.set_defaults(handler=_audit)

    train_baseline = commands.add_parser(
        "train-baseline",
        help="Pre-train and serialize an Isolation Forest baseline from known-normal history",
    )
    _add_common_source_args(train_baseline)
    train_baseline.add_argument("--output", required=True, help="Output baseline path (.pkl or .json)")
    train_baseline.add_argument(
        "--format",
        choices=["pickle", "json"],
        default="pickle",
        help="Serialization format (default: pickle)",
    )
    train_baseline.add_argument(
        "--contamination",
        type=float,
        default=0.05,
        help="Isolation Forest contamination in (0, 0.5]",
    )
    train_baseline.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for reproducible baseline",
    )
    train_baseline.add_argument(
        "--duration-seconds",
        type=float,
        default=0.0,
        help="Seconds to collect from live sources (system/prometheus); 0 collects until source finishes",
    )
    train_baseline.set_defaults(handler=_train_baseline)

    export = commands.add_parser("export-tda", help="Export persistent intervals and sampled Betti curves")
    _add_common_source_args(export, include_system=True)
    export.add_argument("--max-points", type=int, default=300)
    export.add_argument("--max-edge", type=float, default=1.5)
    export.add_argument("--grid-points", type=int, default=100)
    export.add_argument("--format", choices=["json", "csv"], default="json")
    export.add_argument("--output", default="-", help="Output path, or '-' for stdout")
    export.set_defaults(handler=_export_tda)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.source == "auto":
        if not args.input:
            parser.error("--input is required")
        suffix = Path(args.input).suffix.lower()
        args.source = {
            ".log": "web-log",
            ".csv": "csv",
            ".json": "json",
            ".jsonl": "jsonl",
            ".pcap": "pcap",
            ".pcapng": "pcap",
            ".cap": "pcap",
        }.get(suffix)
        if args.source is None:
            parser.error("cannot infer source from input extension; select --source")
    if not 4 <= args.window_size <= 1000 or not 1 <= args.batch_size <= 10_000 or args.interval < 0:
        parser.error(
            "--window-size must be 4..1000, --batch-size 1..10000, and --interval non-negative"
        )
    if args.command == "monitor" and args.ticks < 0:
        parser.error("--ticks must be non-negative")
    if args.command == "export-tda":
        if not 4 <= args.max_points <= 500:
            parser.error("--max-points must be between 4 and 500")
        if args.max_edge <= 0 or args.grid_points < 2:
            parser.error("--max-edge must be positive and --grid-points must be at least 2")
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
