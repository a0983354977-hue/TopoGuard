import z3
import numpy as np
from contextlib import contextmanager
from threading import RLock
from typing import Optional, Dict, Any


class Z3FormalVerificationPlugin:
    DEFAULT_TIMEOUT_MS = 50
    DEFAULT_THRESHOLD_H1 = 50
    _solver_instance: Any = None
    _solver_lock = RLock()

    @staticmethod
    @contextmanager
    def _solver_context(timeout_ms: int = DEFAULT_TIMEOUT_MS):
        """Reuse one solver while isolating each verification in its own scope."""
        with Z3FormalVerificationPlugin._solver_lock:
            solver = Z3FormalVerificationPlugin._solver_instance
            if solver is None:
                solver = z3.Solver()
                Z3FormalVerificationPlugin._solver_instance = solver
            solver.set(timeout=timeout_ms)
            solver.push()
            try:
                yield solver
            finally:
                solver.pop()

    @staticmethod
    def verify_topological_invariants(
        h0_count: int,
        h1_count: int,
        threshold_h1: int = DEFAULT_THRESHOLD_H1,
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
        enable_lazy: bool = True,
        tda_anomaly_score: Optional[float] = None,
        tda_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Verify topological invariants with lazy evaluation and timeout control.

        Args:
            h0_count: H0 feature count
            h1_count: H1 feature count
            threshold_h1: Safety threshold for H1
            timeout_ms: Solver timeout in milliseconds
            enable_lazy: Only verify if TDA indicates anomaly
            tda_anomaly_score: TDA anomaly score (for lazy evaluation)
            tda_threshold: TDA threshold for lazy evaluation

        Returns:
            Dict with verification results and metadata
        """
        # Lazy evaluation: skip Z3 if TDA indicates normal
        if enable_lazy and tda_anomaly_score is not None:
            if tda_anomaly_score < tda_threshold:
                return {
                    'is_violated': False,
                    'reason': 'skipped_by_lazy_evaluation',
                    'h1_count': h1_count,
                    'threshold': threshold_h1,
                    'computation_time_ms': 0.0
                }

        start_time = 0
        try:
            import time
            start_time = time.time() * 1000

            with Z3FormalVerificationPlugin._solver_context(timeout_ms) as solver:
                # Constraint pruning: only add relevant constraints
                h1 = z3.Int('h1_count')
                solver.add(h1 == h1_count)
                is_violated = z3.Bool('is_violated')
                solver.add(is_violated == (h1 > threshold_h1))

                result = solver.check()

                computation_time = (time.time() * 1000) - start_time

                if result == z3.sat:
                    model = solver.model()
                    is_violated_value = model[is_violated]
                    return {
                        'is_violated': bool(is_violated_value),
                        'reason': 'verification_completed',
                        'h1_count': h1_count,
                        'threshold': threshold_h1,
                        'computation_time_ms': computation_time,
                        'solver_status': 'sat'
                    }
                elif result == z3.unknown:
                    return {
                        'is_violated': True,  # Fail-closed: treat unknown as violation
                        'reason': 'timeout',
                        'h1_count': h1_count,
                        'threshold': threshold_h1,
                        'computation_time_ms': computation_time,
                        'solver_status': 'unknown'
                    }
                else:
                    return {
                        'is_violated': False,
                        'reason': 'unsat',
                        'h1_count': h1_count,
                        'threshold': threshold_h1,
                        'computation_time_ms': computation_time,
                        'solver_status': 'unsat'
                    }

        except Exception as e:
            computation_time = (time.time() * 1000) - start_time if start_time > 0 else 0.0
            return {
                'is_violated': True,  # Fail-closed on error
                'reason': f'error: {str(e)}',
                'h1_count': h1_count,
                'threshold': threshold_h1,
                'computation_time_ms': computation_time,
                'solver_status': 'error'
            }

    @staticmethod
    def verify_security_boundary(
        system_state: Dict[str, Any],
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
    ) -> Dict[str, Any]:
        """Prove the supplied telemetry does not violate the resource/escalation boundary."""
        if not isinstance(system_state, dict):
            return {
                "is_violated": True,
                "reason": "system_state_must_be_a_mapping",
                "solver_status": "invalid",
            }
        required = {"cpu", "sockets", "su_attempts"}
        missing = required.difference(system_state)
        if missing:
            return {
                "is_violated": True,
                "reason": f"missing_state_fields: {', '.join(sorted(missing))}",
                "solver_status": "invalid",
            }

        try:
            raw_cpu = system_state["cpu"]
            raw_sockets = system_state["sockets"]
            raw_attempts = system_state["su_attempts"]
            numeric_types = (int, float, np.integer, np.floating)
            if any(isinstance(value, bool) or not isinstance(value, numeric_types)
                   for value in (raw_cpu, raw_sockets, raw_attempts)):
                raise ValueError("state values must be numeric")
            cpu_value = float(raw_cpu)
            socket_value = int(raw_sockets)
            escalation_value = int(raw_attempts)
            if not np.isfinite(cpu_value) or not 0 <= cpu_value <= 100:
                raise ValueError("cpu must be a finite percentage in [0, 100]")
            if (socket_value < 0 or escalation_value < 0
                    or socket_value != raw_sockets or escalation_value != raw_attempts):
                raise ValueError("socket and escalation counts must be non-negative")

            with Z3FormalVerificationPlugin._solver_context(timeout_ms) as solver:
                cpu_usage = z3.Real("boundary_cpu_usage")
                socket_count = z3.Int("boundary_socket_count")
                escalation_attempts = z3.Int("boundary_su_attempts")
                solver.add(cpu_usage == z3.RealVal(str(cpu_value)))
                solver.add(socket_count == socket_value)
                solver.add(escalation_attempts == escalation_value)
                solver.add(z3.Not(z3.And(cpu_usage > 90, escalation_attempts > 0)))
                result = solver.check()

            violated = result != z3.sat
            return {
                "is_violated": violated,
                "reason": "security_boundary_violated" if violated else "verified_safe",
                "solver_status": str(result),
                "state": {
                    "cpu": cpu_value,
                    "sockets": socket_value,
                    "su_attempts": escalation_value,
                },
            }
        except (OverflowError, TypeError, ValueError, z3.Z3Exception) as exc:
            return {
                "is_violated": True,
                "reason": f"verification_error: {exc}",
                "solver_status": "error",
            }


class PhysicsPhaseTransitionPlugin:
    @staticmethod
    def calculate_manifold_entropy(diag_persistence: list) -> float:
        if not diag_persistence:
            return 0.0
        lifespans = np.array([death - birth for _, (birth, death) in diag_persistence if death != float('inf')])
        if len(lifespans) == 0:
            return 0.0
        probs = lifespans / np.sum(lifespans)
        entropy = -np.sum(probs * np.log2(probs + 1e-12))
        return float(entropy)
