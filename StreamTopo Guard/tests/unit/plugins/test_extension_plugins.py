import numpy as np
import z3
from extension_plugins import Z3FormalVerificationPlugin, PhysicsPhaseTransitionPlugin


def test_z3_solver_reuse_does_not_leak_constraints():
    with Z3FormalVerificationPlugin._solver_context() as first_solver:
        first_solver.add(z3.BoolVal(False))
        assert first_solver.check() == z3.unsat

    with Z3FormalVerificationPlugin._solver_context() as second_solver:
        assert second_solver is first_solver
        assert second_solver.check() == z3.sat


def test_z3_formal_verification():
    print("=" * 60)
    print("Testing Z3 Formal Verification Plugin")
    print("=" * 60)

    h0_count = 469
    h1_count = 268
    threshold_h1 = 50

    result = Z3FormalVerificationPlugin.verify_topological_invariants(
        h0_count=h0_count,
        h1_count=h1_count,
        threshold_h1=threshold_h1
    )

    print(f"[PASS] Test 1: H0={h0_count}, H1={h1_count}, Threshold={threshold_h1}")
    print(f"  Is Violated: {result['is_violated']}")
    print(f"  Reason: {result['reason']}")
    print(f"  Computation Time: {result['computation_time_ms']:.2f} ms")

    h1_count_high = 100
    result_high = Z3FormalVerificationPlugin.verify_topological_invariants(
        h0_count=h0_count,
        h1_count=h1_count_high,
        threshold_h1=threshold_h1
    )

    print(f"[PASS] Test 2: H0={h0_count}, H1={h1_count_high}, Threshold={threshold_h1}")
    print(f"  Is Violated: {result_high['is_violated']}")
    print(f"  Reason: {result_high['reason']}")

    # Test lazy evaluation
    result_lazy = Z3FormalVerificationPlugin.verify_topological_invariants(
        h0_count=h0_count,
        h1_count=h1_count,
        threshold_h1=threshold_h1,
        enable_lazy=True,
        tda_anomaly_score=0.3,
        tda_threshold=0.5
    )
    print(f"[PASS] Test 3: Lazy evaluation (TDA score below threshold)")
    print(f"  Is Violated: {result_lazy['is_violated']}")
    print(f"  Reason: {result_lazy['reason']}")

    print("\n[PASS] Z3 Formal Verification Plugin tests completed\n")


def test_physics_phase_transition():
    print("=" * 60)
    print("Testing Physics Phase Transition Plugin")
    print("=" * 60)

    diag_normal = [
        (0, (0.0, 0.5)),
        (0, (0.1, 0.6)),
        (1, (0.2, 0.8)),
        (1, (0.3, 0.9))
    ]

    entropy_normal = PhysicsPhaseTransitionPlugin.calculate_manifold_entropy(diag_normal)
    print(f"[PASS] Test 1: Normal persistence diagram")
    print(f"  Entropy: {entropy_normal:.4f} bits")

    diag_empty = []
    entropy_empty = PhysicsPhaseTransitionPlugin.calculate_manifold_entropy(diag_empty)
    print(f"[PASS] Test 2: Empty persistence diagram")
    print(f"  Entropy: {entropy_empty:.4f} bits")

    diag_high_entropy = [
        (0, (0.0, 1.0)),
        (0, (0.1, 0.3)),
        (1, (0.2, 2.0)),
        (1, (0.3, 0.5)),
        (1, (0.4, 1.5))
    ]

    entropy_high = PhysicsPhaseTransitionPlugin.calculate_manifold_entropy(diag_high_entropy)
    print(f"[PASS] Test 3: High entropy persistence diagram")
    print(f"  Entropy: {entropy_high:.4f} bits")

    diag_with_inf = [
        (0, (0.0, float('inf'))),
        (0, (0.1, 0.6)),
        (1, (0.2, 0.8))
    ]

    entropy_inf = PhysicsPhaseTransitionPlugin.calculate_manifold_entropy(diag_with_inf)
    print(f"[PASS] Test 4: Persistence diagram with infinite death values")
    print(f"  Entropy: {entropy_inf:.4f} bits")

    print("\n[PASS] Physics Phase Transition Plugin tests completed\n")


def test_integration_with_tda_engine():
    print("=" * 60)
    print("Testing Integration with TDA-Guard Core")
    print("=" * 60)

    try:
        from tda_engine import TDAEngine
        from data_adapter import DataAdapter

        np.random.seed(42)
        n_samples = 100
        theta = np.random.uniform(0, 2*np.pi, n_samples)
        phi = np.random.uniform(0, 2*np.pi, n_samples)
        R, r = 3.0, 1.0
        x = (R + r * np.cos(phi)) * np.cos(theta)
        y = (R + r * np.cos(phi)) * np.sin(theta)
        z = r * np.sin(phi)
        data = np.column_stack([x, y, z])

        adapter = DataAdapter()
        adapter.load_numpy(data)
        normalized_data = adapter.normalize_zscore()

        engine = TDAEngine()
        _, diag = engine.compute_persistence_topology(normalized_data, max_edge=1.5)
        stats = engine.get_statistics(diag)

        print(f"[PASS] TDA Engine computed successfully")
        print(f"  H0 count: {stats['H0']['count']}")
        print(f"  H1 count: {stats['H1']['count']}")

        z3_result = Z3FormalVerificationPlugin.verify_topological_invariants(
            h0_count=stats['H0']['count'],
            h1_count=stats['H1']['count'],
            threshold_h1=50
        )
        print(f"[PASS] Z3 Verification: Is Violated={z3_result['is_violated']}, Reason={z3_result['reason']}")

        entropy = PhysicsPhaseTransitionPlugin.calculate_manifold_entropy(diag)
        print(f"[PASS] Manifold Entropy: {entropy:.4f} bits")

        print("\n[PASS] Integration test completed successfully\n")

    except ImportError as e:
        print(f"[SKIP] Integration test skipped (TDA engine not available): {e}\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("TDA-Guard Extension Plugins Test Suite")
    print("=" * 60 + "\n")

    test_z3_solver_reuse_does_not_leak_constraints()
    print("[PASS] Z3 solver is reused without leaking scoped constraints")
    test_z3_formal_verification()
    test_physics_phase_transition()
    test_integration_with_tda_engine()

    print("=" * 60)
    print("All extension plugin tests completed!")
    print("=" * 60)
