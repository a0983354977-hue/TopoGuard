# Test suites

Run an individual suite from the `StreamTopo Guard` project directory with
`python -m <module>`. For example:

```bash
python -m tests.unit.core.test_tda_guard
python -m tests.integration.test_integration
```

## Classification

| Directory | Coverage | Test modules |
| --- | --- | --- |
| `unit/core` | Core TDA data handling and topology calculations | `test_tda_guard.py` |
| `unit/adapters` | Industry data adapters | `test_adapters.py` |
| `unit/defensive` | Defensive guards and fallback behavior | `test_defensive.py` |
| `unit/engines` | Polars, eBPF, Isolation Forest, hybrid analysis, CLI integration, and webhook delivery | `test_polars_engine.py`, `test_ebpf_collector.py`, `test_hybrid_engine.py`, `test_production_cli.py` |
| `unit/plugins` | Extension and verification plugins | `test_extension_plugins.py` |
| `unit/metrics` | Evaluation and topology-based metrics | `test_evaluation_metrics.py` |
| `unit/sources` | Live host metrics, Prometheus scraping, Zeek JSON, access-log monitoring, CSV/JSON/JSONL replay | `test_data_sources.py` |
| `integration` | Full pipeline integration | `test_integration.py` |

## Data source modes

The terminal runner uses real local host metrics by default. Select another real
source explicitly with `--source ebpf`, `--source web-log --input <path>`,
`--source csv --input <path>`, `--source json --input <path>`, or
`--source jsonl --input <path>`. JSONL records may contain nested numeric
fields, which are flattened to stable dotted names for replay. Runtime
collectors do not expose synthetic/mock telemetry, and a requested real source
never silently falls back to generated data.
The production CLI also supports `monitor --source prometheus --input <http(s)://endpoint/metrics>`
and Prometheus TDA export; this endpoint source is live and is not an offline audit input.
Zeek JSON `conn.log`/`http.log` replay uses the explicit `--source zeek` option
and an allowlist of numeric fields.
Slack/Discord delivery is covered by `tests.unit.engines.test_production_cli`;
tests stub the HTTP client and never contact external webhook services.
