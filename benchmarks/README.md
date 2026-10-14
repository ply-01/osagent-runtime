# Memory evaluation pipeline

OSagent has one supported evaluation entry point:

```powershell
python benchmarks/memory_eval_pipeline.py
```

The pipeline reads `config/memory_eval_pipeline.json`, pins external source
revisions and comparison runtimes, runs adapters, normalizes metrics, and writes:

- `doc/evaluation/results/pipeline/manifest.json`
- `doc/evaluation/explicit-memory-pipeline-report.md`

Adapter scripts under `benchmarks/` are implementation details of this pipeline.
Do not maintain separate manual result workflows around them.

## Common commands

Run all deterministic default jobs:

```powershell
python benchmarks/memory_eval_pipeline.py
```

Run one default job or rebuild reports from existing JSON:

```powershell
python benchmarks/memory_eval_pipeline.py --job os-agent-competition
python benchmarks/memory_eval_pipeline.py --offline
python benchmarks/memory_eval_pipeline.py --no-run
```

Run optional live-LLM checks explicitly:

```powershell
python benchmarks/memory_eval_pipeline.py --job native-subagent-e2e
python benchmarks/memory_eval_pipeline.py --job mem2act-subagent-routing
```

Optional jobs are disabled during the default run because they require a local
LLM credential and incur remote API calls. Explicit `--job` selection overrides
the job's default enabled state.

The default comparison currently includes OSagent, Mem0 OSS 2.0.14, and Oh My
Pi 17.3.4's Mnemopi FTS backend on the same LoCoMo evidence slice. The pipeline
installs pinned third-party runtimes under ignored `data/` paths.

## Metric boundaries

- `os-agent-competition` is a deterministic development regression for the four
  competition-oriented memory metrics.
- LoCoMo jobs compare evidence retrieval under a shared slice and top-k protocol.
- The Oh My Pi job evaluates its official Mnemopi backend in FTS fallback mode;
  it does not claim full coding-agent task accuracy.
- PERMA support metrics are retrieval diagnostics, not official answer accuracy.
- `native-subagent-e2e` verifies real process startup, LLM tool routing, durable
  write, restart persistence, cross-session retrieval, and answer use.
- `mem2act-subagent-routing` measures planner routing only. It does not claim
  memory retrieval or answer accuracy.

Every public result must retain its dataset revision, sample range, model/runtime
configuration, and metric scope. Do not combine metrics from different protocols
into one leaderboard score.
