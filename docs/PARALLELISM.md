# CDW Agentic Pipeline — Parallelism Strategy

**Date**: 2026-03-25
**Authors**: Soumitri Chattopadhyay, Claude (Opus 4.6)

A technical analysis of case-level vs tool-level GPU parallelism, with implementation plan and critical assessment of agentic trade-offs.

---

## 1. Current Architecture: Case-Level Parallelism

```
GPU 0: case_A → TotalSeg → MRSeg → VISTA3D → VoxTell → ... → QC → radiomics
GPU 1: case_B → TotalSeg → MRSeg → VISTA3D → VoxTell → ... → QC → radiomics
GPU 2: case_C → ...
...
GPU 7: case_H → ...
```

- **8 GPUs, 8 workers** → 8 cases processed in parallel
- Within each case, tools run **sequentially** on a single GPU
- `BatchRunner` manages a thread-safe `Queue` of `"gpu:0"` ... `"gpu:7"` tokens
- Each worker thread acquires 1 GPU, processes an entire case, releases it
- Tools run via `conda run -n <env>` subprocess — complete environment isolation

### Throughput math

Assume each tool takes ~T minutes per case, 7 tools per case:

- **Time per case**: 7T (sequential tools)
- **Cases in parallel**: 8
- **Throughput**: 8 cases per 7T → **1.14 cases/T**
- **GPU utilization**: 100% (each GPU always running a tool)

---

## 2. Proposed Architecture: Tool-Level Parallelism

```
Case A arrives:
  GPU 0: TotalSeg_CT ──────┐
  GPU 1: MRSegmentator ────┤
  GPU 2: VISTA3D ──────────┤
  GPU 3: VoxTell ──────────┼──→ all tools finish → QC → radiomics
  GPU 4: TextMedSeg3D ─────┤
  GPU 5: VIBESegmentator ──┤
  GPU 6: BiomedParse3D ────┘
  GPU 7: LLMs (Qwen3-8B + MedGemma-27B)
```

- 7 tools run simultaneously on 7 GPUs for a single case
- GPU 7 reserved for LLM inference (metadata, QC interpretation, radiomics gating)
- Case finishes in `max(T_tool_i)` instead of `sum(T_tool_i)`

### Throughput math

- **Time per case**: max(T_i) ≈ T (bottlenecked by slowest tool)
- **Cases in parallel**: 1
- **Throughput**: 1 case per T → **1.0 cases/T**
- **GPU utilization**: varies — fast tools finish early, GPUs idle until slowest completes

### Per-case latency improvement

If tools take roughly equal time: latency drops from 7T to T — a **7× speedup per case**. But throughput is ~14% worse than case-level parallelism.

If tools have unequal runtimes (realistic):

| Tool | Approx. runtime (CT abdomen) |
|------|------------------------------|
| TotalSegmentator_CT | 3-5 min |
| MRSegmentator | 2-4 min |
| VISTA3D | 5-8 min (345 classes) |
| VoxTell | 1-3 min (per organ prompt) |
| TextMedSeg3D | 2-4 min |
| VIBESegmentator | 1-2 min |
| BiomedParse3D | 1-3 min |

Bottleneck: VISTA3D at ~8 min. All other GPUs idle for 3-6 minutes waiting.

---

## 3. Hybrid Architecture (best of both)

```
Batch of cases:
  Case A:                       Case B:
    GPU 0: TotalSeg_CT ───┐      GPU 4: TotalSeg_CT ───┐
    GPU 1: MRSeg ─────────┤      GPU 5: MRSeg ─────────┤
    GPU 2: VISTA3D ────────┤      GPU 6: VISTA3D ────────┤
    GPU 3: VoxTell ────────┘      GPU 7: VoxTell ────────┘
    (remaining tools queue         (remaining tools queue
     on freed GPUs)                 on freed GPUs)
```

- N cases × (8/N) GPUs per case
- Tools within each case run in parallel across their allocated GPUs
- When a tool finishes, its GPU picks up the next queued tool for that case
- LLM runs on whichever GPU finishes first (or CPU, since Qwen3-8B is light)

### Throughput math (2 cases × 4 GPUs each)

- 7 tools per case, 4 GPUs → ~2 rounds of parallel execution
- Time per case: ~2T (vs 7T sequential, vs T full-parallel)
- Throughput: 2 cases per 2T → **1.0 cases/T** (same as full tool-parallel)
- But better GPU utilization — fewer idle GPUs

---

## 4. Implementation Plan for Tool-Level Parallelism

### 4.1 Changes to `orchestrator/pipeline.py`

**Current**: `_step_segmentation()` loops sequentially over tools (line 435).

**Change**: Replace the sequential `for` loop with `ThreadPoolExecutor`:

```python
# Current (sequential)
for tool_name in result.selected_tools:
    seg_dir = os.path.join(case_path, output_dirname)
    inp = ToolInput(..., device=self.device)
    output = tool._timed_run(inp)

# Proposed (parallel)
from concurrent.futures import ThreadPoolExecutor, as_completed

def _run_single_tool(tool_name, device):
    inp = ToolInput(..., device=device)
    return tool_name, tool._timed_run(inp)

with ThreadPoolExecutor(max_workers=len(self.devices)) as executor:
    futures = {}
    device_pool = Queue()
    for d in self.devices:
        device_pool.put(d)

    for tool_name in result.selected_tools:
        device = device_pool.get()
        future = executor.submit(_run_single_tool, tool_name, device)
        futures[future] = (tool_name, device)

    for future in as_completed(futures):
        tool_name, device = futures[future]
        device_pool.put(device)  # return device for next tool
        # ... collect results
```

Key change: `self.device: str` becomes `self.devices: List[str]` — a pool of GPUs allocated to this case.

### 4.2 Changes to `CasePipeline.__init__`

```python
# Current
def __init__(self, ..., device: str = "gpu:0", ...):
    self.device = device

# Proposed
def __init__(self, ..., devices: Optional[List[str]] = None, device: str = "gpu:0", ...):
    # Backwards compatible: single device still works
    if devices:
        self.devices = devices
    else:
        self.devices = [device]
```

### 4.3 Changes to `orchestrator/batch_runner.py`

**Current**: GPU pool distributes 1 GPU per worker.

**Change**: Distribute N GPUs per worker:

```python
# Current
def _run_one_case(self, case_id, case_path):
    device = self._gpu_pool.get()  # 1 GPU
    pipeline = CasePipeline(device=device, ...)

# Proposed
def _run_one_case(self, case_id, case_path):
    devices = []
    for _ in range(self.gpus_per_case):
        devices.append(self._gpu_pool.get())  # N GPUs
    try:
        pipeline = CasePipeline(devices=devices, ...)
        result = pipeline.run(case_path)
    finally:
        for d in devices:
            self._gpu_pool.put(d)
```

New parameter: `gpus_per_case: int = 1` (default preserves current behavior). Workers = `len(gpus) // gpus_per_case`.

CLI:
```bash
# Case-level parallelism (current behavior)
python -m orchestrator.batch_runner --gpus 0,1,2,3,4,5,6,7 --workers 8

# Tool-level parallelism (7 GPUs per case, 1 case at a time)
python -m orchestrator.batch_runner --gpus 0,1,2,3,4,5,6,7 --workers 1 --gpus-per-case 7

# Hybrid (4 GPUs per case, 2 cases at a time)
python -m orchestrator.batch_runner --gpus 0,1,2,3,4,5,6,7 --workers 2 --gpus-per-case 4
```

### 4.4 Device Handling Fixes in Tools

**Problem**: Not all tools respect the `device` parameter from `ToolInput`.

| Tool | Passes device? | Fix needed |
|------|---------------|------------|
| TotalSegmentator_CT | Yes → `--device gpu` | Needs `CUDA_VISIBLE_DEVICES` for multi-GPU |
| MRSegmentator | Yes | Same |
| VISTA3D | Yes → `--device cuda:X` | Works as-is |
| VoxTell | Yes → `--device cuda --gpu X` | Works as-is |
| VIBESegmentator | Yes | Verify |
| TextMedSeg3D | **No** — uses `torchrun` auto-detect | Must set `CUDA_VISIBLE_DEVICES` in subprocess env |
| BiomedParse3D | **No** — auto-detects `torch.cuda.is_available()` | Must set `CUDA_VISIBLE_DEVICES` in subprocess env |

**Fix for tools that ignore device**: Set `CUDA_VISIBLE_DEVICES` in the subprocess environment:

```python
# In base_tool.py _run_in_env()
def _run_in_env(self, cmd, cwd=None, device=None):
    env = os.environ.copy()
    if device and device.startswith("gpu:"):
        gpu_id = device.split(":")[1]
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
    full_cmd = ["conda", "run", "-n", self.conda_env] + cmd
    return subprocess.run(full_cmd, check=True, cwd=cwd, env=env, ...)
```

This is the most robust approach — `CUDA_VISIBLE_DEVICES` works regardless of how the tool internally handles GPU selection. It also prevents tools from accidentally using a GPU allocated to another worker.

**Important**: TotalSegmentator translates `"gpu:0"` → `"gpu"` and lets PyTorch pick the GPU. Without `CUDA_VISIBLE_DEVICES`, if two workers both pass `--device gpu`, they fight for GPU 0. Setting `CUDA_VISIBLE_DEVICES=3` makes `"gpu"` resolve to physical GPU 3. This fix benefits even the current case-level parallelism.

### 4.5 Thread Safety Considerations

Tools run as **subprocesses** (`conda run`), so there are no in-process thread safety issues. Each subprocess has its own memory space, Python interpreter, and CUDA context. The only shared resources are:

1. **Filesystem**: Multiple tools write to the same `case_path` but to different subdirectories (`segmentations_totalseg_ct/`, `segmentations_vista3d/`, etc.). No conflicts.
2. **GPU VRAM**: Isolated by `CUDA_VISIBLE_DEVICES` — each tool sees only its assigned GPU.
3. **System RAM**: Loading 7 tool subprocesses simultaneously uses more RAM. Each tool loads its model weights into CPU RAM before GPU transfer. Estimate ~4-8GB per tool → 28-56GB total. On a system with 8×L40S this should have ample system RAM.

### 4.6 Postprocessing Timing

Currently, `_postprocess_masks()` runs after all tools finish — this doesn't change. QC and radiomics also run after all tools complete. The barrier is natural: `ThreadPoolExecutor` context manager blocks until all futures complete.

### 4.7 Summary of Files to Change

| File | Change | Scope |
|------|--------|-------|
| `orchestrator/pipeline.py` | `self.device` → `self.devices`, parallel `_step_segmentation` | ~50 lines |
| `orchestrator/batch_runner.py` | `--gpus-per-case` param, multi-GPU allocation per worker | ~20 lines |
| `tools/base_tool.py` | `_run_in_env()` gets `device` param, sets `CUDA_VISIBLE_DEVICES` | ~10 lines |
| `tools/textmedseg3d.py` | Pass `device` to `_run_in_env()` | ~2 lines |
| `tools/biomedparse3d.py` | Pass `device` to `_run_in_env()` | ~2 lines |

Total: ~85 lines changed. Backwards compatible via `gpus_per_case=1` default.

---

## 5. Critical Analysis: Which Strategy Is Better?

### 5.1 Throughput Comparison (26,000 cases)

| Strategy | Config | Throughput | Total time (est.) | GPU utilization |
|----------|--------|-----------|-------------------|-----------------|
| Case-level (current) | 8 cases × 1 GPU | 1.14/T | 22,750T | ~100% |
| Tool-level | 1 case × 7 GPUs (+1 LLM) | 1.0/T | 26,000T | ~60-70% (idle wait) |
| Hybrid (4+4) | 2 cases × 4 GPUs | ~1.0/T | ~26,000T | ~80% |
| Hybrid (2+2+2+2) | 4 cases × 2 GPUs | ~1.07/T | ~24,300T | ~90% |

**Case-level parallelism wins on throughput by ~14%.** This matters at 26k scale — it's the difference between ~16 days and ~18 days of continuous processing (at T=5 min average).

The reason is simple: tool runtimes are unequal. In tool-level parallelism, fast tools (VIBESeg ~1min) finish and their GPUs sit idle waiting for slow tools (VISTA3D ~8min). In case-level parallelism, every GPU is always running something.

### 5.2 Latency Comparison (single case)

| Strategy | Time per case |
|----------|--------------|
| Case-level | 7T ≈ 25-35 min |
| Tool-level | max(T_i) ≈ 5-8 min |

**Tool-level wins 4-5× on per-case latency.** This matters for:
- Interactive use (radiologist waiting for results)
- Debugging (faster iteration on single cases)
- Demo/testing scenarios

### 5.3 The Agentic Argument — Sequential Tools Enable Closed-Loop Reasoning

This is the strongest argument for **keeping sequential tool execution**, and it directly connects to the proposals in [CRITICISM.md](CRITICISM.md).

#### Sequential tools enable adaptive refinement (CRITICISM.md Proposal 1)

```
Sequential (possible):
  Run TotalSeg → QC: pancreas fragmented (3 CCs, LCC=0.4)
    → LLM reasons: "TotalSeg struggles with small pancreas"
    → LLM decides: "Run VISTA3D for pancreas only"
    → Run VISTA3D (pancreas) → QC passes → use VISTA3D mask

Parallel (impossible):
  Run ALL tools simultaneously → all finish → QC finds problems
    → too late to inform tool selection
```

With sequential tools, the LLM can observe Tool A's output and decide whether Tool B is even necessary, or whether to run Tool B with different parameters. This is **genuine agency** — the pipeline adapts based on intermediate results.

#### Sequential tools enable adaptive tool selection (CRITICISM.md Proposal 2)

```
Sequential (possible):
  Run TotalSeg → covers 117 organs with high quality
    → LLM reasons: "TotalSeg got kidney, liver, spleen all clean.
       MRSeg would only add 5 organs already covered. Skip it."
    → Skip MRSeg → save 3 minutes
    → Run VISTA3D only for organs TotalSeg missed

Parallel (impossible):
  All tools run regardless → no opportunity to skip redundant tools
```

This is the **30-50% compute savings** mentioned in CRITICISM.md — but it requires sequential execution to work.

#### Sequential tools enable error-aware recovery (CRITICISM.md Proposal 3)

```
Sequential (possible):
  Run VISTA3D → CUDA OOM on 800×800×500 volume
    → LLM diagnoses: "Volume too large for VISTA3D"
    → LLM decides: "Resample to half resolution, retry"
    → Retry with resampled input → succeeds

Parallel (impossible):
  VISTA3D crashes → other tools continue → crash logged → no recovery
  (actually this is also possible in parallel — retry on the freed GPU)
```

Recovery is partially possible in parallel mode (retry on the same GPU after crash), but sequential mode allows the LLM to use information from prior successful tools to inform the recovery strategy.

### 5.4 The Counter-Argument — Parallel Tools with Post-Hoc Correction

Tool-level parallelism doesn't completely preclude agentic behavior. A **two-phase approach**:

```
Phase 1 (parallel): Run all 7 tools simultaneously
Phase 2 (serial, agentic):
  QC all outputs → LLM identifies failures
    → selectively re-run failed organs on freed GPUs
    → use successful tools' outputs to guide re-runs
```

This preserves low latency AND allows corrective action. But:
- Phase 2 adds complexity (which GPUs are free? which tools to re-run?)
- The initial parallel run wastes compute on tools the LLM would have skipped
- You lose the 30-50% adaptive tool selection savings from Proposal 2

### 5.5 Verdict

| Criterion | Case-level wins? | Tool-level wins? |
|-----------|:---:|:---:|
| Throughput at scale (26k) | **Yes** (+14%) | No |
| GPU utilization | **Yes** (~100%) | No (~65%) |
| Per-case latency | No | **Yes** (4-5×) |
| Implementation simplicity | **Yes** (current) | No |
| Agentic: adaptive tool selection | **Yes** | No |
| Agentic: closed-loop refinement | **Yes** | Partial (two-phase) |
| Agentic: error recovery | **Yes** | Partial |
| Interactive/demo use | No | **Yes** |

**Recommendation**: Keep **case-level parallelism as the default** for batch processing at 26k scale. It's faster, simpler, fully utilizes GPUs, and — crucially — enables the agentic patterns that would make the pipeline genuinely intelligent.

Implement tool-level parallelism as an **opt-in mode** (`--gpus-per-case N`) for:
- Interactive single-case processing
- Debugging and testing
- Demos where latency matters

The hybrid (4 cases × 2 GPUs, or 2 cases × 4 GPUs) is worth benchmarking once tool-level parallelism is implemented, but is lower priority than the agentic proposals in CRITICISM.md.

### 5.6 The Real Priority

The most impactful improvement isn't parallelism — it's **making the sequential execution actually agentic**. Right now, sequential tools run in a fixed order with no feedback between them. The pipeline could:

1. Run TotalSeg → inspect results → decide whether MRSeg adds value → skip or run
2. Run VISTA3D → QC finds pancreas failure → re-run with different parameters
3. Run VoxTell → shape mismatch detected → apply resampling fix automatically

This requires the changes described in CRITICISM.md Proposals 1-3, all of which depend on sequential execution. Implementing tool-level parallelism before implementing agentic tool selection would lock out the highest-value improvements.

**Sequence**:
1. First, make sequential execution agentic (adaptive selection, closed-loop refinement)
2. Then, add tool-level parallelism as opt-in for interactive use
3. Finally, benchmark the hybrid approach for batch processing

---

## 6. Quick Reference

```bash
# Current: 8 cases in parallel, sequential tools (default, best for batch)
python -m orchestrator.batch_runner \
    --filelist cases.json \
    --gpus 0,1,2,3,4,5,6,7 --workers 8

# Future: tool-level parallelism (best for single-case / interactive)
python -m orchestrator.batch_runner \
    --filelist cases.json \
    --gpus 0,1,2,3,4,5,6,7 --workers 1 --gpus-per-case 7

# Future: hybrid (balance throughput and latency)
python -m orchestrator.batch_runner \
    --filelist cases.json \
    --gpus 0,1,2,3,4,5,6,7 --workers 2 --gpus-per-case 4
```
