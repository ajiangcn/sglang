# SGLang source labs

[Architecture](architecture.md) | [Deep dives](deep-dives.md) |
[Shared measurement protocol](../../vllm/learning/03-roadmap-and-labs.md)

**Status:** source exercises and proposed experiments; no GPU lab results are
claimed. Use an isolated supported Linux environment for execution, not the
Windows source-reading host.

## Guided lab: two requests, one reusable prefix

**Goal:** explain why B computes eight new prompt positions without requiring
A to stay alive. Read the [worked story](deep-dives.md#a-worked-two-request-story)
after making your predictions. This is a second pass through the existing
course, not a new prerequisite course.

Use the same token-ID workload: cold A has `P0..P19`, finishes, then B has
`P0..P11, U12..U19` with `U12 != P12`. Each greedily produces exactly two output
tokens without an earlier stop. Use the story's dense text-only, device-only,
FULL-cache assumptions, identical namespace, no intervening eviction, and
enough capacity/chunk budget for B's eight new inputs. Page size 4 is a paper
geometry, **not** a server command-line recommendation.

### 1. Predict before locating the implementation

Fill this worksheet with values and a reason. Keep *chosen output tokens*,
*computed KV positions*, and *retained cache positions* in separate columns.

| Prediction | Your value and reason |
|---|---|
| A's newly computed prompt positions | |
| B's equal leading token IDs / request lookup cap / page-aligned lookup length | |
| B's actual reusable prefix / new prompt work | |
| B's `prefix_lens`, `extend_lens`, `seq_lens`, and position range | |
| Computed KV positions just after the first output is chosen | |
| Computed KV positions after feeding that output and choosing the second | |
| Retained page-aligned positions after A finishes | |
| Shared-prefix request holders after A / after B admission / after B finishes | |
| Distinct retained prompt positions after both finish, with no eviction | |
| If B's whole prompt were identical, would new prompt work be zero? | |

### 2. Locate the boundaries on Windows (reading only)

No engine imports, model downloads, or CUDA installation are needed to read.
From this checkout, these commands only locate source:

```powershell
Set-Location C:\Users\yow\Documents\Workspace\sglang
rg -n 'def (init_next_round_input|_compute_max_prefix_len|prepare_for_extend|prepare_for_decode)' python\sglang\srt\managers\schedule_batch.py
rg -n 'def (match_kv_cache|checkpoint_kv_cache|release_kv_cache)' python\sglang\srt\mem_cache\common.py
rg -n 'def (match_prefix|checkpoint_into_tree|lock|unlock)' python\sglang\srt\mem_cache\unified_radix_cache.py
```

Follow this short reading route; record one invariant at each boundary:

1. **Select the implementation.**
   [`default_radix_cache_factory` / `create_unified_radix_cache`](../python/sglang/srt/mem_cache/registry.py)
   select the built-in FULL-only unified controller for this dense case.
   [`select_tree_core_backend`](../python/sglang/srt/mem_cache/unified_cache/tree_core_registry.py)
   is a separate selection. The lab follows the Python core explicitly selected
   by the existing test fixture, not a claim about all deployed defaults.
2. **Match, then protect.**
   [`Req.init_next_round_input`](../python/sglang/srt/managers/schedule_batch.py)
   calls [`match_kv_cache`](../python/sglang/srt/mem_cache/common.py).
   Inspect [`UnifiedTreeCore.match_prefix` / `_split_node`](../python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py)
   and [`FullComponent.redistribute_on_node_split`](../python/sglang/srt/mem_cache/unified_cache/components/full.py).
   What is cloned: K/V tensors or tensors of their location indices?
3. **Admit, then allocate.**
   In [`Scheduler._get_new_batch_prefill_raw`](../python/sglang/srt/managers/scheduler.py)
   and [`PrefillAdder.add_one_req`](../python/sglang/srt/managers/schedule_policy.py),
   find request-slot checks, the temporary `_lock_node`, budget selection, and
   `_commit_prefill_admission`. Why must the selected prefix cease being
   evictable before capacity is checked? Locate the chunk branch that would
   invalidate our one-step assumption if the budget were smaller.
4. **Translate the plan.**
   Follow [`ScheduleBatch.prepare_for_extend`](../python/sglang/srt/managers/schedule_batch.py)
   through [`alloc_for_extend` / `write_cache_indices`](../python/sglang/srt/mem_cache/allocation.py),
   [`resolve_forward_inputs`](../python/sglang/srt/managers/overlap_utils.py),
   and [`TpModelWorker.forward_batch_generation`](../python/sglang/srt/managers/tp_worker.py)
   to [`ForwardBatch.init_new` / `compute_position_torch`](../python/sglang/srt/model_executor/forward_batch_info.py).
   Distinguish staged `prefill_input_ids_cpu` from materialized `input_ids`.
   Show why eight input IDs need a row describing twenty context positions.
   Do not insert the removed `ModelWorkerBatch` representation into this trace.
5. **Finish without destroying reuse.**
   Follow [`SchedulerBatchResultProcessor.process_batch_result_prefill` / `process_batch_result_decode`](../python/sglang/srt/managers/scheduler_components/batch_result_processor.py)
   to [`release_kv_cache`](../python/sglang/srt/mem_cache/common.py)
   and [`UnifiedRadixCache.checkpoint_into_tree`](../python/sglang/srt/mem_cache/unified_radix_cache.py).
   Mark the separate operations: publish eligible KV, free the private tail,
   unlock the prefix, release the request row. Which operation actually evicts
   the retained prefix later?

### 3. Inspect existing tests, not a hand-written mock transcript

Open [the unified cache suite](../test/registered/unit/mem_cache/test_unified_radix_cache_unittest.py).
Read `CacheConfig`, `build_fixture`, `_TREE_CORE_TEST_BACKEND`, `UnifiedRadixCacheSuite`,
and the `_CONFIGS` / `globals()[_name]` class-generation loop. It generates the
real selector **`Test_FULL_ps64`**. Its page size 64 is an existing fixture
geometry; translate the paper example into pages rather than changing the test.
The FULL fixture uses real pools and a `TokenToKVPoolAllocator`, not a launched
language model or a complete serving scheduler.

| Method inherited by `Test_FULL_ps64` | Observable contract to explain |
|---|---|
| `test_shared_prefix_split` | Both branches remain matchable, while a third suffix matches only the common prefix; `sanity_check()` holds |
| `test_evict_respects_lock_ref` | A protected prefix still matches after attempted eviction; releasing its receipt permits later eviction |
| `test_paged_page_boundary_mismatch` | A mismatch in page two leaves one matching page; a mismatch in page one leaves none |
| `test_paged_checkpoint_unaligned_tail_freed` | Finishing restores allocator capacity for the unretained tail while the aligned prefix still matches |

Inspect the last method's allocator setup before interpreting its
`available_size()` assertion: this fixture measures token-slot capacity, not
the physical-page reclamation of every production allocator. Also, these
direct cache tests do **not** exercise `Req._compute_max_prefix_len`; a
full-key cache match and a request's capped lookup are different operations.

### 4. Explain before trying to execute

Write four sentences, each backed by a symbol above:

- B's hit preserves the same physical prefix locations despite a tree split.
- B cannot count its required prefix as evictable admission capacity.
- `prefix_len + extend_len` explains the context length, not the input tensor length.
- A finished request can release its row and lock while its zero-user prefix
  remains reusable; only later eviction reclaims that retained state.

**Pass gate:** reject a wrong implementation by a wrong match length, invalid
location mapping, premature eviction, or missing capacity recovery—not merely
because a mock's method was or was not called. The existing tests establish
component contracts, not an executed end-to-end A/B inference result.

<details>
<summary>Small answer key</summary>

- A computes 20 prompt positions. B has 12 equal leading IDs; the request cap
  is 19, page-aligned to 16, and divergence leaves an actual 12-position hit.
  B therefore computes eight new prompt positions.
- B's lists are `[12]`, `[8]`, `[20]`; its new positions are `12..19`.
  Twenty-position context does not mean twenty new model inputs.
- After choosing the first output, useful KV covers 20 positions. Feeding
  that output in the next step makes 21. The second chosen output has no
  computed KV in this ordinary completed-step ledger.
- Page size 4 retains 20 prompt positions after A; after both, the no-eviction
  union is `12 + 8 + 8 = 28`. Stable request-holder counts on the shared path
  are `0 -> 1 -> 0`, not simultaneous A/B ownership.
- For an entirely identical 20-token B, the request cap and page alignment
  permit 16 reused positions, leaving four to recompute.

</details>

**Optional, unrun:** in an already prepared, isolated, supported **Linux CUDA
development environment**, from the SGLang root, select only these existing
methods:

```bash
python3 test/registered/unit/mem_cache/test_unified_radix_cache_unittest.py \
  Test_FULL_ps64.test_shared_prefix_split \
  Test_FULL_ps64.test_evict_respects_lock_ref \
  Test_FULL_ps64.test_paged_page_boundary_mismatch \
  Test_FULL_ps64.test_paged_checkpoint_unaligned_tail_freed
```

Follow [test/README.md](../test/README.md), not an ad-hoc dependency installation.
Despite its `unit` directory, this file registers accelerator CI and its fixture
allocates device pools via `get_device()`. Do not treat it as a dependency-free
Windows/CPU exercise. **This command has not been run for this learning material.**
If you run it, save revision, selected fixture, environment, and actual results.

## Lab S0: trace three representations of one request

Read [TokenizerManager](../python/sglang/srt/managers/tokenizer_manager.py),
[scheduler batch structures](../python/sglang/srt/managers/schedule_batch.py),
[TP worker](../python/sglang/srt/managers/tp_worker.py), and
[ForwardBatch](../python/sglang/srt/model_executor/forward_batch_info.py).

Map:

```text
API/internal request -> tokenized IPC request -> Req
                     -> ScheduleBatch -> ForwardBatch
```

For a 20-token prompt with a reusable 12-token prefix, identify where each
representation records logical length, new work, cache locations, and request
identity. Choose a page size consistent with that illustrative hit, and account
for the final-token/logit behavior of the selected path.

Also trace an over-context input rejected before dispatch. Locate the
frontend failure cleanup and explain why no scheduler abort is necessary for a
request that never reached it.

Windows source-navigation examples from the SGLang root:

```powershell
rg -n 'async def generate_request|_release_req_states_on_failure' python\sglang\srt\managers\tokenizer_manager.py
rg -n 'def (event_loop_normal|event_loop_overlap|get_next_batch_to_run|run_batch)' python\sglang\srt\managers\scheduler.py
rg -n 'def forward_batch_generation|ForwardBatch.init_new' python\sglang\srt\managers\tp_worker.py
```

Pass gate: explain which owner can mutate each representation and when.

## Lab S1: branch, lock, and evict

This is a **concurrent-sharing extension**, unlike the serial A-then-B guided
lab above. Here there really are two live users of the shared prefix.

Start with [classic radix tests](../test/registered/unit/mem_cache/test_radix_cache_unit.py)
and [lock-reference tests](../test/registered/unit/mem_cache/test_decode_radix_lock_ref.py).

1. Insert a prefix and two divergent suffixes.
2. Predict the tree split.
3. Protect the prefix for two users.
4. Release only one user and attempt eviction.
5. Release the second, evict, and predict allocator capacity.
6. Repeat with a page boundary that cuts through the apparent string prefix.

In a prepared **Linux development environment**, from the SGLang root:

```bash
uv run --no-sync python -m pytest test/registered/unit/mem_cache/test_radix_cache_unit.py -q
```

Use the project's established environment and
[test instructions](../test/README.md). The test's logic can be CPU-based while
imports still require the project stack; do not equate "unit" with dependency-free.

Pass gate: identify the exact moment protected state becomes evictable and the
separate moment its storage is reclaimed.

## Lab S2: establish a baseline without obsolete benchmark paths

The canonical benchmark implementation in this checkout is
[serving benchmark](../python/sglang/benchmark/serving.py).
[The older module](../python/sglang/bench_serving.py) is a deprecated compatibility
shim. Prefer `sglang.benchmark.serving` in new experiments.

Prerequisites: pinned installation, compatible GPU libraries/drivers, enough
memory, model access, and a prepared environment. The following are small
debugging settings, **not** production recommendations.

Check options:

```bash
uv run --no-sync python -m sglang.launch_server --help
uv run --no-sync python -m sglang.benchmark.serving --help
```

In a dedicated Linux terminal:

```bash
uv run --no-sync python -m sglang.launch_server --model-path Qwen/Qwen2.5-0.5B-Instruct --host 127.0.0.1 --port 30000 --context-length 4096 --max-running-requests 16 --chunked-prefill-size 512 --mem-fraction-static 0.7
```

From another terminal:

```bash
curl --fail http://127.0.0.1:30000/health
curl --fail http://127.0.0.1:30000/v1/models
uv run --no-sync python -m sglang.benchmark.serving --backend sglang --host 127.0.0.1 --port 30000 --model Qwen/Qwen2.5-0.5B-Instruct --dataset-name random --random-input-len 256 --random-output-len 64 --num-prompts 100 --request-rate 2 --output-file learning-sglang-smoke.jsonl
```

This native-endpoint smoke test is not directly comparable to a differently
configured vLLM test. For a real comparison, use an identical saved workload,
endpoint semantics, sampling/stopping behavior, and the
[shared protocol](../../vllm/learning/03-roadmap-and-labs.md).
Different random-dataset implementations or even equal seeds do not guarantee
equal token sequences.

Dataset/tokenizer assets may require downloads. Check actual sampled lengths,
successful request counts, and stop reasons. A 100-request smoke run cannot
establish stable p99 behavior.

Stop only your own server with Ctrl+C when finished.

## Lab S3: observe prefix reuse separately from batching

Use the [cache-hit serving tests](../test/registered/radix_cache/test_radix_cache_hit.py)
as prior art and inspect the selected cache implementation.

Prepare:

- a repeated long system/document prefix with varied short questions;
- equal-length prompts that diverge early;
- prompts that change adapter/cache namespace where supported.

First run serially to expose prefill reuse; then run the fixed arrival trace
under load. Compare cold and intentionally warm states. If testing
`--disable-radix-cache`, inspect resolved configuration and initialization logs:
the switch can select a different cache path and interact with model features.

Record newly computed tokens, reusable prefix length, TTFT, decode latency,
memory/capacity, and retractions. A shorter TTFT could be caused by warm
compilation or batching; the cache counters and controlled input identity must
support the reuse explanation.

Pass gate: explain a partial hit and a forced miss from token/key identity,
not merely by quoting a cache-hit percentage.

## Lab S4: compare the normal and overlap loops

Read [scheduler loops](../python/sglang/srt/managers/scheduler.py) and
[worker sampling](../python/sglang/srt/managers/tp_worker.py).

Draw an iteration timeline with:

```text
ingest | plan | prepare inputs | forward | sample | process result | stream
```

Mark which iteration owns each result and where the schedule/forward stream
dependencies lie. Identify the shared-read/write hazard guarded before result
processing.

On a supported configuration, compare baseline with
`--disable-overlap-schedule` after checking current help. Hold graph, sampling,
and workload settings fixed where possible. Use short profiles and separate
unprofiled performance runs.

Pass gate: show an exposed gap removed by overlap and identify a case where a
dependency prevents that overlap. Do not infer benefit solely from the flag.

## Lab S5: unified state and HiCache

Read [component design](../python/sglang/srt/mem_cache/unified_cache/components/README.md)
and [HiCache design](../docs/docs/advanced_features/hicache_design.mdx).

Paper/source exercise:

1. Construct a prefix where Full state is present but a required Mamba checkpoint
   is absent.
2. Explain why the longest token path is not a valid full cache hit.
3. Identify when a branch needs independent recurrent state.
4. Distinguish a device hit from host restoration and storage prefetch.
5. Explain why another local instance cannot read this instance's private L2.

GPU/storage extension, only with a supported model/configuration:

- compare working sets fitting L1, exceeding L1, and needing L3;
- report actual transfer bytes/time and useful tokens restored;
- test timeout/cancellation during restore on an isolated deployment;
- compare against recomputation, not only against "no cache configured."

Pass gate: demonstrate which pool or tier is capacity/performance limiting and
why adding another kind of memory may not fix it.

## Lab S6: multimodal equivalence before multimodal speed

Follow [Qwen2.5-VL](../python/sglang/srt/models/qwen2_5_vl.py) and the selected
[processor](../python/sglang/srt/multimodal/processors).

Use one owned/public image with controlled processor settings:

1. Record processed grid, feature count, placeholder spans, and positions.
2. Repeat the image to test eligible processor/embedding/KV reuse separately.
3. Substitute a same-sized different image; verify no invalid reuse.
4. Increase resolution and attribute TTFT/memory growth to actual feature counts.
5. Compare inline versus supported encoder-separated execution only after
   verifying input/feature equivalence.

The [EPD guide](../docs/docs/advanced_features/epd_disaggregation.mdx) separates
encoder transport from global embedding caching. State which you enabled.

Pass gate: explain why processor, encoder, and LM prefix hit rates differ.

## Lab S7: PD ownership and failure handling

Read [PD guide](../docs/docs/advanced_features/pd_disaggregation.mdx),
[prefill path](../python/sglang/srt/disaggregation/prefill.py), and
[decode path](../python/sglang/srt/disaggregation/decode.py).

Do not copy a large-model cluster recipe onto unverified hardware. First make
a state table for the selected transfer backend:

```text
state | source KV owner | destination reservation | readiness
      | timeout owner | success cleanup | abort cleanup
```

The inspected deferred-allocation policy has substantial compatibility
restrictions. Its lesson is general: postponing destination reservation frees
decode capacity but retains prefill state longer and changes overlap.

If you have a compatible isolated cluster, test normal transfer, slow transfer,
client cancellation, and worker failure. Count total resources on both sides.

Pass gate: prove decode cannot read unfinished state and that repeated failed
handoffs do not leak capacity. A successful happy-path request is insufficient.

## Lab S8: admission reservation versus preemption

Read the [scheduling deep dive](deep-dives.md#4-scheduling-policy-plus-capacity)
and its vLLM comparison table first.

Paper exercise:

1. A KV pool holds 100,000 tokens. Ten running requests each have a 1,000-token
   prompt, 200 generated tokens, and `max_new_tokens = 4000`. For this budget
   exercise, take current **allocated** KV capacity as 12,000 token slots;
   this is not a claim that the last sampled token already has computed KV.
   Assume the clip limit does not bind and `new_token_ratio = 0.5`.
   How many tokens does SGLang's admission estimate reserve for their future
   output? How much room remains for a new 2,000-token prompt with the same
   `max_new_tokens`?
2. Repeat with ratio 0.1. Why does this admit more work but increase
   retraction risk?
3. For the same state, describe vLLM's behavior: what is allocated now, and
   what happens when the pool runs out mid-decode?

<details>
<summary>Answer sketch for item 1</summary>

Current allocated KV: 12,000 token slots by assumption. Estimated running-output
headroom: `10 * (4000 - 200) * 0.5 = 19,000`. Allocated plus reserved: 31,000, leaving
69,000 in this simplified ledger. The new candidate's allowance is not
ratio-scaled: before page overhead it needs `2000 + 4000 = 6,000`, so it still
fits. Check `PrefillAdder.add_one_req` and
`_get_running_request_total_token_offset` in
[schedule policy](../python/sglang/srt/managers/schedule_policy.py).
Real admission also applies chunking, page rounding, request-slot limits,
and the selected allocator's memory budget.

</details>

Experiment, on an isolated supported deployment:

- Use one prompt set with **two output-length distributions**: requests that
  stop early relative to `max_new_tokens`, and requests forced to the limit
  where `ignore_eos` is supported.
- Sweep `--schedule-conservativeness` while holding everything else fixed.
- Record admitted concurrency, retraction counts, the logged
  `new_token_ratio` transitions, recomputed tokens, and p95 TTFT/ITL.
- Run the vLLM equivalent with matched resulting KV capacity and count
  preemptions.

Pass gate: explain which design wastes capacity and which design wastes
compute for each distribution, using measured counters rather than intuition.

## What to save from every lab

Use the [shared manifest](../../vllm/learning/03-roadmap-and-labs.md).
At minimum: exact source revision, effective configuration, workload identity,
prediction, observed result, explanation, and remaining uncertainty.
An honest negative result is more useful than a plausible unverified claim.

**Next:** [Cross-framework comparison and reading list](../../vllm/learning/04-comparison-and-reading.md).
