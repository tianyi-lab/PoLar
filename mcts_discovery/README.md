# Offline MCTS for PoLar

This directory contains the offline Monte Carlo Tree Search (MCTS) code used to discover program-of-layers execution paths for PoLar.

The implementation reuses:

- `llm_depth_router/` — model loading and custom layer-path execution
- `dart_math/` — math answer extraction and equivalence checking

MCTS is used for offline program discovery and is not required during PoLar inference.

## Setup

From the PoLar repository root:

```bash
git clone https://github.com/tianyi-lab/PoLar.git
cd PoLar
pip install -r requirements.txt
```

## Run MCTS

For example:

```bash
python -u mcts_discovery/run_mcts.py \
  --mode sample \
  --metric accuracy \
  --start 0 \
  --end 1 \
  --algorithm mcts \
  --dataset_name "hkust-nlp/dart-math-pool-math" \
  --group_size 0 \
  --num_groups 0 \
  --beam_width 0 \
  --mcts_sims 100 \
  --sample_size 2000 \
  --difficulty_level 1 \
  --base_model "Qwen/Qwen3-8B" \
  --max_new_tokens 50 \
  --output_dir ./mcts_outputs
```

For each difficulty level, `--sample_size 2000` first selects the row-level subset used by the MCTS pipeline. The selected rows are then deduplicated by the public DART-Math `query_id`, retaining the first occurrence of each query before MCTS is run.

`--start` and `--end` specify the range within the resulting deduplicated sample list, allowing the search to be distributed across multiple jobs.

MCTS output shards are written under:

```text
{output_dir}/{base_model}/dart-math-pool-math_results/
  mcts_sample_accuracy_0_0_0_{sims}_{diff}_{sample_size}_{max_new_tokens}/
    {start}_{end}.json
```

## Merge MCTS Shards

After all MCTS jobs finish, merge the output shards into the supervision format used by PoLar:

```bash
python mcts_discovery/merge_mcts_samples.py \
  ./mcts_outputs/Qwen/Qwen3-8B/dart-math-pool-math_results/mcts_sample_accuracy_0_0_0_100_1_2000_50 \
  --output merged_mcts_samples.json
```

Place the merged file where PoLar expects it:

```text
{data_root}/{model_path}/dart-math-diff-{N}/merged_mcts_samples.json
```

For example:

```bash
mkdir -p data/Qwen/Qwen3-8B/dart-math-diff-1
cp .../merged_mcts_samples.json data/Qwen/Qwen3-8B/dart-math-diff-1/
```

The resulting `merged_mcts_samples.json` can then be used directly as supervision for PoLar training.

## Output Fields

Each sample in the merged JSON includes:

| Field | Meaning |
|---|---|
| `query_id` | Public DART-Math query identifier |
| `question` / `gt_ans` | Problem and reference answer |
| `initial_score` | Full-depth baseline accuracy for the sample |
| `final_valid_transitions` | Valid layer programs found by MCTS |
| `final_invalid_transitions` | Evaluated invalid programs |

Path semantics match PoLar: full depth is `[0, 1, ..., D-1]`; skips omit layer indices, while repeats duplicate contiguous layer segments.