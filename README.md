# Skip a Layer or Loop It? Learning Program-of-Layers in LLMs (ICML 2026 Oral)

[PDF](https://arxiv.org/pdf/2606.06574)
[Preliminary Study in 2025](https://arxiv.org/pdf/2507.07996)

This repository contains the release implementation for the ICML 2026 version of the project. The corresponding paper version is not publicly released yet; the previous version is available on arXiv.

![POLAR searches over execution programs that skip, keep, or repeat pretrained transformer layer segments.](search_space.png)

## Abstract

Large language models (LLMs) perform inference by following a fixed depth and order, non-recurrent execution of all layers. We reveal the wide existence of training-free, flexible, dynamic *program-of-layers (PoLar)*, where pretrained layers can be packed as modules and then skipped or looped to form a customized program for each input. For most inputs, substantially shorter program executions can achieve the same or better accuracy, while incorrect predictions of the original LLM can be corrected by alternative programs with fewer layers. These observations indicate that inference admits multiple valid latent computations beyond the standard forward pass. To efficiently achieve PoLar in practice, we propose a lightweight PoLar prediction network, which learns to generate execution programs that dynamically skip or repeat pretrained layers for each input. Experiments on mathematical reasoning benchmarks demonstrate that PoLar consistently improves accuracy over standard inference and prior dynamic-depth methods, often while executing fewer layers, and that these gains persist under out-of-distribution evaluation. Our results suggest that fixed-depth execution captures only a narrow subset of an LLM’s latent reasoning capacity.

The paper uses MCTS as an offline tool to discover valid execution programs and to study the program-of-layers space. This release focuses on the lightweight POLAR predictor trained from those discovered programs.

## Supported Models

This release keeps support for the four models used in the paper:

- `meta-llama/Llama-3.2-3B-Instruct`
- `Qwen/Qwen1.5-MoE-A2.7B-Chat`
- `Qwen/Qwen2.5-3B-Instruct`
- `Qwen/Qwen3-8B`

## Repository Layout

```text
run_polar.py        # CLI entrypoint
polar/
  config.py                    
  data.py                      
  model.py                     # PolarPredictor and beam decoding helpers
  train.py                     # training loop
  eval.py                      # evaluation loop
llm_depth_router/              # model loading and custom layer-path execution patches
dart_math/                     # math answer extraction and equivalence checking
```

## Installation

Install the Python dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Expected Data Layout

The code expects one `merged_mcts_samples.json` file per DART-Math difficulty level. `--data_root` should point to the root directory that contains one subdirectory per supported `model_path`:

```text
{data_root}/{model_path}/
  dart-math-diff-1/merged_mcts_samples.json
  dart-math-diff-2/merged_mcts_samples.json
  dart-math-diff-3/merged_mcts_samples.json
  dart-math-diff-4/merged_mcts_samples.json
  dart-math-diff-5/merged_mcts_samples.json
```

For example, with:

```bash
--model_path meta-llama/Llama-3.2-3B-Instruct
--data_root ./data
```

the diff-1 supervision file is read from:

```text
./data/meta-llama/Llama-3.2-3B-Instruct/dart-math-diff-1/merged_mcts_samples.json
```

## Supervision Format

To train POLAR, prepare each `merged_mcts_samples.json` supervision file as either:

- a JSON object with a top-level `"samples"` list;
- a JSON list of sample objects;
- or a JSON object whose values are sample objects.

Each sample should contain the original problem, the ground-truth answer, and offline-discovered valid execution paths:

```json
{
  "samples": [
    {
      "question": "Solve ...",
      "gt_ans": "\\boxed{42}",
      "initial_score": 0.0,
      "final_valid_transitions": [
        [0, 1, 2, 4, 5, 6],
        [0, 1, 2, 2, 3, 4, 5]
      ],
      "final_invalid_transitions": [
        [0, 1, 3, 4, 5]
      ]
    }
  ]
}
```

Required fields:

- `question`: the math problem text. Alternatively, use `sample_info.question`.
- `gt_ans`: the reference answer. The loader also accepts `ground_truth`, `answer`, `sample_info.ground_truth`, or `sample_info.answer`.
- `final_valid_transitions`: a list of valid layer execution paths found offline. Each path is a list of integer layer indices. Repeated layer indices represent recurrence.

Optional fields:

- `final_invalid_transitions`: paths known to be invalid. Evaluation uses these to avoid unnecessary online checks when `--trust_valid_cache` is enabled.
- `initial_score`: the baseline score for the original full-depth path, saved in the evaluation output for analysis.

Path semantics:

- A standard full-depth path is `[0, 1, ..., D-1]`, where `D` is the base model depth.
- Skipping is represented by omitting layer indices.
- Repeating is represented by repeating one or more layer indices, usually as a contiguous segment.
- During training, each valid path is deterministically parsed into a segmentation target and operation labels over `skip`, `keep`, and `repeat`.


## Sample Command

The sample command trains POLAR on DART-Math difficulty 1 for LLaMA-3.2-3B-Instruct and then evaluates the resulting checkpoint:

```bash
python3 run_polar.py \
  --policy_mode polar \
  --target_diff 1 \
  --model_path "meta-llama/Llama-3.2-3B-Instruct" \
  --data_root "./data" \
  --num_epochs 10 \
  --batch_size 128 \
  --learning_rate 5e-4 \
  --max_paths_per_sample 50 \
  --per_sample_weight_normalize \
  --beam_size 5 \
  --top_k_paths 5 \
  --reweight_original_path_if_shorter_valid \
  --original_path_weight 0.30 \
  --seed 42 \
  --lr_scheduler cosine \
  --warmup_steps 10
```


## Common Options

- `--target_diff {1,2,3,4,5}` trains and evaluates on one DART-Math difficulty level.
- `--eval_all_diffs` trains/evaluates across all five difficulty levels.
- `--save_dir` controls where checkpoints and evaluation JSON files are written. The default is `outputs`.
- `--checkpoint_path` provides an explicit checkpoint for evaluation.
- `--eval` skips training and only runs evaluation.
- `--top_k_paths` controls how many decoded candidate execution paths are checked per sample.
- `--beam_size` controls beam search size during path decoding.

## Citation

Please consider citing our work if you find the code or project useful.

Preliminary study:

```bibtex
@article{li2025CoLa,
  title={Skip a layer or loop it? test-time depth adaptation of pretrained llms},
  author={Li, Ziyue and Li, Yang and Zhou, Tianyi},
  journal={arXiv preprint arXiv:2507.07996},
  year={2025}
}
```

ICML 2026:

```bibtex
@inproceedings{li2026PoLar,
  author = {Ziyue Li and Yang Li and Tianyi Zhou},
  title = {{Skip a Layer or Loop It? Learning Program-of-Layers in LLMs}},
  booktitle = {Forty-third International Conference on Machine Learning (ICML)},
  year = {2026},
  url = {https://arxiv.org/pdf/2606.06574}}
```
