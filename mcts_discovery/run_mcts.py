#!/usr/bin/env python3
"""Offline MCTS discovery of program-of-layers (PoLar) execution paths.

This module is part of the public PoLar repository and reuses PoLar's
`llm_depth_router` and `dart_math` utilities from the repository root.

Example:
  python -u mcts_discovery/run_mcts.py \\
    --mode sample --metric accuracy \\
    --start 4 --end 5 --algorithm mcts \\
    --dataset_name hkust-nlp/dart-math-pool-math \\
    --group_size 0 --num_groups 0 --beam_width 0 \\
    --mcts_sims 100 --sample_size 2000 --difficulty_level 1 \\
    --base_model Qwen/Qwen3-8B --max_new_tokens 50 \\
    --output_dir ./mcts_outputs
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter, OrderedDict, defaultdict
from difflib import SequenceMatcher
from functools import wraps
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
from datasets import load_dataset

from dart_math.eval import EvaluatorMathBatch
from llm_depth_router.model import get_model, get_tokenizer, setup_custom_path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

CACHE_SIZE = 1000
ARGS: Optional[argparse.Namespace] = None

# Globals set in main()
model = None
tokenizer = None
INITIAL_PATH: List[int] = []
ORIGINAL_DEPTH = 0
MAX_LENGTH = 0
MIN_LENGTH = 0
SEARCH_MODE = "sample"
EVAL_METRIC = "accuracy"
DATA_NAME = ""
SEARCH_ALGO = "mcts"
GROUP_SIZE = 0
NUM_GROUPS = 0
BEAM_WIDTH = 0
MCTS_SIMS = 100
DIFF_LEVEL = 1
SAMPLE_SIZE = 2000
MAX_NEW_TOKEN = 50
USE_MAJORITY_VOTING = False
NUM_GENERATIONS = 1
BASE_MODEL = ""
OUTPUT_DIR = ""
arc_samples: List[Any] = []


# ---------------------------------------------------------------------------
# Model depth presets (PoLar paper models + a few nearby variants)
# ---------------------------------------------------------------------------

MODEL_DEPTH_PRESETS = {
    "Qwen2.5-3B": dict(depth=36, min_length=24, max_length=44),
    "Qwen2.5-1.5B": dict(depth=28, min_length=16, max_length=32),
    "Llama-3.2-3B": dict(depth=28, min_length=16, max_length=32),
    "Qwen1.5-MoE-A2.7B": dict(depth=24, min_length=16, max_length=28),
    "Qwen3-8B": dict(depth=36, min_length=24, max_length=44),
    "Qwen3-30B-A3B": dict(depth=48, min_length=36, max_length=54),
}


def resolve_depth_config(base_model: str) -> Tuple[List[int], int, int, int]:
    for key, cfg in MODEL_DEPTH_PRESETS.items():
        if key in base_model:
            depth = cfg["depth"]
            return (
                list(range(depth)),
                depth,
                cfg["min_length"],
                cfg["max_length"],
            )
    # Fallback: read from HF config if available after load
    raise ValueError(
        f"Unsupported model: {base_model}. "
        f"Known keys: {list(MODEL_DEPTH_PRESETS)}"
    )


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def cached_evaluation(cache_size: int = CACHE_SIZE):
    def decorator(func):
        func.cache = OrderedDict()

        @wraps(func)
        def wrapper(transition, sample_set):
            sample_key = tuple(sorted(id(sample) for sample in sample_set))
            key = (tuple(transition), sample_key)
            if key in func.cache:
                return func.cache[key]
            result = func(transition, sample_set)
            if len(func.cache) >= cache_size:
                func.cache.popitem(last=False)
            func.cache[key] = result
            return result

        return wrapper

    return decorator


def batch_compare_answers(generated_texts, gt_texts):
    evaluator = EvaluatorMathBatch(
        strict_extract=True,
        use_orig_eq_for_olympiadbench=True,
        timeout=60,
    )
    samples = [
        SimpleNamespace(
            resp=gen_text,
            ref_ans=gt_text,
            ans=None,
            query="",
            dataset="math",
        )
        for gen_text, gt_text in zip(generated_texts, gt_texts)
    ]
    _, corrects = evaluator.batch_eval(samples, n_procs=4)
    return corrects


@cached_evaluation()
def evaluate_accuracy(transition, samples):
    print("------------- enter evaluate_accuracy -------------")
    print("transition: ", transition)
    setup_custom_path(model, transition)

    correct = 0.0
    sample_details = []
    prompt = "Solve the following math problem and provide the answer directly: "

    for sample_idx, sample in enumerate(samples):
        print(f"Processing sample {sample_idx + 1}/{len(samples)}")
        sample_result = evaluate_single_sample_with_majority_voting(
            sample, transition, prompt, NUM_GENERATIONS
        )
        if isinstance(sample_result, dict):
            sample_accuracy = sample_result["accuracy"]
        else:
            sample_accuracy = sample_result
        correct += sample_accuracy
        sample_details.append(
            {
                "sample_id": sample_idx,
                "sample_data": sample,
                "evaluation_result": sample_result,
            }
        )

    print("----------------------------------------------------")
    return {
        "overall_accuracy": correct / len(samples),
        "sample_details": sample_details,
        "transition": transition,
    }


def evaluate_single_sample_with_majority_voting(sample, transition, prompt, num_generations):
    model.model.custom_layer_indices = transition
    if not USE_MAJORITY_VOTING:
        num_generations = 1
    return evaluate_math_sample_with_majority_voting(
        sample, transition, prompt, num_generations
    )


def evaluate_math_sample_with_majority_voting(
    sample,
    transition,
    prompt,
    num_generations,
    data_name=None,
    max_new_tokens=None,
    temperature=0.0,
):
    if data_name is not None:
        question = sample["query"]
        answer_key = sample["gt_ans"]
    elif "dart" in DATA_NAME:
        if isinstance(sample, list) and len(sample) > 0:
            sample = sample[0]
        question = sample["query"]
        answer_key = sample["gt_ans"]
    else:
        raise ValueError(f"Unsupported dataset for math eval: {DATA_NAME}")

    input_text = (
        "Solve the following math problem and output ONLY the final answer directly, "
        "formatted strictly as \\boxed{ANSWER}.\n"
        "### Problem Start\n"
        f"{question}\n"
        "### Problem End\n"
        "Answer:"
    )

    if max_new_tokens is not None:
        max_new_token = int(max_new_tokens)
    else:
        max_new_token = (
            int(getattr(ARGS, "max_new_tokens", 50)) if ARGS is not None else 50
        )

    predictions = []
    generation_details = []
    for gen_idx in range(num_generations):
        with torch.no_grad():
            inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
            if temperature > 0:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_token,
                    do_sample=True,
                    temperature=float(temperature),
                )
            else:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_token,
                    do_sample=False,
                )
            pred_answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            answer_part = pred_answer.split("Answer:")[-1].strip()

        predictions.append(answer_part)
        generation_details.append(
            {
                "generation_id": gen_idx + 1,
                "full_response": pred_answer,
                "extracted_answer": answer_part,
            }
        )

    if num_generations == 1:
        answer_part = predictions[0]
        if "oxed{" in answer_part:
            correct = 1.0 if batch_compare_answers([answer_part], [answer_key])[0] else 0.0
        else:
            correct = 0.0
        print(f"answer_part: {answer_part} | answer_key: {answer_key} | correct: {correct}")
        return {
            "accuracy": correct,
            "generations": generation_details,
            "final_result": {
                "answer_part": answer_part,
                "answer_key": answer_key,
                "correct": correct,
                "num_generations": num_generations,
            },
        }

    correct_predictions = batch_compare_answers(
        predictions, [answer_key] * num_generations
    )
    for gen_idx, is_correct in enumerate(correct_predictions):
        generation_details[gen_idx]["correct"] = 1.0 if is_correct else 0.0

    correct_count = sum(correct_predictions)
    majority_correct = correct_count >= (num_generations + 1) // 2
    unique_predictions = set(predictions)
    return {
        "accuracy": 1.0 if majority_correct else 0.0,
        "sample_info": {
            "question": question,
            "prompt_text": "",
            "input_text": input_text,
            "ground_truth": answer_key,
        },
        "generations": generation_details,
        "majority_voting": {
            "correct_predictions": correct_predictions,
            "correct_count": correct_count,
            "total_generations": num_generations,
            "majority_decision": majority_correct,
            "unique_predictions": list(unique_predictions),
            "prediction_counts": {
                pred: predictions.count(pred) for pred in unique_predictions
            },
        },
    }


# ---------------------------------------------------------------------------
# MCTS
# ---------------------------------------------------------------------------

class MCTSNode:
    def __init__(self, path, parent=None, action=None):
        self.path = path
        self.parent = parent
        self.children = []
        self.visits = 0
        self.total_score = 0.0
        self.ucb = -float("inf")
        self.score = -float("inf")
        self.length = float("inf")
        self.fully_explored = False
        self.action = action
        self.similarity_to_valid = 0.0
        self.sample_details = []

    def get_action_sequence(self):
        actions = []
        node = self
        while node.parent:
            actions.append(node.action)
            node = node.parent
        return list(reversed(actions))


class MCTSSearch:
    def __init__(self):
        self.valid_paths = []

    def path_similarity(self, path1, path2):
        return SequenceMatcher(None, path1, path2).ratio()

    def calculate_max_similarity(self, target_path):
        if not self.valid_paths:
            return 0.0
        return max(
            self.path_similarity(target_path, valid_path)
            for valid_path in self.valid_paths
        )


def get_all_nodes(root):
    all_nodes = []
    queue = [root]
    while queue:
        node = queue.pop(0)
        all_nodes.append(node)
        queue.extend(node.children)
    return all_nodes


def generate_mcts_candidates(path, visited_paths, is_root=False):
    candidates = []
    path_len = len(path)

    for i in range(path_len):
        skipped = path[:i] + path[i + 1 :]
        if tuple(skipped) not in visited_paths:
            if skipped and MIN_LENGTH <= len(skipped) <= MAX_LENGTH:
                action = {
                    "type": "skip",
                    "values": [path[i]],
                    "positions": [i],
                    "count": 1,
                }
                candidates.append((skipped, action))
                visited_paths.add(tuple(skipped))

        if i + 1 < path_len and is_root:
            skipped_two = path[:i] + path[i + 2 :]
            if tuple(skipped_two) not in visited_paths:
                if skipped_two and MIN_LENGTH <= len(skipped_two) <= MAX_LENGTH:
                    action = {
                        "type": "skip",
                        "values": [path[i], path[i + 1]],
                        "positions": [i, i + 1],
                        "count": 2,
                    }
                    candidates.append((skipped_two, action))
                    visited_paths.add(tuple(skipped_two))

        if i + 2 < path_len and is_root:
            skipped_three = path[:i] + path[i + 3 :]
            if tuple(skipped_three) not in visited_paths:
                if skipped_three and MIN_LENGTH <= len(skipped_three) <= MAX_LENGTH:
                    action = {
                        "type": "skip",
                        "values": [path[i], path[i + 1], path[i + 2]],
                        "positions": [i, i + 1, i + 2],
                        "count": 3,
                    }
                    candidates.append((skipped_three, action))
                    visited_paths.add(tuple(skipped_three))

        if i + 3 < path_len and is_root:
            skipped_four = path[:i] + path[i + 4 :]
            if tuple(skipped_four) not in visited_paths:
                if skipped_four and MIN_LENGTH <= len(skipped_four) <= MAX_LENGTH:
                    action = {
                        "type": "skip",
                        "values": [path[i], path[i + 1], path[i + 2], path[i + 3]],
                        "positions": [i, i + 1, i + 2, i + 3],
                        "count": 4,
                    }
                    candidates.append((skipped_four, action))
                    visited_paths.add(tuple(skipped_four))

        repeated = path[:i] + [path[i]] + path[i:]
        element_counts = Counter(repeated)
        if max(element_counts.values()) <= 2 and tuple(repeated) not in visited_paths:
            if repeated and MIN_LENGTH <= len(repeated) <= MAX_LENGTH:
                action = {
                    "type": "repeat",
                    "values": [path[i]],
                    "positions": [i],
                    "count": 1,
                }
                candidates.append((repeated, action))
                visited_paths.add(tuple(repeated))

        if i + 1 < path_len and is_root:
            repeated_two = path[:i] + path[i : i + 2] + path[i:]
            if tuple(repeated_two) not in visited_paths:
                if repeated_two and MIN_LENGTH <= len(repeated_two) <= MAX_LENGTH:
                    action = {
                        "type": "repeat",
                        "values": [path[i], path[i + 1]],
                        "positions": [i, i + 1],
                        "count": 2,
                    }
                    candidates.append((repeated_two, action))
                    visited_paths.add(tuple(repeated_two))

        if i + 2 < path_len and is_root:
            repeated_three = path[:i] + path[i : i + 3] + path[i:]
            if tuple(repeated_three) not in visited_paths:
                if repeated_three and MIN_LENGTH <= len(repeated_three) <= MAX_LENGTH:
                    action = {
                        "type": "repeat",
                        "values": [path[i], path[i + 1], path[i + 2]],
                        "positions": [i, i + 1, i + 2],
                        "count": 3,
                    }
                    candidates.append((repeated_three, action))
                    visited_paths.add(tuple(repeated_three))

        if i + 3 < path_len and is_root:
            repeated_four = path[:i] + path[i : i + 4] + path[i:]
            if tuple(repeated_four) not in visited_paths:
                if repeated_four and MIN_LENGTH <= len(repeated_four) <= MAX_LENGTH:
                    action = {
                        "type": "repeat",
                        "values": [path[i], path[i + 1], path[i + 2], path[i + 3]],
                        "positions": [i, i + 1, i + 2, i + 3],
                        "count": 4,
                    }
                    candidates.append((repeated_four, action))
                    visited_paths.add(tuple(repeated_four))

    return candidates


def mcts_search(initial_path, samples, simulations=100, evaluated_paths=None):
    random.seed(42)
    search = MCTSSearch()
    length_strength = 5

    if evaluated_paths is None:
        evaluated_paths = {}

    root = MCTSNode(initial_path)
    initial_path_tuple = tuple(initial_path)

    if initial_path_tuple in evaluated_paths:
        cached_result = evaluated_paths[initial_path_tuple]
        score = cached_result.get("score", -float("inf"))
        root.sample_details = cached_result.get("sample_details", [])
        print(f"Using cached result for initial path: score={score}")
    else:
        score_result = evaluate_accuracy(root.path, samples)
        if isinstance(score_result, dict):
            score = score_result["overall_accuracy"]
            root.sample_details = score_result["sample_details"]
        else:
            score = score_result
            root.sample_details = []
        evaluated_paths[initial_path_tuple] = {
            "score": score,
            "sample_details": root.sample_details,
        }

    root.score = score
    root.length = len(root.path)
    visited_paths = set([tuple(initial_path)])
    trajectories = []
    valid_threshold = 0.8

    for i in range(simulations):
        node = root

        while node.children:
            unexplored_children = [
                child for child in node.children if not child.fully_explored
            ]
            if not unexplored_children:
                break
            if random.random() < 0.1:
                node = random.choice(unexplored_children)
            else:
                node = max(unexplored_children, key=lambda x: x.ucb)

        if not node.fully_explored:
            new_paths = generate_mcts_candidates(node.path, visited_paths, node == root)
            for path, action in new_paths:
                child_node = MCTSNode(path, parent=node, action=action)
                node.children.append(child_node)
                visited_paths.add(tuple(path))

            if node.children:
                node = random.choice(sorted(node.children, key=lambda x: x.path))
            else:
                node.fully_explored = True

        if node.score == -float("inf"):
            path_tuple = tuple(node.path)
            if path_tuple in evaluated_paths:
                cached_result = evaluated_paths[path_tuple]
                score = cached_result.get("score", -float("inf"))
                node.sample_details = cached_result.get("sample_details", [])
                print(f"Using cached result for path: score={score}")
            else:
                score_result = evaluate_accuracy(node.path, samples)
                if isinstance(score_result, dict):
                    score = score_result["overall_accuracy"]
                    node.sample_details = score_result["sample_details"]
                else:
                    score = score_result
                    node.sample_details = []
                evaluated_paths[path_tuple] = {
                    "score": score,
                    "sample_details": node.sample_details,
                }

            node.score = score
            node.length = len(node.path)

            if score == 1.0 and node.path not in search.valid_paths:
                search.valid_paths.append(node.path.copy())

            if score < 1.0:
                node.similarity_to_valid = search.calculate_max_similarity(node.path)
            else:
                node.similarity_to_valid = 1.0

            trajectories.append(
                {
                    "path": node.path,
                    "final_length": node.length,
                    "score": score,
                    "similarity": node.similarity_to_valid,
                    "is_valid": score >= valid_threshold,
                    "action_sequence": node.get_action_sequence(),
                    "visits": node.visits,
                    "ucb": node.ucb,
                }
            )

        while node:
            node.visits += 1
            node.total_score += score
            node.ucb = (
                (node.total_score / node.visits)
                - length_strength * len(node.path) / ORIGINAL_DEPTH
                + math.sqrt(2 * math.log(root.visits + 1) / node.visits)
            )
            if node.children and all(child.fully_explored for child in node.children):
                node.fully_explored = True
            node = node.parent

        all_nodes = get_all_nodes(root)
        sorted_results = sorted(
            all_nodes, key=lambda x: (x.score, -len(x.path)), reverse=True
        )
        print(f"{i}-th simulation")
        for item in sorted_results[:5]:
            print(
                f"path: {item.path}, score: {item.score}, length: {len(item.path)}"
            )

    all_nodes = get_all_nodes(root)
    evaluated_nodes = [node for node in all_nodes if node.score != -float("inf")]
    sorted_results = sorted(
        evaluated_nodes, key=lambda x: (x.score, -len(x.path)), reverse=True
    )
    max_score = sorted_results[0].score if sorted_results else -float("inf")

    best_transitions = [
        {
            "path": item.path,
            "score": item.score,
            "length": len(item.path),
            "sample_details": item.sample_details,
        }
        for item in sorted_results
    ]

    valid_count = sum(1 for item in sorted_results if item.score == 1.0)
    invalid_count = sum(1 for item in sorted_results if item.score != 1.0)
    print(
        f"max_score: {max_score}, valid_count: {valid_count}, "
        f"invalid_count: {invalid_count}, total_evaluated: {len(evaluated_nodes)}"
    )
    return best_transitions, max_score, trajectories


# ---------------------------------------------------------------------------
# Data + search loop
# ---------------------------------------------------------------------------

def sample_dataset(dataset, sample_size, seed=42, level=-1):
    random.seed(seed)
    level_dict = defaultdict(list)
    for i, data in enumerate(dataset):
        lvl = data["query_metadata"].get("level", 0)
        level_dict[lvl].append(i)

    if level is not None:
        indices = level_dict.get(level, [])
        num_samples = min(sample_size, len(indices))
        print("len(indices): ", len(indices))
        sampled_indices = indices[:num_samples]
    else:
        total_levels = len(level_dict)
        samples_per_level = max(1, sample_size // total_levels)
        sampled_indices = []
        for _, indices in level_dict.items():
            num_samples = min(samples_per_level, len(indices))
            sampled_indices.extend(indices[:num_samples])

    sampled_indices.sort()
    # Deduplicate the selected rows by query_id, keeping the first occurrence.
    deduplicated_indices = []
    seen_query_ids = set()
    for idx in sampled_indices:
        query_id = dataset[idx]["query_id"]
        if query_id in seen_query_ids:
            continue
        seen_query_ids.add(query_id)
        deduplicated_indices.append(idx)

    print(
        f"len(sampled_indices): {len(sampled_indices)}, "
        f"unique query_ids: {len(deduplicated_indices)}"
    )
    return [dataset[i] for i in deduplicated_indices]


def _path_list(transitions: List[Dict[str, Any]], score: float) -> List[List[int]]:
    return [p["path"] for p in transitions if p.get("score") == score]


def tree_search_single_sample(idx: int):
    print(f"idx {idx}")
    sample_set = [arc_samples[idx : idx + 1]]
    initial_result = evaluate_accuracy(INITIAL_PATH, sample_set)
    if isinstance(initial_result, dict):
        initial_metric = initial_result["overall_accuracy"]
    else:
        initial_metric = initial_result

    print(
        "Initial beam:",
        [{"path": INITIAL_PATH, "score": initial_metric}],
    )

    if SEARCH_ALGO != "mcts":
        raise ValueError("Only --algorithm mcts is supported in this release.")

    best_transitions, min_score, trajectories = mcts_search(
        INITIAL_PATH, sample_set, MCTS_SIMS
    )
    return best_transitions, initial_metric, min_score, trajectories


def build_results_dir() -> Path:
    dataset_tag = DATA_NAME.split("/")[-1]
    run_name = (
        f"{SEARCH_ALGO}_{SEARCH_MODE}_{EVAL_METRIC}_"
        f"{GROUP_SIZE}_{NUM_GROUPS}_{BEAM_WIDTH}_"
        f"{MCTS_SIMS}_{DIFF_LEVEL}_{SAMPLE_SIZE}_{MAX_NEW_TOKEN}"
    )
    return Path(OUTPUT_DIR) / BASE_MODEL / f"{dataset_tag}_results" / run_name


def tree_search(start_idx: int, end_idx: int):
    results_dir = build_results_dir()
    results_file = results_dir / f"{start_idx}_{end_idx}.json"
    results_dir.mkdir(parents=True, exist_ok=True)

    if results_file.exists():
        print(f"File {results_file} already exists.")
        return

    results = {
        "SEARCH_ALGO": SEARCH_ALGO,
        "SEARCH_MODE": SEARCH_MODE,
        "EVAL_METRIC": EVAL_METRIC,
        "GROUP_SIZE": GROUP_SIZE,
        "NUM_GROUPS": NUM_GROUPS,
        "BEAM_WIDTH": BEAM_WIDTH,
        "BASE_MODEL": BASE_MODEL,
        "DIFFICULTY_LEVEL": DIFF_LEVEL,
        "samples": [],
    }

    for i in range(start_idx, end_idx):
        print(f"start processing {i}-th sample")
        best_transitions, initial_metric, min_score, trajectories = (
            tree_search_single_sample(i)
        )
        sample_data = arc_samples[i]
        question = sample_data["query"]
        answer_key = sample_data["gt_ans"]
        prompt_text = (
            "Solve the following math problem and output ONLY the final answer, "
            "formatted strictly as \\boxed{ANSWER}.\n"
            "### Problem Start\n"
            f"{question}\n"
            "### Problem End\n"
            "Answer:"
        )

        # PoLar-compatible fields (preferred) + legacy fields for debugging
        final_valid = _path_list(best_transitions, 1.0)
        final_invalid = _path_list(best_transitions, 0.0)

        item = {
            "sample_id": i,
            "global_idx": (DIFF_LEVEL - 1) * SAMPLE_SIZE + i,
            "query_id": sample_data["query_id"],
            "question": question,
            "gt_ans": answer_key,
            "initial_score": initial_metric,
            "final_valid_transitions": final_valid,
            "final_invalid_transitions": final_invalid,
            "sample_info": {
                "question": question,
                "prompt_text": prompt_text,
                "ground_truth": answer_key,
            },
            "initial_transition_metric": initial_metric,
            "new_metric": min_score,
            "best_transitions": best_transitions,
            "valid_transitions": [p for p in best_transitions if p["score"] == 1],
            "invalid_transitions": [p for p in best_transitions if p["score"] == 0],
            "trajectories": {
                "valid": [t for t in trajectories if t["score"] == 1.0],
                "invalid": [t for t in trajectories if t["score"] < 1.0],
            },
        }
        results["samples"].append(item)
        print(
            f"sample {i}: valid={len(final_valid)} invalid={len(final_invalid)} "
            f"initial_score={initial_metric}"
        )

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline MCTS discovery of program-of-layers for PoLar."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["sample"],
        default="sample",
        help="Only sample-level MCTS is supported (extended mode removed).",
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["accuracy"],
        default="accuracy",
        help="Evaluation metric (accuracy only in this release).",
    )
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--sample_size", type=int, default=2000)
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="hkust-nlp/dart-math-pool-math",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="meta-llama/Llama-3.2-3B-Instruct",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["mcts"],
        default="mcts",
    )
    parser.add_argument(
        "--max_new_token",
        "--max_new_tokens",
        type=int,
        default=50,
        dest="max_new_tokens",
    )

    parser.add_argument("--group_size", type=int, default=0)
    parser.add_argument("--num_groups", type=int, default=0)
    parser.add_argument("--beam_width", type=int, default=0)
    parser.add_argument("--mcts_sims", type=int, default=100)
    parser.add_argument("--difficulty_level", type=int, default=1)
    parser.add_argument("--use_majority_voting", action="store_true")
    parser.add_argument("--num_generations", type=int, default=3)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./mcts_outputs",
        help="Root directory for shard JSON outputs.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
    )
    return parser.parse_args()


def main():
    global ARGS, model, tokenizer
    global INITIAL_PATH, ORIGINAL_DEPTH, MAX_LENGTH, MIN_LENGTH
    global SEARCH_MODE, EVAL_METRIC, DATA_NAME, SEARCH_ALGO
    global GROUP_SIZE, NUM_GROUPS, BEAM_WIDTH, MCTS_SIMS, DIFF_LEVEL
    global SAMPLE_SIZE, MAX_NEW_TOKEN, USE_MAJORITY_VOTING, NUM_GENERATIONS
    global BASE_MODEL, OUTPUT_DIR, arc_samples

    args = parse_args()
    ARGS = args

    BASE_MODEL = args.base_model
    INITIAL_PATH, ORIGINAL_DEPTH, MIN_LENGTH, MAX_LENGTH = resolve_depth_config(
        BASE_MODEL
    )

    model = get_model(BASE_MODEL, device=args.device)
    tokenizer = get_tokenizer(BASE_MODEL)
    model.eval()
    print("model.config:", model.config)

    SEARCH_MODE = args.mode
    EVAL_METRIC = args.metric
    DATA_NAME = args.dataset_name
    SEARCH_ALGO = args.algorithm
    GROUP_SIZE = args.group_size
    NUM_GROUPS = args.num_groups
    BEAM_WIDTH = args.beam_width
    MCTS_SIMS = args.mcts_sims
    DIFF_LEVEL = args.difficulty_level
    SAMPLE_SIZE = args.sample_size
    MAX_NEW_TOKEN = args.max_new_tokens
    USE_MAJORITY_VOTING = args.use_majority_voting
    NUM_GENERATIONS = args.num_generations
    OUTPUT_DIR = args.output_dir

    if "dart" not in DATA_NAME:
        raise ValueError(
            "This cleaned release targets DART-Math "
            f"(got dataset_name={DATA_NAME})."
        )

    cache_dir = os.environ.get("HF_DATASETS_CACHE") or os.environ.get("HF_HOME")
    ds_kwargs = {"cache_dir": cache_dir} if cache_dir else {}
    raw = load_dataset(DATA_NAME, **ds_kwargs)["train"]
    arc_samples = sample_dataset(raw, sample_size=SAMPLE_SIZE, level=DIFF_LEVEL)
    print("data loaded")

    start_time = time.time()
    print("begin sample-level tree search")
    tree_search(start_idx=args.start, end_idx=args.end)
    print(f"total time: {time.time() - start_time:.6f} s")


if __name__ == "__main__":
    main()
