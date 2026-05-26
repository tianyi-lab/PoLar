import json
import random
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from .config import OP_EXECUTE, OP_REPEAT, OP_SKIP


def _coerce_path_list(p: Any) -> Optional[List[int]]:
    """
    Normalize a stored path to List[int].
    Valid paths may be stored as:
    - list[int]
    - tuple[int]
    - dict {"path": [...], ...}
    """
    if isinstance(p, dict):
        p = p.get("path")
    if isinstance(p, tuple):
        p = list(p)
    if not isinstance(p, list):
        return None
    try:
        return [int(x) for x in p]
    except Exception:
        return None

def _maybe_drop_original_path_from_valid_paths(
    valid_paths: List[Any],
    *,
    original_depth: int,
    enabled: bool,
    keep_original_prob: float = 0.0,
    rng: Optional[random.Random] = None,
    deterministic: bool = True,
) -> List[Any]:
    """
    If enabled and both:
      - original path exists in valid_paths, and
      - there exists another valid path with length < original_depth,
    then drop the original path from valid_paths to reduce over-preference.

    Soft strategy:
      - if 0 < keep_original_prob < 1 and rng is provided and deterministic=False,
        keep the original path with probability keep_original_prob.
    """
    if not enabled or not valid_paths or not isinstance(original_depth, int) or original_depth <= 0:
        return valid_paths

    original = list(range(original_depth))
    has_original = False
    min_other_len: Optional[int] = None

    for p in valid_paths:
        pl = _coerce_path_list(p)
        if pl is None:
            continue
        if pl == original:
            has_original = True
        else:
            l = len(pl)
            if min_other_len is None or l < min_other_len:
                min_other_len = l

    if not has_original or min_other_len is None or min_other_len >= len(original):
        return valid_paths

    # Probability sanity
    try:
        kp = float(keep_original_prob)
    except Exception:
        kp = 0.0
    kp = max(0.0, min(1.0, kp))

    if not deterministic and rng is not None and 0.0 < kp < 1.0:
        if rng.random() < kp:
            return valid_paths
    else:
        # Deterministic behavior: only keep when kp==1
        if kp >= 1.0:
            return valid_paths

    kept: List[Any] = []
    for p in valid_paths:
        pl = _coerce_path_list(p)
        if pl is not None and pl == original:
            continue
        kept.append(p)
    return kept

def _detect_original_path_and_shorter_valid(
    valid_paths: List[Any],
    *,
    original_depth: int,
) -> Tuple[bool, bool]:
    """
    Returns:
      (has_original, has_shorter_valid)
    where:
      - has_original: original full-depth path [0..original_depth-1] exists in valid_paths
      - has_shorter_valid: exists some other valid path with length < original_depth
    """
    if not valid_paths or not isinstance(original_depth, int) or original_depth <= 0:
        return False, False
    original = list(range(original_depth))
    has_original = False
    has_shorter = False
    for p in valid_paths:
        pl = _coerce_path_list(p)
        if pl is None:
            continue
        if pl == original:
            has_original = True
        else:
            if len(pl) < original_depth:
                has_shorter = True
    return has_original, has_shorter

def _clamp01(x: Any, default: float = 1.0) -> float:
    try:
        v = float(x)
    except Exception:
        v = float(default)
    return max(0.0, min(1.0, v))

def _apply_segment_action(cursor: int, action: Tuple[int, int]) -> Tuple[int, List[int]]:
    """
    action: (size, op) where op in {0,1,2}; repeat means 2x (count=1)
    returns: (new_cursor, appended_tokens)
    """
    size, op = action
    chunk = list(range(cursor, cursor + size))
    new_cursor = cursor + size
    if op == OP_SKIP:
        return new_cursor, []
    if op == OP_EXECUTE:
        return new_cursor, chunk
    # OP_REPEAT
    return new_cursor, chunk + chunk

def parse_path_to_seg_and_ops(
    path: List[int],
    original_depth: int,
    max_pack: int = 4,
    allow_repeat: bool = True,
) -> Optional[Tuple[List[int], List[int]]]:
    """
    Parse a final path (list of executed layer indices, including repeats) into:
    - seg_flip_mask: length original_depth, seg_flip[i]=1 indicates a segment boundary at i
        (i.e., compared to i-1, the segmentation bit flips; boundaries correspond to 0->1 or 1->0)
    - op_labels: length original_depth, op at segment start; -100 elsewhere

    This provides a UNIQUE (canonical) segmentation per path by DP + tie-breaking.
    """
    # DP state: (cursor, pos) -> best parse list[(cursor, size, op)]
    memo: Dict[Tuple[int, int], Optional[List[Tuple[int, int, int]]]] = {}

    def better(a: List[Tuple[int, int, int]], b: List[Tuple[int, int, int]]) -> bool:
        """
        Return True if a is better than b by canonical tie-break:
        - fewer segments (prefer larger packs)
        - fewer skips
        - more repeats (explicit loops)
        - larger pack sizes earlier
        """
        if b is None:
            return True
        if len(a) != len(b):
            return len(a) < len(b)
        a_skips = sum(1 for _, _, op in a if op == OP_SKIP)
        b_skips = sum(1 for _, _, op in b if op == OP_SKIP)
        if a_skips != b_skips:
            return a_skips < b_skips
        a_reps = sum(1 for _, _, op in a if op == OP_REPEAT)
        b_reps = sum(1 for _, _, op in b if op == OP_REPEAT)
        if a_reps != b_reps:
            return a_reps > b_reps
        # lexicographic by -size to prefer larger packs
        a_sizes = [s for _, s, _ in a]
        b_sizes = [s for _, s, _ in b]
        return a_sizes > b_sizes

    def dp(cursor: int, pos: int) -> Optional[List[Tuple[int, int, int]]]:
        key = (cursor, pos)
        if key in memo:
            return memo[key]
        if cursor == original_depth:
            memo[key] = [] if pos == len(path) else None
            return memo[key]
        if pos > len(path):
            memo[key] = None
            return None

        best: Optional[List[Tuple[int, int, int]]] = None
        for size in range(1, max_pack + 1):
            if cursor + size > original_depth:
                continue
            chunk = list(range(cursor, cursor + size))

            # 1) skip
            tail = dp(cursor + size, pos)
            if tail is not None:
                cand = [(cursor, size, OP_SKIP)] + tail
                if best is None or better(cand, best):
                    best = cand

            # 2) execute/keep
            if pos + size <= len(path) and path[pos:pos + size] == chunk:
                tail = dp(cursor + size, pos + size)
                if tail is not None:
                    cand = [(cursor, size, OP_EXECUTE)] + tail
                    if best is None or better(cand, best):
                        best = cand

            # 3) repeat (2x only)
            if allow_repeat and pos + 2 * size <= len(path) and path[pos:pos + 2 * size] == chunk + chunk:
                tail = dp(cursor + size, pos + 2 * size)
                if tail is not None:
                    cand = [(cursor, size, OP_REPEAT)] + tail
                    if best is None or better(cand, best):
                        best = cand

        memo[key] = best
        return best

    parse = dp(0, 0)
    if parse is None:
        return None

    # Build labels
    seg_start = [0] * original_depth
    op_labels = [-100] * original_depth
    for cursor, size, op in parse:
        seg_start[cursor] = 1
        op_labels[cursor] = op

    # Convert start-mask to flip-mask (boundary indicator). seg_flip[0] is always 0.
    seg_flip = [0] * original_depth
    for i in range(1, original_depth):
        if seg_start[i] == 1:
            seg_flip[i] = 1

    # Sanity: reproduce path
    out: List[int] = []
    cur = 0
    for cursor, size, op in parse:
        assert cursor == cur
        cur, appended = _apply_segment_action(cur, (size, op))
        out.extend(appended)
    if cur != original_depth or out != path:
        # Should not happen; treat as invalid
        return None

    return seg_flip, op_labels

def extract_question_and_gt(sample: Dict[str, Any]) -> Tuple[str, str]:
    """
    `merged_mcts_samples.json` stores question and answer fields in several common layouts.
    """
    q = sample.get("question") or sample.get("sample_info", {}).get("question") or ""
    gt = (
        sample.get("gt_ans")
        or sample.get("ground_truth")
        or sample.get("answer")
        or sample.get("sample_info", {}).get("ground_truth")
        or sample.get("sample_info", {}).get("answer")
        or ""
    )
    return q, gt

def extract_initial_score(sample: Dict[str, Any]) -> float:
    """
    Read the baseline score from the merged sample record when available.
    """
    v = (
        sample.get("initial_transition_metric")
        if isinstance(sample, dict)
        else None
    )
    if v is None and isinstance(sample, dict):
        v = sample.get("initial_score")
    if v is None and isinstance(sample, dict) and isinstance(sample.get("evaluation_result"), dict):
        v = sample["evaluation_result"].get("initial_score")
    try:
        return float(v) if v is not None else 0.0
    except Exception:
        return 0.0

class PolarDataset(Dataset):
    """
    Each item corresponds to a question, a segmentation target, and operation labels
    with length `original_depth`.
    """
    def __init__(
        self,
        merged_samples_json: str,
        start_idx: int,
        end_idx: int,
        original_depth: int,
        indices: Optional[List[int]] = None,
        drop_original_path_if_shorter_valid: bool = False,
        keep_original_prob: float = 0.0,
        reweight_original_path_if_shorter_valid: bool = False,
        original_path_weight: float = 1.0,
        anti_original_when_invalid: bool = False,
        per_sample_weight_normalize: bool = False,
        max_paths_per_sample: int = 50,
        max_total_examples: Optional[int] = None,
        seed: int = 42,
    ):
        self.examples: List[Dict[str, Any]] = []
        rnd = random.Random(seed)

        with open(merged_samples_json, "r") as f:
            data = json.load(f)
        samples = data["samples"] if isinstance(data, dict) and "samples" in data else (list(data.values()) if isinstance(data, dict) else data)

        # Choose a deterministic subset either by explicit indices or by slice.
        if indices is not None:
            picked = []
            for ii in indices:
                try:
                    j = int(ii)
                except Exception:
                    continue
                if 0 <= j < len(samples):
                    picked.append(samples[j])
            samples = picked
        else:
            start_idx = min(start_idx, len(samples))
            end_idx = min(end_idx, len(samples))
            samples = samples[start_idx:end_idx]

        def _extract_path_list(p: Any) -> Optional[List[int]]:
            return _coerce_path_list(p)

        empty_valid_cnt = 0
        parsed_ok_cnt = 0
        parsed_fail_cnt = 0

        for s_idx, sample in enumerate(samples):
            question, gt = extract_question_and_gt(sample)
            # Only use final_valid_transitions as requested
            valid_paths = sample.get("final_valid_transitions", []) or []
            # TRAIN-ONLY biasing:
            # - If reweight is enabled: DO NOT drop original path; just assign a lower weight to original-path examples
            #   when shorter valid paths exist for the same sample.
            # - Else: allow soft-drop/hard-drop via keep_original_prob / drop_original_path_if_shorter_valid.
            has_original, has_shorter_valid = _detect_original_path_and_shorter_valid(valid_paths, original_depth=original_depth)
            trigger = bool(has_original and has_shorter_valid)
            original_is_valid = bool(has_original)  # original path appears in valid list <=> considered valid by dataset
            ex_original_weight = _clamp01(original_path_weight, default=1.0)
            if not reweight_original_path_if_shorter_valid:
                enabled = bool(drop_original_path_if_shorter_valid) or (float(keep_original_prob) > 0.0)
                valid_paths = _maybe_drop_original_path_from_valid_paths(
                    valid_paths,
                    original_depth=original_depth,
                    enabled=enabled,
                    keep_original_prob=float(keep_original_prob),
                    rng=rnd,
                    deterministic=False,
                )
            if not valid_paths:
                empty_valid_cnt += 1
                continue

            if len(valid_paths) > max_paths_per_sample:
                valid_paths = rnd.sample(valid_paths, max_paths_per_sample)

            # Collect per-sample examples first so we can optionally normalize total weight per sample.
            # This prevents samples with many valid paths from dominating training.
            sample_examples: List[Dict[str, Any]] = []
            for p in valid_paths:
                p_list = _extract_path_list(p)
                if p_list is None:
                    continue
                parsed = parse_path_to_seg_and_ops(p_list, original_depth=original_depth, max_pack=4, allow_repeat=True)
                if parsed is None:
                    parsed_fail_cnt += 1
                    continue
                parsed_ok_cnt += 1
                seg_flip, op_labels = parsed
                is_original = (p_list == list(range(original_depth)))
                weight = ex_original_weight if (reweight_original_path_if_shorter_valid and trigger and is_original) else 1.0
                anti_orig_active = bool(anti_original_when_invalid and (not original_is_valid))
                sample_examples.append({
                    "question": question,
                    "gt": gt,
                    "seg_flip": seg_flip,
                    "op_labels": op_labels,
                    "path_len": len(p_list),
                    "is_original_path": bool(is_original),
                    "original_path_is_valid": bool(original_is_valid),
                    "anti_original_active": bool(anti_orig_active),
                    "trigger_shorter_valid": bool(trigger),
                    "weight": float(weight),
                })
                if max_total_examples is not None and (len(self.examples) + len(sample_examples)) >= max_total_examples:
                    break

            if not sample_examples:
                continue

            if per_sample_weight_normalize:
                # Make each sample contribute ~1 total weight (after sampling valid paths).
                # This is crucial when some samples have very large numbers of valid paths.
                scale = 1.0 / float(len(sample_examples))
                for ex in sample_examples:
                    ex["weight"] = float(ex.get("weight", 1.0)) * scale

            self.examples.extend(sample_examples)

            if max_total_examples is not None and len(self.examples) >= max_total_examples:
                break

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.examples[idx]

def collate_fn_polar(batch: List[Dict[str, Any]]):
    questions = [b["question"] for b in batch]
    seg_flip = torch.tensor([b["seg_flip"] for b in batch], dtype=torch.float32)
    op_labels = torch.tensor([b["op_labels"] for b in batch], dtype=torch.long)
    path_lens = torch.tensor([b.get("path_len", 0) for b in batch], dtype=torch.float32)
    weights = torch.tensor([b.get("weight", 1.0) for b in batch], dtype=torch.float32)
    anti_orig = torch.tensor([1.0 if b.get("anti_original_active", False) else 0.0 for b in batch], dtype=torch.float32)
    return questions, seg_flip, op_labels, path_lens, weights, anti_orig

def actions_to_path(actions: List[Tuple[str, int, int]], original_depth: int) -> List[int]:
    cursor = 0
    out: List[int] = []
    for act_type, size, cnt in actions:
        chunk = list(range(cursor, cursor + size))
        cursor = cursor + size
        if act_type == "keep":
            out.extend(chunk)
        elif act_type == "repeat":
            # cnt=1 means 2x
            for _ in range(cnt + 1):
                out.extend(chunk)
        # skip: nothing
    # cursor should reach original_depth
    return out

def estimate_path_length_from_actions(actions: List[Tuple[str, int, int]]) -> int:
    """
    Estimate final executable path length (number of layer indices) from actions.
    keep adds size, repeat adds (cnt+1)*size, skip adds 0.
    """
    total = 0
    for act_type, size, cnt in actions:
        if act_type == "keep":
            total += size
        elif act_type == "repeat":
            total += (cnt + 1) * size
    return total

