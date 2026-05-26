import json
import math
import os
import random
from typing import Any, List, Optional

import numpy as np
import torch

OP_SKIP = 0
OP_EXECUTE = 1
OP_REPEAT = 2
DEFAULT_DATA_ROOT = "./data"


def _format_prob_for_suffix(p: float) -> str:
    """
    Stable short float formatting for filenames.
    """
    try:
        v = float(p)
    except Exception:
        v = 0.0
    v = max(0.0, min(1.0, v))
    s = f"{v:.3f}".rstrip("0").rstrip(".")
    return s if s else "0"

def is_finite(x: Any) -> bool:
    try:
        if isinstance(x, torch.Tensor):
            return bool(torch.isfinite(x).all().item())
        v = float(x)
        return math.isfinite(v)
    except Exception:
        return False

def maybe_lr_backoff(optimizer: torch.optim.Optimizer, factor: float, min_lr: float = 0.0) -> float:
    """
    Multiply LR by factor (0<factor<1). Returns new LR (of the first param group).
    """
    try:
        f = float(factor)
    except Exception:
        return float(optimizer.param_groups[0].get("lr", 0.0))
    if not (0.0 < f < 1.0):
        return float(optimizer.param_groups[0].get("lr", 0.0))
    min_lr = max(0.0, float(min_lr or 0.0))
    for pg in optimizer.param_groups:
        lr0 = float(pg.get("lr", 0.0))
        pg["lr"] = max(min_lr, lr0 * f)
    return float(optimizer.param_groups[0].get("lr", 0.0))

def _nan_mitigation_suffix(args) -> str:
    """
    Encode enabled training-stability options into output filenames.
    """
    parts: List[str] = []
    if bool(getattr(args, "nan_guard", False)):
        parts.append("ng")
    clip = float(getattr(args, "grad_clip_norm", 0.0) or 0.0)
    if clip > 0:
        parts.append(f"clip{_format_prob_for_suffix(clip)}")
    if bool(getattr(args, "use_amp", False)):
        parts.append("amp")
    backoff = float(getattr(args, "nan_lr_backoff", 0.0) or 0.0)
    if bool(getattr(args, "nan_guard", False)) and 0.0 < backoff < 1.0:
        parts.append(f"bk{_format_prob_for_suffix(backoff)}")
    tag = str(getattr(args, "run_tag", "") or "").strip()
    if tag:
        parts.append(f"tag{tag}")
    return "" if not parts else "_" + "_".join(parts)

def maybe_delete_checkpoint(path: Optional[str], enabled: bool) -> None:
    """
    Only deletes if enabled and the file exists.
    """
    if not enabled:
        return
    if not path:
        return
    try:
        if os.path.exists(path) and os.path.isfile(path):
            os.remove(path)
            print(f"[Cleanup] Deleted checkpoint after eval: {path}")
    except Exception as e:
        print(f"[Cleanup] Warning: failed to delete checkpoint: {path}. Error: {e}")

def infer_original_depth(model_path: str) -> int:
    name = str(model_path or "").lower()
    if ("qwen1.5" in name or "qwen1_5" in name) and "moe" in name:
        return 24
    if ("qwen2.5" in name or "qwen2_5" in name) and "3b" in name:
        return 36
    if "qwen3" in name and "8b" in name:
        return 36
    if "llama-3.2" in name and "3b" in name:
        return 28
    raise ValueError(
        "Unsupported model_path for this release. Supported models: "
        "LLaMA-3.2-3B-Instruct, Qwen1.5-MoE-A2.7B-Chat, "
        "Qwen2.5-3B-Instruct, and Qwen3-8B."
    )

def resolve_dart_base_path(model_path: str, data_root: Optional[str] = None) -> str:
    """Resolve the directory containing per-diff `merged_mcts_samples.json` files."""
    root = data_root or os.environ.get("DART_MATH_RESULTS_ROOT")
    if not root:
        raise ValueError("--data_root or DART_MATH_RESULTS_ROOT must be set.")
    root = os.path.abspath(os.path.expanduser(root))
    model_rel = str(model_path).strip("/")
    if os.path.normpath(root).endswith(os.path.normpath(model_rel)):
        return root
    return os.path.join(root, model_rel)


def merged_samples_path(base_path: str, diff: int) -> str:
    """Return the supervision file path for one DART-Math difficulty level."""
    return os.path.join(base_path, f"dart-math-diff-{int(diff)}", "merged_mcts_samples.json")

def _infer_model_id_for_naming(model_path: str) -> str:
    """
    Infer a stable model_id for filename suffixes.
    - If model_path looks like `org/model`, use the last component.
    - If empty/unknown, default to Qwen2.5-3B-Instruct.
    """
    s = str(model_path or "").strip()
    if not s:
        return "Qwen2.5-3B-Instruct"
    # handle local paths too
    if "/" in s:
        tail = s.split("/")[-1].strip()
        return tail or "Qwen2.5-3B-Instruct"
    return s

def _qwen_model_suffix_for_naming(model_path: str) -> str:
    """
    Keep filenames compact for the default Qwen2.5 model and explicit for
    other Qwen variants.
    """
    if "Qwen" not in str(model_path):
        return ""
    default_id = "Qwen2.5-3B-Instruct"
    mid = _infer_model_id_for_naming(model_path)
    return "" if mid == default_id else f"_model{mid}"

def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def print_all_args(args) -> None:
    """Print CLI arguments in a stable, copy-friendly format."""
    try:
        payload = vars(args)
    except Exception:
        payload = {"args": str(args)}
    try:
        print("\n" + "=" * 80)
        print("[Args] Full configuration")
        print("=" * 80)
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
        print("=" * 80 + "\n")
    except Exception:
        print("[Args] " + str(args))


def _train_mode_suffix(args, train_mode_suffix: Optional[str] = None) -> str:
    if train_mode_suffix:
        return train_mode_suffix
    if getattr(args, "target_diff", None) is not None:
        return f"_target_diff{int(args.target_diff)}"
    return "_all_diffs"


def _method_tag(args) -> str:
    return "polar_lenpref" if getattr(args, "policy_mode", "polar") == "polar_lenpref" else "polar"


def _run_suffixes(args, *, include_eval: bool = False, diff: Optional[int] = None) -> str:
    lenpref_enabled = getattr(args, "policy_mode", "polar") == "polar_lenpref"
    lr_sched_name = str(getattr(args, "lr_scheduler", "none") or "none").lower()
    warmup_steps = max(0, int(getattr(args, "warmup_steps", 0) or 0))

    pieces = []
    if lenpref_enabled:
        pieces.append(f"_beta{getattr(args, 'lenpref_beta')}")
    if include_eval:
        lpen = float(getattr(args, "len_penalty", 0.0) or 0.0)
        if lpen > 0:
            pieces.append(f"_lpen{lpen}")
        pieces.append(f"_topkops{getattr(args, 'top_k_ops')}")
        pieces.append(f"_beam{getattr(args, 'beam_size')}")

    keep_prob = float(getattr(args, "keep_original_prob", 0.0) or 0.0)
    drop_enabled = bool(getattr(args, "drop_original_path_if_shorter_valid", False)) or keep_prob > 0.0
    if include_eval and drop_enabled:
        suffix = "_droporig"
        if keep_prob > 0.0:
            suffix += f"_keeporig{_format_prob_for_suffix(keep_prob)}"
        pieces.append(suffix)

    if bool(getattr(args, "reweight_original_path_if_shorter_valid", False)):
        pieces.append(f"_origw{_format_prob_for_suffix(float(getattr(args, 'original_path_weight', 1.0)))}")
    if float(getattr(args, "anti_original_lambda", 0.0) or 0.0) > 0.0:
        pieces.append(f"_antiorig{_format_prob_for_suffix(float(getattr(args, 'anti_original_lambda', 0.0)))}")
    if bool(getattr(args, "per_sample_weight_normalize", False)):
        pieces.append("_psnorm")
    if getattr(args, "max_paths_per_sample", None) is not None:
        pieces.append(f"_mpps{int(getattr(args, 'max_paths_per_sample', 0))}")
    if lr_sched_name != "none":
        pieces.append(f"_sched{lr_sched_name}")
    if warmup_steps > 0:
        pieces.append(f"_warmup{warmup_steps}")
    if include_eval and bool(getattr(args, "eval_cache_breakdown", False)):
        pieces.append("_cachebreakdown")
    return "".join(pieces)


def checkpoint_path_for_args(args, train_mode_suffix: Optional[str] = None) -> str:
    suffix = _train_mode_suffix(args, train_mode_suffix)
    method_tag = _method_tag(args)
    train_mode_dir = suffix.lstrip("_")
    ckpt_dir = os.path.join(str(args.save_dir), method_tag, train_mode_dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    model_suffix = _qwen_model_suffix_for_naming(args.model_path)
    run_suffix = _run_suffixes(args)
    nan_suffix = _nan_mitigation_suffix(args)
    return os.path.join(
        ckpt_dir,
        f"{method_tag}_policy_epochs{args.num_epochs}_bs{args.batch_size}_lr{args.learning_rate}"
        f"{nan_suffix}{model_suffix}{run_suffix}_seed{args.seed}{suffix}.pt",
    )


def eval_results_path_for_args(args, *, diff: int, train_mode_suffix: Optional[str] = None) -> str:
    suffix = _train_mode_suffix(args, train_mode_suffix)
    method_tag = _method_tag(args)
    train_mode_dir = suffix.lstrip("_")
    out_dir = os.path.join(str(args.save_dir), method_tag, train_mode_dir)
    os.makedirs(out_dir, exist_ok=True)
    model_suffix = _qwen_model_suffix_for_naming(args.model_path)
    lenpref_enabled = getattr(args, "policy_mode", "polar") == "polar_lenpref"
    beta_suffix = f"_beta{args.lenpref_beta}" if lenpref_enabled else ""
    lpen = float(getattr(args, "len_penalty", 0.0) or 0.0)
    lpen_suffix = f"_lpen{lpen}" if lpen > 0 else ""
    topkops_suffix = f"_topkops{args.top_k_ops}"
    beam_suffix = f"_beam{args.beam_size}"

    keep_prob = float(getattr(args, "keep_original_prob", 0.0) or 0.0)
    drop_enabled = bool(getattr(args, "drop_original_path_if_shorter_valid", False)) or keep_prob > 0.0
    plugin_suffix = "_droporig" if drop_enabled else ""
    if drop_enabled and keep_prob > 0.0:
        plugin_suffix += f"_keeporig{_format_prob_for_suffix(keep_prob)}"
    origw_suffix = ""
    if bool(getattr(args, "reweight_original_path_if_shorter_valid", False)):
        origw_suffix = f"_origw{_format_prob_for_suffix(float(getattr(args, 'original_path_weight', 1.0)))}"
    antiorig_suffix = ""
    if float(getattr(args, "anti_original_lambda", 0.0) or 0.0) > 0.0:
        antiorig_suffix = f"_antiorig{_format_prob_for_suffix(float(getattr(args, 'anti_original_lambda', 0.0)))}"
    psnorm_suffix = "_psnorm" if bool(getattr(args, "per_sample_weight_normalize", False)) else ""
    mpps_suffix = f"_mpps{int(getattr(args, 'max_paths_per_sample', 0))}" if getattr(args, "max_paths_per_sample", None) is not None else ""
    lr_sched_name = str(getattr(args, "lr_scheduler", "none") or "none").lower()
    sched_suffix = f"_sched{lr_sched_name}" if lr_sched_name != "none" else ""
    warmup_steps = max(0, int(getattr(args, "warmup_steps", 0) or 0))
    warmup_suffix = f"_warmup{warmup_steps}" if warmup_steps > 0 else ""
    breakdown_suffix = "_cachebreakdown" if bool(getattr(args, "eval_cache_breakdown", False)) else ""
    nan_suffix = _nan_mitigation_suffix(args)
    return os.path.join(
        out_dir,
        f"eval_results_{method_tag}_epochs{args.num_epochs}_bs{args.batch_size}_lr{args.learning_rate}"
        f"{nan_suffix}{beta_suffix}{lpen_suffix}{topkops_suffix}{beam_suffix}_diff{diff}_seed{args.seed}_num_samples{args.num_samples}"
        f"{model_suffix}{plugin_suffix}{origw_suffix}{antiorig_suffix}{psnorm_suffix}{mpps_suffix}{sched_suffix}{warmup_suffix}{breakdown_suffix}{suffix}.json",
    )
