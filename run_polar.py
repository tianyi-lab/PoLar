import argparse
import os

from polar.config import (
    DEFAULT_DATA_ROOT,
    checkpoint_path_for_args,
    infer_original_depth,
    maybe_delete_checkpoint,
    print_all_args,
    resolve_dart_base_path,
)
from polar.eval import evaluate_polar
from polar.train import train_polar


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train/evaluate the Polar depth-routing policy.")

    paths = parser.add_argument_group("paths and model")
    paths.add_argument("--model_path", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    paths.add_argument(
        "--data_root",
        type=str,
        default=DEFAULT_DATA_ROOT,
        help="Root directory containing per-model POLAR supervision folders.",
    )
    paths.add_argument(
        "--save_dir",
        type=str,
        default=os.environ.get("OUTPUT_DIR", "outputs"),
        help="Directory for checkpoints and evaluation JSON files.",
    )
    paths.add_argument(
        "--hf_cache_dir",
        type=str,
        default=os.environ.get("HF_HOME") or os.environ.get("TRANSFORMERS_CACHE"),
        help="Optional Hugging Face cache directory.",
    )

    training = parser.add_argument_group("training")
    training.add_argument("--num_epochs", type=int, default=20)
    training.add_argument("--batch_size", type=int, default=16)
    training.add_argument("--learning_rate", type=float, default=1e-4)
    training.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    training.add_argument("--no_validation", action="store_true", help="Disable validation split and save final epoch checkpoint")
    training.add_argument("--policy_mode", type=str, default="polar", choices=["polar", "polar_lenpref"])
    training.add_argument("--max_paths_per_sample", type=int, default=50, help="Max valid paths sampled per sample for training")
    training.add_argument("--max_total_examples", type=int, default=None, help="Cap total training examples")
    training.add_argument("--max_val_examples", type=int, default=20000, help="Cap total validation examples")
    training.add_argument("--polar_d_model", type=int, default=256, help="Polar model hidden size")
    training.add_argument("--polar_heads", type=int, default=4, help="Polar model attention heads")
    training.add_argument("--polar_layers", type=int, default=2, help="Polar layer-encoder blocks")
    training.add_argument("--lenpref_beta", type=float, default=0.05, help="polar_lenpref weight = exp(-beta * path_len)")
    training.add_argument("--drop_original_path_if_shorter_valid", action="store_true", help="Drop full-depth path when a shorter valid path exists")
    training.add_argument("--keep_original_prob", type=float, default=0.0, help="Probability of keeping full-depth path when dropping is enabled")
    training.add_argument("--reweight_original_path_if_shorter_valid", action="store_true", help="Downweight full-depth path instead of dropping it")
    training.add_argument("--original_path_weight", type=float, default=1.0, help="Weight multiplier for full-depth path examples")
    training.add_argument("--anti_original_lambda", type=float, default=0.0, help="Penalty against collapsing to the full-depth path")
    training.add_argument("--per_sample_weight_normalize", action="store_true", help="Normalize total training weight per sample")

    evaluation = parser.add_argument_group("evaluation")
    evaluation.add_argument("--eval", action="store_true", help="Skip training and only run evaluation")
    evaluation.add_argument("--max_new_tokens", type=int, default=50, help="Max new tokens for online evaluation generation")
    evaluation.add_argument("--num_samples", type=int, default=500)
    evaluation.add_argument("--eval_all_diffs", action="store_true", help="Evaluate/train on diff1-5 instead of a single target diff")
    evaluation.add_argument("--target_diff", type=int, default=None, choices=[1, 2, 3, 4, 5], help="Use only this difficulty level")
    evaluation.add_argument("--checkpoint_path", type=str, default=None, help="Path to model checkpoint for evaluation")
    evaluation.add_argument("--trust_valid_cache", action="store_true", default=True, help="Treat MCTS final_valid_transitions as correct during eval")
    evaluation.add_argument("--no_trust_valid_cache", action="store_false", dest="trust_valid_cache", help="Re-run online eval on cache hits")
    evaluation.add_argument("--seg_threshold", type=float, default=0.5, help="Polar decode threshold")
    evaluation.add_argument("--beam_size", type=int, default=5, help="Polar decode beam size")
    evaluation.add_argument("--top_k_ops", type=int, default=2, help="Per-segment top-k ops to expand")
    evaluation.add_argument("--top_k_paths", type=int, default=5, help="Number of candidate paths to evaluate per sample")
    evaluation.add_argument("--len_penalty", type=float, default=0.0, help="Decode rerank length penalty")
    evaluation.add_argument("--eval_cache_breakdown", action="store_true", help="Print and save cache vs online eval breakdown")

    output = parser.add_argument_group("logging and output")
    output.add_argument("--use_wandb", action="store_true", help="Enable wandb logging for training metrics")
    output.add_argument("--wandb_project", type=str, default="mcts_reward_predictor", help="Wandb project name")
    output.add_argument("--wandb_run_name", type=str, default=None, help="Wandb run name")
    output.add_argument("--wandb_entity", type=str, default=None, help="Wandb entity")
    output.add_argument("--wandb_dir", type=str, default=None, help="Directory to save wandb data")
    output.add_argument("--delete_checkpoint_after_eval", action="store_true", help="Delete checkpoint after evaluation finishes")
    output.add_argument("--run_tag", type=str, default="", help="Optional filename suffix tag")

    stability = parser.add_argument_group("training stability")
    stability.add_argument("--lr_scheduler", type=str, default="none", choices=["none", "linear", "cosine"], help="Optional LR scheduler")
    stability.add_argument("--warmup_steps", type=int, default=0, help="LR scheduler warmup steps")
    stability.add_argument("--nan_guard", action="store_true", help="Skip non-finite loss steps")
    stability.add_argument("--nan_lr_backoff", type=float, default=0.0, help="LR multiplier after non-finite loss")
    stability.add_argument("--min_lr", type=float, default=0.0, help="Minimum LR for nan_lr_backoff")
    stability.add_argument("--grad_clip_norm", type=float, default=0.0, help="Global grad clipping norm")
    stability.add_argument("--use_amp", action="store_true", help="Enable CUDA AMP")
    stability.add_argument("--optimizer_eps", type=float, default=1e-8, help="AdamW epsilon")
    stability.add_argument("--weight_decay", type=float, default=0.0, help="AdamW weight decay")
    return parser


def normalize_args(args):
    args.data_root = resolve_dart_base_path(args.model_path, args.data_root)
    args.save_dir = os.path.abspath(os.path.expanduser(args.save_dir))

    if args.hf_cache_dir:
        cache_dir = os.path.abspath(os.path.expanduser(args.hf_cache_dir))
        os.environ["HF_HOME"] = cache_dir
        os.environ["TRANSFORMERS_CACHE"] = cache_dir
        os.environ["HF_DATASETS_CACHE"] = cache_dir

    if args.use_wandb and args.wandb_dir:
        os.environ["WANDB_DIR"] = os.path.abspath(os.path.expanduser(args.wandb_dir))
        os.makedirs(os.environ["WANDB_DIR"], exist_ok=True)

    args.original_depth = int(infer_original_depth(args.model_path))
    os.makedirs(args.save_dir, exist_ok=True)
    return args


def main() -> None:
    args = normalize_args(build_arg_parser().parse_args())
    print(f"[Config] original_depth={args.original_depth} model_path={args.model_path}")
    print(f"[Config] data_root={args.data_root}")
    print(f"[Config] save_dir={args.save_dir}")
    print_all_args(args)

    if args.eval and not args.checkpoint_path:
        expected_ckpt = checkpoint_path_for_args(args)
        if os.path.exists(expected_ckpt) and os.path.isfile(expected_ckpt):
            args.checkpoint_path = expected_ckpt
            print(f"[Polar] Auto-selected existing final checkpoint: {args.checkpoint_path}")
        else:
            raise ValueError(
                "--checkpoint_path is required for --eval unless the expected checkpoint exists. "
                f"Expected: {expected_ckpt}"
            )

    if not args.eval:
        existing_ckpt = checkpoint_path_for_args(args)
        if os.path.exists(existing_ckpt) and os.path.isfile(existing_ckpt):
            print(f"[Polar] Found existing final checkpoint, skip training: {existing_ckpt}")
            args.checkpoint_path = existing_ckpt
            args.eval = True

    if args.eval:
        evaluate_polar(args)
        maybe_delete_checkpoint(args.checkpoint_path, bool(args.delete_checkpoint_after_eval))
    else:
        ckpt = train_polar(args)
        args.checkpoint_path = ckpt
        print("\n[Polar] Training complete. Starting evaluation...")
        evaluate_polar(args)
        maybe_delete_checkpoint(ckpt, bool(args.delete_checkpoint_after_eval))


if __name__ == "__main__":
    main()
