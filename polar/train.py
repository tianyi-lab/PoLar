from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

from .config import (
    OP_EXECUTE,
    checkpoint_path_for_args,
    is_finite,
    maybe_lr_backoff,
    merged_samples_path,
    resolve_dart_base_path,
    set_random_seed,
)
from .data import PolarDataset, collate_fn_polar
from .model import PolarPredictor


def train_polar(args):
    """
    Train Polar segmentation and operation predictors from valid paths.
    """
    set_random_seed(args.seed)

    base_path = resolve_dart_base_path(args.model_path, getattr(args, "data_root", None))

    if args.target_diff is not None:
        diffs = [args.target_diff]
        train_mode_suffix = f"_target_diff{args.target_diff}"
    else:
        # If no target_diff is provided, train on all supported difficulty levels.
        if not args.eval_all_diffs:
            raise ValueError("Polar training: set --target_diff (1-5) OR use --eval_all_diffs to train on diff1-5.")
        diffs = list(range(1, 6))
        train_mode_suffix = "_all_diffs"

    merged_files = [
        merged_samples_path(base_path, d)
        for d in diffs
    ]
    print(f"[Polar] Training from {len(merged_files)} file(s):")
    for mf in merged_files:
        print(f"  - {mf}")

    # Build datasets (concat across diffs if needed)
    train_datasets = []
    for mf in merged_files:
        train_indices = None
        train_datasets.append(
            PolarDataset(
                merged_samples_json=mf,
                start_idx=0,
                end_idx=1500 if args.no_validation else 1250,
                indices=train_indices,
                original_depth=args.original_depth,
                drop_original_path_if_shorter_valid=bool(getattr(args, "drop_original_path_if_shorter_valid", False)),
                keep_original_prob=float(getattr(args, "keep_original_prob", 0.0)),
                reweight_original_path_if_shorter_valid=bool(getattr(args, "reweight_original_path_if_shorter_valid", False)),
                original_path_weight=float(getattr(args, "original_path_weight", 1.0)),
                anti_original_when_invalid=(float(getattr(args, "anti_original_lambda", 0.0)) > 0.0),
                per_sample_weight_normalize=bool(getattr(args, "per_sample_weight_normalize", False)),
                max_paths_per_sample=args.max_paths_per_sample,
                max_total_examples=args.max_total_examples,
                seed=args.seed,
            )
        )
    train_ds = ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_polar)

    val_loader = None
    if not args.no_validation:
        val_datasets = []
        for mf in merged_files:
            val_indices = None
            val_datasets.append(
                PolarDataset(
                    merged_samples_json=mf,
                    start_idx=1250,
                    end_idx=1500,
                    indices=val_indices,
                    original_depth=args.original_depth,
                    drop_original_path_if_shorter_valid=bool(getattr(args, "drop_original_path_if_shorter_valid", False)),
                    keep_original_prob=float(getattr(args, "keep_original_prob", 0.0)),
                    reweight_original_path_if_shorter_valid=bool(getattr(args, "reweight_original_path_if_shorter_valid", False)),
                    original_path_weight=float(getattr(args, "original_path_weight", 1.0)),
                    anti_original_when_invalid=(float(getattr(args, "anti_original_lambda", 0.0)) > 0.0),
                    per_sample_weight_normalize=bool(getattr(args, "per_sample_weight_normalize", False)),
                    max_paths_per_sample=args.max_paths_per_sample,
                    max_total_examples=args.max_val_examples,
                    seed=args.seed,
                )
            )
        val_ds = ConcatDataset(val_datasets) if len(val_datasets) > 1 else val_datasets[0]
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_polar)

    model = PolarPredictor(num_layers=args.original_depth, d_model=args.polar_d_model, nheads=args.polar_heads, n_layer_blocks=args.polar_layers).cuda()
    opt_eps = float(getattr(args, "optimizer_eps", 1e-8) or 1e-8)
    opt_wd = float(getattr(args, "weight_decay", 0.0) or 0.0)
    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate, eps=opt_eps, weight_decay=opt_wd)
    # Optional LR scheduler (warmup + decay). TRAIN-ONLY.
    lr_scheduler = None
    lr_sched_name = str(getattr(args, "lr_scheduler", "none") or "none").lower()
    warmup_steps = int(getattr(args, "warmup_steps", 0) or 0)
    warmup_steps = max(0, warmup_steps)

    bce_none = nn.BCEWithLogitsLoss(reduction="none")
    ce_none = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")

    lenpref_enabled = (args.policy_mode == "polar_lenpref")
    method_tag = "polar_lenpref" if lenpref_enabled else "polar"

    if args.use_wandb and WANDB_AVAILABLE and wandb is not None:
        wandb_run_name = args.wandb_run_name or f"{method_tag}_diff{args.target_diff}_ep{args.num_epochs}_bs{args.batch_size}_lr{args.learning_rate}_seed{args.seed}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, entity=args.wandb_entity, config=vars(args))

    best_val = float("inf")
    best_state = None

    total_steps = int(max(1, args.num_epochs * max(1, len(train_loader))))
    warmup_steps = min(warmup_steps, max(0, total_steps - 1))
    if lr_sched_name != "none":
        import math as _math

        def _lr_lambda(step: int) -> float:
            step = int(step)
            if warmup_steps > 0 and step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            if total_steps <= warmup_steps:
                return 1.0
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            progress = max(0.0, min(1.0, progress))
            if lr_sched_name == "linear":
                return max(0.0, 1.0 - progress)
            if lr_sched_name == "cosine":
                return 0.5 * (1.0 + _math.cos(_math.pi * progress))
            # Fallback: constant
            return 1.0

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)

    for epoch in range(args.num_epochs):
        model.train()
        total = 0.0
        for step, (qs, seg_t, op_t, path_lens, ex_w, anti_orig) in enumerate(train_loader):
            seg_t = seg_t.cuda()  # seg_flip targets
            op_t = op_t.cuda()
            path_lens = path_lens.cuda()
            ex_w = ex_w.cuda()
            anti_orig = anti_orig.cuda()
            optimizer.zero_grad()

            use_amp = bool(getattr(args, "use_amp", False))
            nan_guard = bool(getattr(args, "nan_guard", False))
            grad_clip = float(getattr(args, "grad_clip_norm", 0.0) or 0.0)
            nan_backoff = float(getattr(args, "nan_lr_backoff", 0.0) or 0.0)
            min_lr = float(getattr(args, "min_lr", 0.0) or 0.0)

            if not hasattr(train_polar, "_scaler"):
                train_polar._scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

            with torch.cuda.amp.autocast(enabled=use_amp):
                seg_logits, op_logits = model(qs)

            # segmentation loss: seg_flip[0] is always 0, ignore it
            seg_elem = bce_none(seg_logits[:, 1:], seg_t[:, 1:])  # (B, D-1)
            seg_loss_per = seg_elem.mean(dim=1)  # (B,)

            # op loss: only segment starts have labels, rest are -100
            B, D, _ = op_logits.shape
            ce_tok = ce_none(op_logits.view(-1, 3), op_t.view(-1)).view(B, D)  # (B,D), ignored positions -> 0
            mask = (op_t != -100).float()
            op_sum = (ce_tok * mask).sum(dim=1)
            op_cnt = mask.sum(dim=1).clamp(min=1.0)
            op_loss_per = op_sum / op_cnt  # (B,)

            loss_per = seg_loss_per + op_loss_per

            # Length preference weighting: shorter valid paths get larger weights.
            # w = exp(-beta * L); normalized to mean 1 to stabilize loss scale.
            if lenpref_enabled:
                w = torch.exp(-args.lenpref_beta * path_lens)
                w = w / (w.mean() + 1e-8)
            else:
                w = torch.ones_like(loss_per)

            # Combine with example-level weight (e.g., downweight original path when shorter valid exists)
            w = w * ex_w
            w = w / (w.mean() + 1e-8)

            loss = (w * loss_per).mean()

            # ------------------------------------------------------------
            # B) Anti-original penalty (TRAIN-ONLY)
            # ------------------------------------------------------------
            # If original path is NOT valid for this sample, discourage the model from collapsing to "all keep".
            # Heuristic: penalize mean probability of OP_EXECUTE ("keep") across layers.
            anti_lambda = float(getattr(args, "anti_original_lambda", 0.0))
            if anti_lambda > 0.0:
                # op_logits: (B, D, 3)
                op_probs = torch.softmax(op_logits, dim=-1)
                keep_prob = op_probs[:, :, OP_EXECUTE].mean(dim=1)  # (B,)
                denom = anti_orig.sum().clamp(min=1.0)
                anti_loss = (anti_orig * keep_prob).sum() / denom
                loss = loss + anti_lambda * anti_loss

            if nan_guard and (not is_finite(loss)):
                new_lr = maybe_lr_backoff(optimizer, nan_backoff, min_lr=min_lr)
                if step % 20 == 0:
                    print(f"[NaN-Guard][{method_tag}] Non-finite loss at epoch={epoch} step={step}. Skipping. lr={new_lr}")
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler = train_polar._scaler
            if use_amp:
                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()
            total += float(loss.item())

            if step % 20 == 0:
                print(f"[{method_tag}] Epoch {epoch} Step {step} Loss {loss.item():.4f}")
                if args.use_wandb and WANDB_AVAILABLE and wandb is not None:
                    wandb.log({"train/loss": loss.item()})

        avg = total / max(1, len(train_loader))
        print(f"[{method_tag}] Epoch {epoch} Avg Loss {avg:.4f}")
        if args.use_wandb and WANDB_AVAILABLE and wandb is not None:
            wandb.log({"train/epoch_loss": avg})

        if val_loader is not None:
            model.eval()
            vtotal = 0.0
            with torch.no_grad():
                for (qs, seg_t, op_t, path_lens, ex_w, anti_orig) in val_loader:
                    seg_t = seg_t.cuda()
                    op_t = op_t.cuda()
                    path_lens = path_lens.cuda()
                    ex_w = ex_w.cuda()
                    anti_orig = anti_orig.cuda()
                    seg_logits, op_logits = model(qs)

                    seg_elem = bce_none(seg_logits[:, 1:], seg_t[:, 1:])
                    seg_loss_per = seg_elem.mean(dim=1)
                    B, D, _ = op_logits.shape
                    ce_tok = ce_none(op_logits.view(-1, 3), op_t.view(-1)).view(B, D)
                    mask = (op_t != -100).float()
                    op_sum = (ce_tok * mask).sum(dim=1)
                    op_cnt = mask.sum(dim=1).clamp(min=1.0)
                    op_loss_per = op_sum / op_cnt
                    loss_per = seg_loss_per + op_loss_per
                    if lenpref_enabled:
                        w = torch.exp(-args.lenpref_beta * path_lens)
                        w = w / (w.mean() + 1e-8)
                    else:
                        w = torch.ones_like(loss_per)
                    w = w * ex_w
                    w = w / (w.mean() + 1e-8)
                    vloss = (w * loss_per).mean()
                    anti_lambda = float(getattr(args, "anti_original_lambda", 0.0))
                    if anti_lambda > 0.0:
                        op_probs = torch.softmax(op_logits, dim=-1)
                        keep_prob = op_probs[:, :, OP_EXECUTE].mean(dim=1)
                        denom = anti_orig.sum().clamp(min=1.0)
                        anti_loss = (anti_orig * keep_prob).sum() / denom
                        vloss = vloss + anti_lambda * anti_loss
                    vtotal += float(vloss.item())
            vavg = vtotal / max(1, len(val_loader))
            print(f"[{method_tag}] Epoch {epoch} Val Loss {vavg:.4f}")
            if args.use_wandb and WANDB_AVAILABLE and wandb is not None:
                wandb.log({"val/loss": vavg})
            if vavg < best_val:
                best_val = vavg
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                print(f"[{method_tag}] New best val {best_val:.4f}")

    ckpt = checkpoint_path_for_args(args, train_mode_suffix=train_mode_suffix)
    if best_state is not None and not args.no_validation:
        torch.save(best_state, ckpt)
    else:
        cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        torch.save(cpu_state, ckpt)
    print(f"[{method_tag}] Saved checkpoint: {ckpt}")

    if args.use_wandb and WANDB_AVAILABLE and wandb is not None:
        wandb.finish()

    return ckpt
