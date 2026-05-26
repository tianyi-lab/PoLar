import json
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import torch

from dart_math.eval import EvaluatorMathBatch
from llm_depth_router.model import get_model, get_tokenizer, setup_custom_path

from .config import checkpoint_path_for_args, eval_results_path_for_args, merged_samples_path, resolve_dart_base_path
from .data import actions_to_path, estimate_path_length_from_actions, extract_initial_score, extract_question_and_gt
from .model import PolarPredictor, decode_polar_to_actions

QWEN3_THINK_END_TOKEN_ID = 151668
_ONLINE_LLM = None
_ONLINE_TOKENIZER = None


def _is_qwen3_model_path(model_path: str) -> bool:
    try:
        return "qwen3" in str(model_path).lower()
    except Exception:
        return False

def _is_qwen15_moe_chat_model_path(model_path: str) -> bool:
    try:
        s = str(model_path).lower()
        return ("qwen1.5" in s or "qwen1_5" in s) and ("moe" in s) and ("chat" in s)
    except Exception:
        return False

def _is_qwen25_instruct_model_path(model_path: str) -> bool:
    """
    Qwen2.5 Instruct models generally require chat templates for best behavior.
    """
    try:
        s = str(model_path).lower()
        return ("qwen2.5" in s or "qwen2_5" in s) and ("instruct" in s)
    except Exception:
        return False

def _qwen25_apply_chat_template(tokenizer, prompt_text: str) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt_text},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt_text

def _qwen15_moe_apply_chat_template(tokenizer, prompt_text: str) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt_text},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt_text

def _qwen3_apply_chat_template(tokenizer, prompt_text: str) -> str:
    messages = [{"role": "user", "content": prompt_text}]
    if hasattr(tokenizer, "apply_chat_template"):
        fn = tokenizer.apply_chat_template
        try:
            return fn(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return fn(messages, tokenize=False, add_generation_prompt=True)
    return prompt_text

def _qwen3_split_thinking(output_ids: List[int], tokenizer) -> Tuple[str, str]:
    try:
        idx = len(output_ids) - output_ids[::-1].index(QWEN3_THINK_END_TOKEN_ID)
    except ValueError:
        idx = 0
    thinking = tokenizer.decode(output_ids[:idx], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[idx:], skip_special_tokens=True).strip("\n")
    return thinking, content

def _batch_compare_answers(generated_texts: List[str], gt_texts: List[str]) -> List[bool]:
    evaluator = EvaluatorMathBatch(
        strict_extract=True,
        use_orig_eq_for_olympiadbench=True,
        timeout=60,
    )
    samples = []
    for gen_text, gt_text in zip(generated_texts, gt_texts):
        samples.append(
            SimpleNamespace(
                resp=gen_text,
                ref_ans=gt_text,
                ans=None,
                query="",
                dataset="math",
            )
        )
    _answers, corrects = evaluator.batch_eval(samples, n_procs=4)
    return [bool(x) for x in corrects]

def _online_init_llm(model_path: str):
    global _ONLINE_LLM, _ONLINE_TOKENIZER
    if _ONLINE_LLM is None or _ONLINE_TOKENIZER is None:
        print(f"[OnlineEval] Initializing LLM {model_path} for online evaluation...")
        _ONLINE_LLM = get_model(model_path, device="cuda")
        _ONLINE_TOKENIZER = get_tokenizer(model_path)
    return _ONLINE_LLM, _ONLINE_TOKENIZER

@torch.no_grad()
def _online_eval_math_single(
    *,
    model,
    tokenizer,
    model_path: str,
    transition: List[int],
    question: str,
    gt: str,
    max_new_tokens: int,
    temperature: float = 0.0,
) -> float:
    """
    Evaluate a candidate layer path by generating a boxed answer and checking it
    with the math evaluator.
    """
    input_text = (
        "Solve the following math problem and output ONLY the final answer directly, "
        "formatted strictly as \\boxed{ANSWER}.\n"
        "### Problem Start\n"
        f"{question}\n"
        "### Problem End\n"
        "Answer:"
    )

    # Apply depth-router custom path
    setup_custom_path(model, transition)

    # Generation
    if _is_qwen3_model_path(model_path):
        text = _qwen3_apply_chat_template(tokenizer, input_text)
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        if temperature > 0:
            generated_ids = model.generate(**model_inputs, max_new_tokens=int(max_new_tokens), do_sample=True, temperature=float(temperature))
        else:
            generated_ids = model.generate(**model_inputs, max_new_tokens=int(max_new_tokens), do_sample=False)
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        _thinking, content = _qwen3_split_thinking(output_ids, tokenizer)
        answer_part = content.strip()
    elif _is_qwen15_moe_chat_model_path(model_path):
        text = _qwen15_moe_apply_chat_template(tokenizer, input_text)
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        if temperature > 0:
            generated_ids = model.generate(**model_inputs, max_new_tokens=int(max_new_tokens), do_sample=True, temperature=float(temperature))
        else:
            generated_ids = model.generate(**model_inputs, max_new_tokens=int(max_new_tokens), do_sample=False)
        new_token_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        answer_part = tokenizer.batch_decode(new_token_ids, skip_special_tokens=True)[0].strip()
    elif _is_qwen25_instruct_model_path(model_path):
        text = _qwen25_apply_chat_template(tokenizer, input_text)
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        if temperature > 0:
            generated_ids = model.generate(**model_inputs, max_new_tokens=int(max_new_tokens), do_sample=True, temperature=float(temperature))
        else:
            generated_ids = model.generate(**model_inputs, max_new_tokens=int(max_new_tokens), do_sample=False)
        new_token_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        answer_part = tokenizer.batch_decode(new_token_ids, skip_special_tokens=True)[0].strip()
    else:
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        if temperature > 0:
            outputs = model.generate(**inputs, max_new_tokens=int(max_new_tokens), do_sample=True, temperature=float(temperature))
        else:
            outputs = model.generate(**inputs, max_new_tokens=int(max_new_tokens), do_sample=False)
        pred_answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        answer_part = pred_answer.split("Answer:")[-1].strip()

    # Only accept generations that follow the requested boxed-answer format.
    if "oxed{" not in answer_part:
        return 0.0
    return 1.0 if _batch_compare_answers([answer_part], [gt])[0] else 0.0

def _polar_path_lookup_type(
    pt: Tuple[int, ...],
    final_valid_transitions: set,
    final_invalid_transitions: set,
) -> str:
    """How a decoded path was scored during in-domain eval."""
    if pt in final_valid_transitions:
        return "valid_cache"
    if pt in final_invalid_transitions:
        return "invalid_cache"
    return "online"

def _print_polar_cache_breakdown(
    *,
    diff: int,
    sample_results: List[Dict[str, Any]],
    total_correct: int,
) -> None:
    """Summarize correct_found by valid-cache vs online LLM."""
    n = len(sample_results)
    via_valid = 0
    via_online = 0
    via_both = 0
    online_path_calls = 0
    valid_path_hits = 0

    for row in sample_results:
        if not row.get("has_correct"):
            continue
        v = bool(row.get("correct_via_valid_cache"))
        o = bool(row.get("correct_via_online"))
        if v:
            via_valid += 1
        if o:
            via_online += 1
        if v and o:
            via_both += 1
        online_path_calls += int(row.get("n_online_paths", 0) or 0)
        valid_path_hits += int(row.get("n_valid_cache_paths", 0) or 0)

    online_only = via_online - via_both
    valid_only = via_valid - via_both

    print(f"[Polar] diff{diff} correct_found breakdown ({total_correct}/{n} = {total_correct/max(1,n):.4f}):")
    print(f"  via valid_cache (MCTS final_valid): {via_valid} ({via_valid/max(1,n):.4f})")
    print(f"  via online LLM only:              {online_only} ({online_only/max(1,n):.4f})")
    print(f"  via both (same sample):           {via_both} ({via_both/max(1,n):.4f})")
    print(f"  valid_cache-only:                 {valid_only} ({valid_only/max(1,n):.4f})")
    top_k = sample_results[0].get("_top_k_paths", "?") if sample_results else "?"
    print(f"  path hits in valid_cache:         {valid_path_hits} (across top-{top_k} paths/sample)")
    print(f"  online path evaluations:          {online_path_calls}")

def evaluate_polar(args):
    base_path = resolve_dart_base_path(args.model_path, getattr(args, "data_root", None))

    # Determine diffs/files to evaluate
    if args.target_diff is not None:
        diffs = [args.target_diff]
        train_mode_suffix = f"_target_diff{args.target_diff}"
    else:
        if not args.eval_all_diffs:
            raise ValueError("Polar evaluation: set --target_diff OR use --eval_all_diffs to evaluate diff1-5.")
        diffs = list(range(1, 6))
        train_mode_suffix = "_all_diffs"

    lenpref_enabled = (args.policy_mode == "polar_lenpref")
    method_tag = "polar_lenpref" if lenpref_enabled else "polar"
    lr_sched_name = str(getattr(args, "lr_scheduler", "none") or "none").lower()
    warmup_steps = int(getattr(args, "warmup_steps", 0) or 0)
    warmup_steps = max(0, warmup_steps)

    if not args.checkpoint_path:
        args.checkpoint_path = checkpoint_path_for_args(args)
    print(f"[{method_tag}] Loading checkpoint: {args.checkpoint_path}")
    predictor = PolarPredictor(num_layers=args.original_depth, d_model=args.polar_d_model, nheads=args.polar_heads, n_layer_blocks=args.polar_layers).cuda()
    state = torch.load(args.checkpoint_path, map_location="cpu")
    predictor.load_state_dict(state)
    predictor.eval()

    for d in diffs:
        merged_file = merged_samples_path(base_path, d)
        with open(merged_file, "r") as f:
            data = json.load(f)
        samples = data["samples"] if isinstance(data, dict) and "samples" in data else (list(data.values()) if isinstance(data, dict) else data)
        start_idx, end_idx = 1500, 2000
        if args.num_samples:
            end_idx = min(start_idx + args.num_samples, len(samples))
        samples = samples[start_idx:end_idx]
        print(f"[Polar] Evaluating diff{d} on {len(samples)} samples ({start_idx}-{end_idx}) from {merged_file}")

        results = []
        total_correct = 0
        for i, sample in enumerate(samples):
            question, gt = extract_question_and_gt(sample)

            valid_paths_list = sample.get("final_valid_transitions", []) or []
            final_valid_transitions = set(tuple(p) for p in valid_paths_list if isinstance(p, (list, tuple)))
            final_invalid_transitions = set(tuple(p) for p in sample.get("final_invalid_transitions", []) if isinstance(p, (list, tuple)))

            with torch.no_grad():
                seg_logits, op_logits = predictor([question])
            seg_logits = seg_logits[0]
            op_logits = op_logits[0]

            beams = decode_polar_to_actions(
                seg_logits=seg_logits,
                op_logits=op_logits,
                threshold=args.seg_threshold,
                max_pack=4,
                beam_size=args.beam_size,
                top_k_ops=args.top_k_ops,
            )

            # Optional reranking that favors shorter predicted paths.
            # score' = sum(op_logprob) - len_penalty * path_len(actions)
            if args.len_penalty and args.len_penalty > 0:
                beams = sorted(
                    beams,
                    key=lambda t: (t[1] - float(args.len_penalty) * estimate_path_length_from_actions(t[0])),
                    reverse=True,
                )

            evaluated_paths = []
            has_correct = False
            correct_via_valid_cache = False
            correct_via_online = False
            n_online_paths = 0
            n_valid_cache_paths = 0
            for actions, _score in beams[: args.top_k_paths]:
                path = actions_to_path(actions, args.original_depth)

                if not path:
                    evaluated_paths.append({"path": path, "score": 0.0, "lookup_type": "empty"})
                    continue
                pt = tuple(path)
                lookup_type = _polar_path_lookup_type(pt, final_valid_transitions, final_invalid_transitions)
                trust_valid_cache = bool(getattr(args, "trust_valid_cache", True))
                max_new_tokens = int(getattr(args, "max_new_tokens", 50) or 50)

                if lookup_type == "valid_cache" and trust_valid_cache:
                    score = 1.0
                    has_correct = True
                    correct_via_valid_cache = True
                    n_valid_cache_paths += 1
                elif lookup_type == "invalid_cache" and trust_valid_cache:
                    score = 0.0
                else:
                    if lookup_type == "valid_cache":
                        n_valid_cache_paths += 1
                    else:
                        n_online_paths += 1
                    model, tokenizer = _online_init_llm(args.model_path)

                    score = _online_eval_math_single(
                        model=model,
                        tokenizer=tokenizer,
                        model_path=args.model_path,
                        transition=path,
                        question=question,
                        gt=gt,
                        max_new_tokens=max_new_tokens,
                        temperature=0.0,
                    )

                    if score == 1.0:
                        has_correct = True
                        if lookup_type == "valid_cache":
                            correct_via_valid_cache = True
                        else:
                            correct_via_online = True

                evaluated_paths.append({"path": path, "score": score, "lookup_type": lookup_type})
                if len(evaluated_paths) >= args.top_k_paths:
                    break

            if has_correct:
                total_correct += 1

            results.append({
                "sample_id": i,
                "question": question,
                "top_paths": evaluated_paths,
                "has_correct": has_correct,
                "correct_via_valid_cache": correct_via_valid_cache,
                "correct_via_online": correct_via_online,
                "n_online_paths": n_online_paths,
                "n_valid_cache_paths": n_valid_cache_paths,
                "initial_score": extract_initial_score(sample),
                "_top_k_paths": args.top_k_paths,
            })

        print(f"[Polar] diff{d} correct found rate: {total_correct}/{len(samples)} = {total_correct/max(1,len(samples)):.4f}")
        _print_polar_cache_breakdown(diff=d, sample_results=results, total_correct=total_correct)

        out_path = eval_results_path_for_args(args, diff=d, train_mode_suffix=train_mode_suffix)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[Polar] Saved results to {out_path}")

