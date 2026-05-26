import importlib
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import logging

logging.set_verbosity_error()

SUPPORTED_MODEL_HINTS = (
    "llama-3.2-3b-instruct",
    "qwen1.5-moe-a2.7b-chat",
    "qwen2.5-3b-instruct",
    "qwen3-8b",
)

SUPPORTED_FOR_CAUSAL_LM = {
    "LlamaForCausalLM",
    "Qwen2ForCausalLM",
    "Qwen2MoeForCausalLM",
    "Qwen3ForCausalLM",
}


def _import_first_attr(module_path: str, candidates: List[str]):
    """Return the first available attribute from `candidates` in `module_path`."""
    mod = importlib.import_module(module_path)
    for name in candidates:
        if hasattr(mod, name):
            return getattr(mod, name)
    raise AttributeError(f"None of the candidate attributes exist in {module_path}: {candidates}")


def _supported_model_key(model_name: str) -> str:
    name = str(model_name or "").lower()
    if "llama-3.2-3b-instruct" in name:
        return "llama"
    if ("qwen1.5" in name or "qwen1_5" in name) and "moe-a2.7b" in name and "chat" in name:
        return "qwen1.5-moe"
    if "qwen2.5-3b-instruct" in name or "qwen2_5-3b-instruct" in name:
        return "qwen2.5"
    if "qwen3-8b" in name:
        return "qwen3"
    supported = ", ".join(SUPPORTED_MODEL_HINTS)
    raise ValueError(f"Unsupported release model `{model_name}`. Supported model identifiers must include one of: {supported}")


def apply_ulysses_patch(model_name: str) -> None:
    model_key = _supported_model_key(model_name)

    if model_key == "qwen1.5-moe":
        from llm_depth_router.patches.qwen2_moe import _qwen2_moe_forward_patch

        Qwen2MoeModel = _import_first_attr(
            "transformers.models.qwen2_moe.modeling_qwen2_moe",
            ["Qwen2MoeModel", "Qwen2MoEModel"],
        )
        Qwen2MoeModel.forward = _qwen2_moe_forward_patch
    elif model_key == "qwen3":
        from llm_depth_router.patches.qwen3 import _qwen3_forward_patch

        Qwen3Model = _import_first_attr(
            "transformers.models.qwen3.modeling_qwen3",
            ["Qwen3Model"],
        )
        Qwen3Model.forward = _qwen3_forward_patch
    elif model_key == "qwen2.5":
        from llm_depth_router.patches.qwen2 import _qwen2_forward_patch

        Qwen2Model = _import_first_attr(
            "transformers.models.qwen2.modeling_qwen2",
            ["Qwen2Model"],
        )
        Qwen2Model.forward = _qwen2_forward_patch
    elif model_key == "llama":
        from llm_depth_router.patches.llama import _llama_forward_patch

        LlamaModel = _import_first_attr(
            "transformers.models.llama.modeling_llama",
            ["LlamaModel"],
        )
        LlamaModel.forward = _llama_forward_patch


def get_model(model_name: str, device: str):
    apply_ulysses_patch(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype="auto",
    )
    model.to(device)
    for p in model.parameters():
        if not p.is_cuda and "cuda" in device:
            p.data = p.data.to(device)
    return model


def get_tokenizer(model_name: str, custom_chat_template=None):
    _supported_model_key(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if custom_chat_template is not None:
        tokenizer.chat_template = custom_chat_template
    return tokenizer


def setup_custom_path(model, path):
    cls_name = type(model).__name__
    if cls_name not in SUPPORTED_FOR_CAUSAL_LM:
        raise ValueError(f"Unsupported release model wrapper `{cls_name}` for custom path execution.")
    if not hasattr(model, "model"):
        raise ValueError(f"Model wrapper `{cls_name}` does not expose `.model` for custom path execution.")
    model.model.custom_path = list(path)
