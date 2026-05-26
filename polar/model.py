from typing import List, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from .config import OP_EXECUTE, OP_REPEAT, OP_SKIP


class PolarPredictor(nn.Module):
    """
    Question (token) embeddings -> cross-attend into per-layer module embeddings -> self-attend over layers
    Heads:
      - segmentation start logits per layer (B, D)
      - op logits per layer (B, D, 3) (loss only on segment starts)
    """
    def __init__(
        self,
        num_layers: int,
        embedding_model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        d_model: int = 256,
        nheads: int = 4,
        n_layer_blocks: int = 2,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model

        self.embedding_model = AutoModel.from_pretrained(embedding_model_name)
        self.embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name, padding_side="left")
        for p in self.embedding_model.parameters():
            p.requires_grad = False

        q_dim = self.embedding_model.config.hidden_size
        self.q_proj = nn.Linear(q_dim, d_model)

        self.layer_embedding = nn.Embedding(num_layers, d_model)

        # Cross-attention: module(layer) queries question tokens
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nheads, batch_first=True, dropout=0.1)

        # Global context across layers
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nheads, dim_feedforward=d_model * 4, dropout=0.1, batch_first=True)
        self.layer_encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layer_blocks)

        self.seg_head = nn.Linear(d_model, 1)
        self.op_head = nn.Linear(d_model, 3)

    def encode_question_tokens(self, questions: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          token_feats: (B, T, d_model)
          key_padding_mask: (B, T) True where padding
        """
        inputs = self.embedding_tokenizer(questions, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.embedding_model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.embedding_model(**inputs)
        token_h = out.last_hidden_state  # (B,T,q_dim)
        token_feats = self.q_proj(token_h)  # (B,T,d_model)
        key_padding_mask = inputs["attention_mask"] == 0
        return token_feats, key_padding_mask

    def forward(self, questions: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        device = self.embedding_model.device
        token_feats, key_padding_mask = self.encode_question_tokens(questions)

        # Build layer queries
        layer_ids = torch.arange(self.num_layers, device=device).unsqueeze(0).repeat(token_feats.size(0), 1)  # (B,D)
        layer_q = self.layer_embedding(layer_ids)  # (B,D,d_model)

        # Cross-attend: Q=layers, K/V=question tokens
        x, _ = self.cross_attn(query=layer_q, key=token_feats, value=token_feats, key_padding_mask=key_padding_mask)

        # Self-attend across layers for global decisions
        x = self.layer_encoder(x)  # (B,D,d_model)

        seg_logits = self.seg_head(x).squeeze(-1)  # (B,D)
        op_logits = self.op_head(x)  # (B,D,3)
        return seg_logits, op_logits

def decode_polar_to_actions(
    seg_logits: torch.Tensor,
    op_logits: torch.Tensor,
    threshold: float = 0.5,
    max_pack: int = 4,
    beam_size: int = 5,
    top_k_ops: int = 2,
) -> List[Tuple[List[Tuple[str, int, int]], float]]:
    """
    Decode a single example to candidate action sequences using a simple beam search.
    Returns list of (actions, score_logprob).
    """
    # seg_flip probabilities: seg_flip[i]=1 means boundary at i (i>0)
    seg_probs = torch.sigmoid(seg_logits).detach().cpu().tolist()  # length D
    op_logp = torch.log_softmax(op_logits, dim=-1).detach().cpu().tolist()  # (D,3)
    D = len(seg_probs)

    # 1) derive segment starts from flip mask
    starts = [0]
    for i in range(1, D):
        if seg_probs[i] >= threshold:
            starts.append(i)
    starts = sorted(set(starts))

    # enforce max_pack by inserting boundaries if needed
    fixed_starts = [0]
    for s in starts[1:]:
        while s - fixed_starts[-1] > max_pack:
            fixed_starts.append(fixed_starts[-1] + max_pack)
        if s != fixed_starts[-1]:
            fixed_starts.append(s)
    starts = fixed_starts

    # finalize segments
    segments: List[Tuple[int, int]] = []
    for idx, st in enumerate(starts):
        ed = starts[idx + 1] if idx + 1 < len(starts) else D
        # if empty, skip
        if ed <= st:
            continue
        # enforce max_pack again
        cur = st
        while cur < ed:
            nxt = min(cur + max_pack, ed)
            segments.append((cur, nxt))
            cur = nxt

    # 2) beam over ops per segment start
    beams: List[Tuple[List[Tuple[str, int, int]], float]] = [([], 0.0)]
    op_map = {OP_SKIP: "skip", OP_EXECUTE: "keep", OP_REPEAT: "repeat"}
    for (st, ed) in segments:
        size = ed - st
        # choose ops by logp at start layer
        op_lp = op_logp[st]  # [3]
        best_ops = sorted(list(range(3)), key=lambda o: op_lp[o], reverse=True)[:top_k_ops]
        new_beams: List[Tuple[List[Tuple[str, int, int]], float]] = []
        for actions, score in beams:
            for o in best_ops:
                # repeat count fixed to 1 for OP_REPEAT else 0
                cnt = 1 if o == OP_REPEAT else 0
                new_beams.append((actions + [(op_map[o], size, cnt)], score + float(op_lp[o])))
        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:beam_size]

    return beams

