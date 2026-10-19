import math

import torch
from torch import nn
import torch.nn.functional as F


class TimestepEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps):
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=timesteps.device).float() / max(half - 1, 1)
        )
        args = timesteps[:, None].float() * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class CandidateDiffusionRefiner(nn.Module):
    """Few-step label-space refinement over CAMS candidate pair logits."""

    def __init__(self, feature_dim, hidden_dim=512, time_dim=128, dropout=0.1):
        super().__init__()
        self.time_embedding = TimestepEmbedding(time_dim)
        self.image_proj = nn.Linear(feature_dim, hidden_dim)
        self.pair_proj = nn.Linear(feature_dim, hidden_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.scalar_proj = nn.Linear(4, hidden_dim)
        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        base_logits,
        image_features,
        pair_features,
        attr_pair_scores=None,
        obj_pair_scores=None,
        steps=4,
        topk=64,
        step_size=1.0,
        force_indices=None,
        zero_mean_delta=False,
        use_gate=False,
    ):
        if steps <= 0:
            return base_logits

        batch_size, num_pairs = base_logits.shape
        topk = num_pairs if topk is None or topk <= 0 else min(topk, num_pairs)

        image_context = self.image_proj(image_features).unsqueeze(1)
        pair_context = self.pair_proj(pair_features).unsqueeze(0)
        refined = base_logits

        if attr_pair_scores is None:
            attr_pair_scores = torch.zeros_like(base_logits)
        if obj_pair_scores is None:
            obj_pair_scores = torch.zeros_like(base_logits)

        base_centered = base_logits - base_logits.mean(dim=1, keepdim=True)
        base_scaled = base_centered / (base_centered.std(dim=1, keepdim=True) + 1e-6)

        candidate_mask = self._candidate_mask(base_logits, topk, force_indices)
        for step in range(steps):
            t_value = 1.0 - float(step) / max(steps, 1)
            timesteps = base_logits.new_full((batch_size,), t_value)
            time_context = self.time_proj(self.time_embedding(timesteps)).unsqueeze(1)
            current_probs = torch.softmax(refined, dim=-1)
            scalar_features = torch.stack(
                [base_scaled, current_probs, attr_pair_scores, obj_pair_scores],
                dim=-1,
            )
            hidden = image_context + pair_context + time_context + self.scalar_proj(scalar_features)
            delta = self.mlp(hidden).squeeze(-1)
            if zero_mean_delta:
                denom = candidate_mask.sum(dim=1, keepdim=True).clamp_min(1).to(delta.dtype)
                delta = delta - (delta * candidate_mask.to(delta.dtype)).sum(dim=1, keepdim=True) / denom
            if use_gate:
                gate = torch.sigmoid(self.gate(hidden).squeeze(-1))
                delta = gate * delta
            delta = delta * candidate_mask.to(delta.dtype)
            refined = refined + step_size * self.residual_scale * delta

        return refined

    @staticmethod
    def _candidate_mask(base_logits, topk, force_indices=None):
        batch_size, num_pairs = base_logits.shape
        _, top_indices = base_logits.topk(topk, dim=1)
        mask = torch.zeros(batch_size, num_pairs, device=base_logits.device, dtype=torch.bool)
        mask.scatter_(1, top_indices, True)
        if force_indices is not None:
            mask.scatter_(1, force_indices.view(-1, 1), True)
        return mask
