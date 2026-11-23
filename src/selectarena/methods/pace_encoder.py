"""Frozen PACE encoder, matching the project's cumulative TabICL-v1.1 layer 2.

Only training feature values enter the encoder. No current-task labels are used.
Rank preprocessing and reserved-token layout match TorchAcceleratedEmbedder.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.stats import norm, rankdata


class PaceEncoder:
    def __init__(self, checkpoint, device):
        from tabicl import TabICLClassifier
        model = TabICLClassifier(model_path=str(checkpoint), n_estimators=1,
                                 random_state=42, device=str(device))
        model.fit([[0, 1], [1, 0]], [0, 1])
        self.col = model.model_.col_embedder.eval()
        self.device = device
        self.d = self.col.embed_dim
        self.n_cls = getattr(self.col, 'reserve_cls_tokens', 4)
        assert self.d == 128 and len(self.col.tf_col.blocks) >= 2
        for parameter in self.col.parameters():
            parameter.requires_grad_(False)

    @torch.inference_mode()
    def encode(self, x):
        n = len(x)
        q = norm.ppf((np.arange(1, n + 1) - 0.5) / n)
        r = rankdata(x, method='average').astype(int) - 1
        values = q[np.clip(r, 0, n - 1)].astype(np.float32)
        t = torch.zeros((1, 1 + self.n_cls, n, 1), device=self.device)
        t[0, self.n_cls, :, 0] = torch.as_tensor(values, device=self.device)
        t = self.col.in_linear(t)
        for block in self.col.tf_col.blocks[:2]:
            t = block(t)
        return t[0, self.n_cls, :, :].detach()
