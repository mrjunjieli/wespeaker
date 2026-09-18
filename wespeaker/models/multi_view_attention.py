# Copyright (c) 2026 Junjie LI (mrjunjieli@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''Multi-view self-attention for uncertainty estimation.

Reference:
    U^3-xi: Pushing the Boundaries of Speaker Recognition by
    Incorporating Uncertainty. https://arxiv.org/abs/2601.15719
'''

import torch
from torch import nn


class MultiViewSelfAttention(nn.Module):
    """Multi-View Self-Attention.

    Each attention head attends over a different temporal band, so that
    the heads jointly capture short- and long-range structures. Head ``i``
    covers a symmetric window of ``w_i = 2 ** (i + 1) + 1`` frames, i.e.
    a half window of ``2 ** i`` on each side.

    Args:
        embed_dim: model embedding dim, must be divisible by num_heads.
        num_heads: number of heads.
        dropout: attention / output dropout.
    """

    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, \
            "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # projections for Q, K, V and output
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

    def _head_windows(self, device):
        """Half-window size (radius) per head: 2 ** i for head i."""
        half_windows = [2 ** i for i in range(self.num_heads)]
        return torch.tensor(half_windows, device=device)  # (H,)

    def forward(self, x, mask=None):
        """
        Args:
            x: (B, T, E)
            mask: optional bool mask (B, T), True = valid frame.

        Returns:
            (B, T, E)
        """
        B, T, E = x.shape
        device = x.device

        qkv = self.qkv_proj(x)  # (B, T, 3E)
        qkv = qkv.reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        # (B, T, H, D) -> (B, H, T, D)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        logits = torch.matmul(q * self.scale,
                              k.transpose(-2, -1))  # (B, H, T, T)

        # head-wise band mask: (H, T, T) -> (B, H, T, T)
        half_windows = self._head_windows(device)  # (H,)
        idx = torch.arange(T, device=device)
        dist = (idx[None, :] - idx[:, None]).abs()  # (T, T)
        allowed_heads = (dist[None, :, :] <=
                         half_windows[:, None, None])  # (H, T, T)
        allowed_heads = allowed_heads.unsqueeze(0).expand(B, -1, -1, -1)

        neg_inf = -1e9
        logits = torch.where(
            allowed_heads, logits,
            torch.tensor(neg_inf, device=device, dtype=logits.dtype))

        # block attention to padded positions
        if mask is not None:
            key_mask = (~mask).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
            logits = logits.masked_fill(key_mask, neg_inf)

        attn = torch.softmax(logits, dim=-1)  # (B, H, T, T)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)  # (B, H, T, D)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, T, E)
        out = self.out_proj(out)
        out = self.out_dropout(out)
        return out


class MultiViewTransformerEncoderLayer(nn.Module):
    """Pre-norm transformer layer built on :class:`MultiViewSelfAttention`."""

    def __init__(self, embed_dim, num_heads, ff_hidden=2048, dropout=0.2):
        super().__init__()
        self.embed_dim = embed_dim
        self.self_attn = MultiViewSelfAttention(embed_dim,
                                                num_heads,
                                                dropout=dropout)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, mask=None):
        """
        Args:
            x: (B, T, E)
            mask: optional bool mask (B, T), True = valid frame.

        Returns:
            (B, T, E)
        """
        # self attention + residual
        x = self.norm1(x + self.self_attn(x, mask=mask))
        # feed-forward + residual
        x = self.norm2(x + self.ff(x))
        return x
