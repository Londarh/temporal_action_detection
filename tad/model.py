"""Anchor-free multi-scale temporal action detector.

Pipeline: masked conv embedding -> local-attention transformer stem -> 6-level
temporal pyramid (stride 1..32) -> shared classification / boundary heads.
Every pyramid step predicts class probabilities and its distances to the action's
start and end, so segments are regressed directly instead of being stitched
together from per-frame labels.

Design follows ActionFormer (Zhang et al., ECCV 2022); implemented from scratch.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn


class ChannelLayerNorm(nn.Module):
    """LayerNorm over channels for [B, C, T] tensors."""

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class MaskedConv1d(nn.Module):
    def __init__(self, cin, cout, k=3, bias=True):
        super().__init__()
        self.conv = nn.Conv1d(cin, cout, k, padding=k // 2, bias=bias)

    def forward(self, x, mask):
        return self.conv(x) * mask.to(x.dtype)


class LocalAttentionBlock(nn.Module):
    """Pre-norm transformer block with self-attention restricted to a local window."""

    def __init__(self, dim, heads=4, window=19, mlp_ratio=4, drop_path=0.1):
        super().__init__()
        self.heads, self.window, self.drop_path = heads, window, drop_path
        self.norm1 = ChannelLayerNorm(dim)
        self.qkv = nn.Conv1d(dim, 3 * dim, 1)
        self.proj = nn.Conv1d(dim, dim, 1)
        self.norm2 = ChannelLayerNorm(dim)
        self.mlp = nn.Sequential(nn.Conv1d(dim, mlp_ratio * dim, 1), nn.GELU(), nn.Conv1d(mlp_ratio * dim, dim, 1))

    def _drop(self, x):
        if not self.training or self.drop_path == 0:
            return x
        keep = torch.rand(x.shape[0], 1, 1, device=x.device) >= self.drop_path
        return x * keep / (1 - self.drop_path)

    def _attn_mask(self, mask):
        T = mask.shape[-1]
        idx = torch.arange(T, device=mask.device)
        band = (idx[None, :] - idx[:, None]).abs() <= self.window // 2  # [T, T]
        allowed = band[None] & mask  # keys outside the video are masked: [B, T, T]
        allowed = allowed | torch.eye(T, dtype=torch.bool, device=mask.device)[None]  # never an empty row
        return allowed[:, None]  # [B, 1, T, T]

    def forward(self, x, mask):
        B, C, T = x.shape
        h = self.norm1(x)
        q, k, v = self.qkv(h).reshape(B, 3, self.heads, C // self.heads, T).transpose(-1, -2).unbind(1)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=self._attn_mask(mask))
        out = out.transpose(-1, -2).reshape(B, C, T)
        m = mask.to(x.dtype)
        x = x + self._drop(self.proj(out) * m)
        x = x + self._drop(self.mlp(self.norm2(x)) * m)
        return x


class Scale(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x * self.scale


class Head(nn.Module):
    def __init__(self, dim, out_dim, num_layers=3, prior=None):
        super().__init__()
        self.convs = nn.ModuleList(MaskedConv1d(dim, dim, bias=False) for _ in range(num_layers - 1))
        self.norms = nn.ModuleList(ChannelLayerNorm(dim) for _ in range(num_layers - 1))
        self.out = MaskedConv1d(dim, out_dim)
        if prior is not None:  # start with low foreground probability (focal-loss init)
            nn.init.constant_(self.out.conv.bias, -math.log((1 - prior) / prior))

    def forward(self, x, mask):
        for conv, norm in zip(self.convs, self.norms):
            x = F.relu(norm(conv(x, mask)))
        return self.out(x, mask)


class TemporalDetector(nn.Module):
    def __init__(self, in_dim=2048, dim=512, num_classes=20, levels=6, stem_blocks=2,
                 window=19, heads=4, drop_path=0.1, neck="transformer",
                 regression_range=((0, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 10000))):
        super().__init__()
        assert len(regression_range) == levels
        self.levels, self.num_classes, self.neck_type = levels, num_classes, neck
        self.strides = [2 ** i for i in range(levels)]
        self.regression_range = regression_range

        self.embed = nn.ModuleList([MaskedConv1d(in_dim, dim), MaskedConv1d(dim, dim)])
        self.embed_norm = nn.ModuleList([ChannelLayerNorm(dim), ChannelLayerNorm(dim)])
        make = (lambda: LocalAttentionBlock(dim, heads, window, drop_path=drop_path)) if neck == "transformer" \
            else (lambda: ConvBlock(dim))
        self.stem = nn.ModuleList(make() for _ in range(stem_blocks))
        self.branch = nn.ModuleList(make() for _ in range(levels - 1))
        self.fpn_norm = nn.ModuleList(ChannelLayerNorm(dim) for _ in range(levels))
        self.cls_head = Head(dim, num_classes, prior=0.01)
        self.reg_head = Head(dim, 2)
        self.scales = nn.ModuleList(Scale() for _ in range(levels))

    def forward(self, x, mask):
        """x: [B, C, T] features, mask: [B, 1, T] bool. T must be divisible by 2**(levels-1).

        Returns per-level lists of cls logits [B, T_l, K], offsets [B, T_l, 2] (in units of
        the level's stride) and masks [B, T_l].
        """
        for conv, norm in zip(self.embed, self.embed_norm):
            x = F.relu(norm(conv(x, mask)))
        for blk in self.stem:
            x = blk(x, mask)
        feats, masks = [x], [mask]
        for blk in self.branch:
            x = F.max_pool1d(x, 3, stride=2, padding=1)
            mask = mask[:, :, ::2]
            x = blk(x, mask)
            feats.append(x)
            masks.append(mask)

        cls_out, reg_out, mask_out = [], [], []
        for l, (f, m) in enumerate(zip(feats, masks)):
            f = self.fpn_norm[l](f) * m.to(f.dtype)
            cls_out.append(self.cls_head(f, m).transpose(1, 2))
            reg_out.append(F.relu(self.scales[l](self.reg_head(f, m))).transpose(1, 2))
            mask_out.append(m[:, 0])
        return cls_out, reg_out, mask_out

    def points(self, lengths, device):
        """Pyramid points per level: [T_l, 4] = (position, range_lo, range_hi, stride)."""
        pts = []
        for l, (T, s) in enumerate(zip(lengths, self.strides)):
            p = torch.arange(T, device=device, dtype=torch.float32) * s
            lo, hi = self.regression_range[l]
            pts.append(torch.stack([p, torch.full_like(p, lo), torch.full_like(p, hi), torch.full_like(p, s)], 1))
        return pts


class ConvBlock(nn.Module):
    """Residual dilation-free conv block; used for the 'conv' neck ablation."""

    def __init__(self, dim):
        super().__init__()
        self.norm = ChannelLayerNorm(dim)
        self.conv1 = MaskedConv1d(dim, dim)
        self.conv2 = MaskedConv1d(dim, dim)

    def forward(self, x, mask):
        h = F.relu(self.conv1(self.norm(x), mask))
        return x + self.conv2(h, mask)
