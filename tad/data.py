"""THUMOS-14 annotations and I3D feature loading.

Time conventions
----------------
Features were extracted with 16-frame clips at a stride of 4 frames, so feature
index ``i`` is centred on frame ``4 * i + 8``. A time ``t`` (seconds) maps to the
feature grid as ``t * fps / 4 - 2``; the inverse is ``(i * 4 + 8) / fps``.
"""
import json
import os
import random
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset

FEAT_STRIDE = 4
CLIP_FRAMES = 16
FEAT_OFFSET = 0.5 * CLIP_FRAMES / FEAT_STRIDE  # = 2 feature steps
DEFAULT_FPS = 30.0

# Standard protocol (G-TAD / OpenTAD): this test video has wrong annotations.
EXCLUDED_VIDEOS = {"video_test_0000270"}


def load_annotations(ann_dirs, json_file=None):
    """Read THUMOS-14 temporal annotations.

    ``ann_dirs`` are folders with ``<Class>_val.txt`` / ``<Class>_test.txt`` files
    (lines: ``video_name start end`` in seconds). Returns
    (classes, {video: [(start, end, class_id), ...]}, {video: fps}).
    If ``json_file`` (ActionFormer's thumos14.json) is given, per-video fps is read from it.
    """
    rows = []
    for d in ann_dirs:
        for fname in sorted(os.listdir(d)):
            if not fname.endswith(".txt") or fname.startswith(("Ambiguous", "readme")):
                continue
            label = fname.rsplit("_", 1)[0]
            with open(os.path.join(d, fname)) as f:
                for line in f:
                    parts = line.split()
                    if len(parts) == 3:
                        rows.append((parts[0], float(parts[1]), float(parts[2]), label))

    classes = sorted({r[3] for r in rows})
    cls2id = {c: i for i, c in enumerate(classes)}
    ann = defaultdict(list)
    for vid, s, e, label in rows:
        if vid in EXCLUDED_VIDEOS or e - s < 1e-3:
            continue
        item = (s, e, cls2id[label])
        if item not in ann[vid]:  # drop exact duplicates
            ann[vid].append(item)

    fps = defaultdict(lambda: DEFAULT_FPS)
    if json_file and os.path.exists(json_file):
        db = json.load(open(json_file))["database"]
        for vid, v in db.items():
            if "fps" in v:
                fps[vid] = float(v["fps"])
    return classes, dict(ann), fps


def seconds_to_grid(t, fps):
    return t * fps / FEAT_STRIDE - FEAT_OFFSET


def grid_to_seconds(x, fps):
    return (x * FEAT_STRIDE + CLIP_FRAMES / 2) / fps


class FeatureStore:
    """Lazily loads per-video I3D features stored as ``<video>.npy`` ([T, 2048])."""

    def __init__(self, feat_dir):
        self.feat_dir = feat_dir
        self.videos = sorted(f[:-4] for f in os.listdir(feat_dir) if f.endswith(".npy"))

    def __getitem__(self, vid):
        return np.load(os.path.join(self.feat_dir, vid + ".npy")).astype(np.float32)

    def split(self, name):
        return [v for v in self.videos if f"_{name}_" in v]


class ThumosTrainSet(Dataset):
    """Training videos with random temporal crops (as in ActionFormer)."""

    def __init__(self, store, videos, ann, fps, max_len=2304, crop_ratio=(0.9, 1.0), trunc_thresh=0.5):
        self.store, self.ann, self.fps = store, ann, fps
        self.videos = [v for v in videos if v in ann]
        self.max_len, self.crop_ratio, self.trunc_thresh = max_len, crop_ratio, trunc_thresh

    def __len__(self):
        return len(self.videos)

    def _segments(self, vid):
        f = self.fps[vid]
        segs = np.array([[seconds_to_grid(s, f), seconds_to_grid(e, f)] for s, e, _ in self.ann[vid]], np.float32)
        labels = np.array([c for _, _, c in self.ann[vid]], np.int64)
        return segs, labels

    def __getitem__(self, idx):
        vid = self.videos[idx]
        feats = self.store[vid]
        segs, labels = self._segments(vid)
        T = feats.shape[0]
        if T > self.max_len:
            crop_len = self.max_len
        else:
            crop_len = random.randint(max(round(self.crop_ratio[0] * T), 1), round(self.crop_ratio[1] * T))
        if crop_len < T:
            for _ in range(10):  # retry until the crop keeps at least one action
                st = random.randint(0, T - crop_len)
                ed = st + crop_len
                cs = np.clip(segs, st, ed) - st
                inter = cs[:, 1] - cs[:, 0]
                valid = inter / np.maximum(segs[:, 1] - segs[:, 0], 1e-6) >= self.trunc_thresh
                if valid.any():
                    break
            feats, segs, labels = feats[st:ed], cs[valid], labels[valid]
        return {
            "video_id": vid,
            "feats": torch.from_numpy(feats.T.copy()),  # [C, T]
            "segments": torch.from_numpy(segs),
            "labels": torch.from_numpy(labels),
        }


def collate(batch, pad_to=None, stride=32):
    """Pads a list of [C, T] features to one length; returns feats [B,C,T], mask [B,1,T]."""
    lens = [b["feats"].shape[1] for b in batch]
    L = pad_to if pad_to is not None else max(lens)
    L = int(np.ceil(max(L, max(lens)) / stride) * stride)
    C = batch[0]["feats"].shape[0]
    feats = torch.zeros(len(batch), C, L)
    mask = torch.zeros(len(batch), 1, L, dtype=torch.bool)
    for i, b in enumerate(batch):
        feats[i, :, : lens[i]] = b["feats"]
        mask[i, :, : lens[i]] = True
    return feats, mask, batch
