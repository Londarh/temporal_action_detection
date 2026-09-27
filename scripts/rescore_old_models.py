"""Re-score the original class-project models (May 2026) under the standard protocol.

The original notebook had three evaluation problems: 11-point AP at a single IoU,
a confidence threshold picked on the test set, and a shape bug that gave the
Transformer an mAP of exactly 0. This script reloads the four saved checkpoints,
runs them the way they were trained (per frame for the MLP, 15-step windows for the
others), keeps the original threshold-and-merge proposal step with a fixed
threshold of 0.3, and scores everything with the same evaluator as the new detector.

Usage:
    python scripts/rescore_old_models.py --ckpt_dir /content/drive/MyDrive/thumos14_checkpoints
"""
import argparse
import os
import pickle
import sys

import numpy as np
import torch
import torch.nn as nn
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tad.data import FeatureStore, load_annotations  # noqa: E402
from tad.engine import save_json  # noqa: E402
from tad.evaluate import evaluate  # noqa: E402

NUM_OLD_CLASSES = 21  # background + 20 actions
STRIDE_SEC = 4 / 30   # the original notebook's timestep length
WINDOW = 15


# ---- original model definitions (unchanged, so the saved weights load) ----
class BasicMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.3):
        super().__init__()
        self.process = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, num_classes))

    def forward(self, x):
        return self.process(x)


class BaseCNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, window=15, kernel_size=3, dropout=0.3):
        super().__init__()
        self.process = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, 3, padding=1), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.out = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(hidden_dim, num_classes))

    def forward(self, x):
        return self.out(self.process(x.permute(0, 2, 1)))


class BaseLSTM(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=256, num_layers=2, num_classes=21, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=False, dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.classifier(self.dropout(hn[-1]))


class BaseTransformer(nn.Module):
    def __init__(self, input_dim=2048, d_model=256, nhead=4, num_layers=2, num_classes=21, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
                                           dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.classifier(self.dropout(self.transformer(self.input_proj(x))))
        return x[:, x.shape[1] // 2, :]  # centre frame of the window


OLD_MODELS = {
    "MLP": (lambda: BasicMLP(2048, 512, NUM_OLD_CLASSES, 0.25), "mlp_best.pt", False),
    "1D-CNN": (lambda: BaseCNN(2048, 256, NUM_OLD_CLASSES, 15, dropout=0.25), "cnn_best.pt", True),
    "LSTM": (lambda: BaseLSTM(2048, 256, 2, NUM_OLD_CLASSES, 0.3), "lstm_best.pt", True),
    "Transformer": (lambda: BaseTransformer(2048, 256, 4, 2, NUM_OLD_CLASSES, 0.1), "transformer_best.pt", True),
}


@torch.no_grad()
def frame_probs(model, feats, windowed, device, chunk=512):
    """Per-timestep class probabilities [T, 21]. The Transformer bug came from feeding it the
    whole video; here every model gets exactly the input shape it was trained on."""
    T = feats.shape[0]
    x = torch.from_numpy(feats)
    if not windowed:
        return torch.softmax(model(x.to(device)), -1).cpu().numpy()
    if T < WINDOW:
        x = torch.cat([x, x.new_zeros(WINDOW - T, x.shape[1])])
    start = np.clip(np.arange(T) - WINDOW // 2, 0, max(len(x) - WINDOW, 0))
    idx = torch.from_numpy(start[:, None] + np.arange(WINDOW))
    out = []
    for i in range(0, T, chunk):
        out.append(torch.softmax(model(x[idx[i:i + chunk]].to(device)), -1).cpu())
    return torch.cat(out).numpy()


def threshold_and_merge(probs, video, th=0.3):
    """The original proposal step: threshold each class, merge consecutive steps."""
    props, min_len = [], int(0.5 / STRIDE_SEC)
    for c in range(1, probs.shape[1]):
        m = np.concatenate([[False], probs[:, c] >= th, [False]])
        d = np.diff(m.astype(int))
        for s, e in zip(np.where(d == 1)[0], np.where(d == -1)[0] - 1):
            if e - s + 1 < min_len or e <= s:
                continue
            props.append({"video": video, "old_label": c, "start": s * STRIDE_SEC,
                          "end": (e + 1) * STRIDE_SEC, "score": float(probs[s:e + 1, c].mean())})
    return props


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/thumos_i3d.yaml")
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--out", default="results/baselines.json")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    classes, ann, _ = load_annotations(cfg["data"]["ann_dirs"], cfg["data"].get("json_file"))
    store = FeatureStore(cfg["data"]["feat_dir"])
    test_videos = [v for v in store.split("test") if v in ann]
    meta = pickle.load(open(os.path.join(args.ckpt_dir, "metadata.pkl"), "rb"))
    old_to_new = {i: classes.index(name) for i, name in meta["idx2label"].items() if name != "background"}
    device = "cuda" if torch.cuda.is_available() else "cpu"

    results = {}
    for name, (build, fname, windowed) in OLD_MODELS.items():
        path = os.path.join(args.ckpt_dir, fname)
        if not os.path.exists(path):
            print(f"skip {name}: {fname} not found")
            continue
        model = build()
        try:
            model.load_state_dict(torch.load(path, map_location="cpu"))
        except Exception as err:  # e.g. a checkpoint that was not fully written to Drive
            print(f"skip {name}: could not load {fname} ({type(err).__name__})")
            continue
        model.to(device).eval()
        preds = []
        for vid in test_videos:
            for p in threshold_and_merge(frame_probs(model, store[vid], windowed, device), vid, args.threshold):
                p["label"] = old_to_new[p.pop("old_label")]
                preds.append(p)
        res = evaluate(ann, preds, classes, test_videos)
        results[name] = res
        print(f"{name:12s} mAP@0.3 {res['mAP']['0.3']:5.2f}  @0.5 {res['mAP']['0.5']:5.2f}  "
              f"@0.7 {res['mAP']['0.7']:5.2f}  avg {res['avg_mAP']:5.2f}")
        save_json(results, args.out)  # save after every model


if __name__ == "__main__":
    main()
