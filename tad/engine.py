"""Training and inference loops."""
import copy
import json
import math
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import collate, grid_to_seconds
from .evaluate import evaluate
from .losses import DetectionLoss
from .postprocess import decode_video


class EMA:
    """Exponential moving average of model weights (evaluated instead of the raw model)."""

    def __init__(self, model, decay=0.999):
        self.model = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for e, m in zip(self.model.state_dict().values(), model.state_dict().values()):
            if e.dtype.is_floating_point:
                e.mul_(self.decay).add_(m.detach(), alpha=1 - self.decay)
            else:
                e.copy_(m)


def make_optimizer(model, lr, weight_decay):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if p.ndim <= 1 or "norm" in name or "scale" in name else decay).append(p)
    return torch.optim.AdamW([{"params": decay, "weight_decay": weight_decay},
                              {"params": no_decay, "weight_decay": 0.0}], lr=lr)


def lr_at(step, total, warmup, base):
    if step < warmup:
        return base * (step + 1) / warmup
    return 0.5 * base * (1 + math.cos(math.pi * (step - warmup) / max(total - warmup, 1)))


def train(model, train_set, cfg, device, log=print):
    loader = DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=True, drop_last=True,
                        num_workers=cfg.get("num_workers", 2),
                        collate_fn=lambda b: collate(b, pad_to=cfg["max_len"]))
    opt = make_optimizer(model, cfg["lr"], cfg["weight_decay"])
    ema = EMA(model, cfg.get("ema_decay", 0.999))
    criterion = DetectionLoss(model.num_classes, init_norm=cfg.get("init_loss_norm", 100.0))
    epochs = cfg["epochs"] + cfg["warmup_epochs"]
    total, warmup = epochs * len(loader), cfg["warmup_epochs"] * len(loader)
    history, step = [], 0
    model.to(device).train()
    ema.model.to(device)
    for epoch in range(epochs):
        t0, losses = time.time(), []
        for feats, mask, batch in loader:
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total, warmup, cfg["lr"])
            cls_out, reg_out, masks = model(feats.to(device), mask.to(device))
            loss, parts = criterion(model, cls_out, reg_out, masks, batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("clip_grad", 1.0))
            opt.step()
            ema.update(model)
            losses.append([loss.item(), parts["cls"], parts["reg"]])
            step += 1
        l = np.mean(losses, 0)
        history.append({"epoch": epoch + 1, "loss": float(l[0]), "cls_loss": float(l[1]), "reg_loss": float(l[2])})
        log(f"epoch {epoch + 1:3d}/{epochs}  loss {l[0]:.3f} (cls {l[1]:.3f}, reg {l[2]:.3f})  "
            f"lr {opt.param_groups[0]['lr']:.2e}  {time.time() - t0:.0f}s")
    return ema.model, history


@torch.no_grad()
def predict(model, store, videos, fps, device, classes):
    """Runs the detector on whole videos; returns a list of predictions in seconds."""
    model.eval().to(device)
    preds = []
    for vid in videos:
        feats = torch.from_numpy(store[vid].T.copy())
        T = feats.shape[1]
        x, mask, _ = collate([{"feats": feats}], stride=2 ** (model.levels - 1))
        cls_out, reg_out, masks = model(x.to(device), mask.to(device))
        segs, scores, labels = decode_video(model, cls_out, reg_out, masks, 0)
        duration = grid_to_seconds(T - 1, fps[vid]) + 8 / fps[vid]
        segs = np.clip(grid_to_seconds(segs, fps[vid]), 0, duration)
        for (s, e), sc, c in zip(segs, scores, labels):
            preds.append({"video": vid, "start": float(s), "end": float(e), "label": int(c),
                          "class": classes[int(c)], "score": float(sc)})
    return preds


def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)


def evaluate_predictions(ann, preds, classes, videos):
    return evaluate(ann, preds, classes, videos)
