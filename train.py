"""Train on the THUMOS-14 validation videos, evaluate once on the test videos.

Usage:
    python train.py --config configs/thumos_i3d.yaml --out runs/i3d_transformer
    python train.py --config configs/thumos_i3d.yaml --out runs/i3d_conv --set model.neck=conv
"""
import argparse
import json
import os
import random

import numpy as np
import torch
import yaml

from tad.data import FeatureStore, ThumosTrainSet, load_annotations
from tad.engine import predict, save_json, train
from tad.evaluate import evaluate
from tad.model import TemporalDetector


def apply_overrides(cfg, overrides):
    for item in overrides or []:
        key, value = item.split("=", 1)
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node[p]
        node[leaf] = yaml.safe_load(value)
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/thumos_i3d.yaml")
    ap.add_argument("--out", default="runs/i3d_transformer")
    ap.add_argument("--set", nargs="*", help="override config values, e.g. model.neck=conv train.seed=1")
    ap.add_argument("--max_videos", type=int, default=None, help="debug: use only a few videos")
    args = ap.parse_args()

    cfg = apply_overrides(yaml.safe_load(open(args.config)), args.set)
    seed = cfg["train"]["seed"]
    random.seed(seed), np.random.seed(seed), torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)
    log_file = open(os.path.join(args.out, "train_log.txt"), "w")

    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    classes, ann, fps = load_annotations(cfg["data"]["ann_dirs"], cfg["data"].get("json_file"))
    store = FeatureStore(cfg["data"]["feat_dir"])
    train_videos = store.split("validation")
    test_videos = [v for v in store.split("test") if v in ann]
    if args.max_videos:
        train_videos, test_videos = train_videos[: args.max_videos], test_videos[: args.max_videos]
    log(f"device {device} | {len(classes)} classes | train videos {len(train_videos)} | test videos {len(test_videos)}")

    model = TemporalDetector(num_classes=len(classes), **cfg["model"])
    log(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    train_set = ThumosTrainSet(store, train_videos, ann, fps, max_len=cfg["train"]["max_len"])
    model, history = train(model, train_set, cfg["train"], device, log)
    torch.save({"model": model.state_dict(), "config": cfg, "classes": classes}, os.path.join(args.out, "model.pt"))

    preds = predict(model, store, test_videos, fps, device, classes)
    results = evaluate(ann, preds, classes, test_videos)
    results["config"], results["history"] = cfg, history
    save_json(preds, os.path.join(args.out, "test_predictions.json"))
    save_json(results, os.path.join(args.out, "results.json"))
    log("test mAP by tIoU: " + json.dumps(results["mAP"]) + f" | average: {results['avg_mAP']:.2f}")


if __name__ == "__main__":
    main()
