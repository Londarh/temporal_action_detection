"""ActivityNet-style temporal detection mAP (the protocol used for THUMOS-14 in
ActionFormer, TriDet and OpenTAD): all-point interpolated AP per class, averaged
over classes, reported at tIoU 0.3, 0.4, 0.5, 0.6, 0.7 and their mean."""
from collections import defaultdict

import numpy as np

TIOUS = (0.3, 0.4, 0.5, 0.6, 0.7)


def _tiou(seg, segs):
    inter = np.clip(np.minimum(seg[1], segs[:, 1]) - np.maximum(seg[0], segs[:, 0]), 0, None)
    union = (segs[:, 1] - segs[:, 0]) + (seg[1] - seg[0]) - inter
    return inter / union


def _interpolated_ap(prec, rec):
    mprec = np.hstack([[0], prec, [0]])
    mrec = np.hstack([[0], rec, [1]])
    for i in range(len(mprec) - 1)[::-1]:
        mprec[i] = max(mprec[i], mprec[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0] + 1
    return float(np.sum((mrec[idx] - mrec[idx - 1]) * mprec[idx]))


def average_precision(gt, preds, tious=TIOUS):
    """gt: {video: [[s, e], ...]} for one class; preds: list of (video, s, e, score)."""
    ap = np.zeros(len(tious))
    npos = sum(len(v) for v in gt.values())
    if npos == 0 or not preds:
        return ap
    preds = sorted(preds, key=lambda p: -p[3])
    gt = {v: np.asarray(s, float) for v, s in gt.items()}
    locked = {v: np.zeros((len(tious), len(s)), bool) for v, s in gt.items()}
    tp = np.zeros((len(tious), len(preds)))
    fp = np.zeros((len(tious), len(preds)))
    for i, (vid, s, e, _) in enumerate(preds):
        if vid not in gt:
            fp[:, i] = 1
            continue
        ious = _tiou(np.array([s, e]), gt[vid])
        order = ious.argsort()[::-1]
        for t, thr in enumerate(tious):
            for j in order:
                if ious[j] < thr:
                    break
                if locked[vid][t, j]:
                    continue
                tp[t, i] = 1
                locked[vid][t, j] = True
                break
            if tp[t, i] == 0:
                fp[t, i] = 1
    tp_c, fp_c = np.cumsum(tp, 1), np.cumsum(fp, 1)
    rec, prec = tp_c / npos, tp_c / (tp_c + fp_c)
    for t in range(len(tious)):
        ap[t] = _interpolated_ap(prec[t], rec[t])
    return ap


def evaluate(ann, predictions, classes, videos=None, tious=TIOUS):
    """ann: {video: [(s, e, class_id)]} in seconds; predictions: list of dicts
    {video, start, end, label, score}. Only ``videos`` (default: all annotated) are scored.

    Returns {"mAP": {tiou: value}, "avg_mAP": float, "per_class_AP": {class: [AP per tiou]}}.
    """
    videos = set(videos) if videos is not None else set(ann)
    gt = defaultdict(lambda: defaultdict(list))
    for v in videos:
        for s, e, c in ann.get(v, []):
            gt[c][v].append([s, e])
    by_cls = defaultdict(list)
    for p in predictions:
        if p["video"] in videos:
            by_cls[p["label"]].append((p["video"], p["start"], p["end"], p["score"]))
    per_class = np.stack([average_precision(gt[c], by_cls[c], tious) for c in range(len(classes))])
    mAP = per_class.mean(0)
    return {
        "mAP": {f"{t:.1f}": round(float(m) * 100, 2) for t, m in zip(tious, mAP)},
        "avg_mAP": round(float(mAP.mean()) * 100, 2),
        "per_class_AP": {classes[c]: [round(float(x) * 100, 2) for x in per_class[c]] for c in range(len(classes))},
    }
