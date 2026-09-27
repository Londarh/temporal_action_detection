"""Decoding pyramid outputs into segments, and Gaussian Soft-NMS."""
import numpy as np
import torch


def tiou(seg, segs):
    inter = np.clip(np.minimum(seg[1], segs[:, 1]) - np.maximum(seg[0], segs[:, 0]), 0, None)
    union = (seg[1] - seg[0]) + (segs[:, 1] - segs[:, 0]) - inter
    return inter / np.maximum(union, 1e-8)


def soft_nms(segs, scores, sigma=0.5, min_score=0.001, max_num=200):
    """Gaussian Soft-NMS: overlapping segments are down-weighted, not deleted."""
    segs, scores = segs.copy(), scores.copy()
    keep_segs, keep_scores = [], []
    while scores.size and len(keep_scores) < max_num:
        i = int(scores.argmax())
        best_seg, best_score = segs[i].copy(), scores[i]
        keep_segs.append(best_seg)
        keep_scores.append(best_score)
        segs, scores = np.delete(segs, i, 0), np.delete(scores, i)
        if not scores.size:
            break
        scores = scores * np.exp(-(tiou(best_seg, segs) ** 2) / sigma)
        alive = scores > min_score
        segs, scores = segs[alive], scores[alive]
    return np.array(keep_segs).reshape(-1, 2), np.array(keep_scores)


@torch.no_grad()
def decode_video(model, cls_out, reg_out, masks, b, pre_thresh=0.001, topk=2000, min_dur=0.05,
                 sigma=0.5, min_score=0.001, max_num=200):
    """Turns one video's pyramid outputs (batch index ``b``) into (segments_grid, scores, labels)."""
    lengths = [c.shape[1] for c in cls_out]
    points = model.points(lengths, cls_out[0].device)
    all_segs, all_scores, all_labels = [], [], []
    K = model.num_classes
    for cls, reg, m, pts in zip(cls_out, reg_out, masks, points):
        prob = (cls[b].sigmoid() * m[b, :, None]).flatten()
        idx = (prob > pre_thresh).nonzero(as_tuple=True)[0]
        prob = prob[idx]
        order = prob.argsort(descending=True)[:topk]
        prob, idx = prob[order], idx[order]
        pt_idx, lab = idx // K, idx % K
        off, p = reg[b][pt_idx], pts[pt_idx]
        segs = torch.stack([p[:, 0] - off[:, 0] * p[:, 3], p[:, 0] + off[:, 1] * p[:, 3]], 1)
        ok = (segs[:, 1] - segs[:, 0]) > min_dur
        all_segs.append(segs[ok]), all_scores.append(prob[ok]), all_labels.append(lab[ok])
    segs = torch.cat(all_segs).cpu().numpy()
    scores = torch.cat(all_scores).cpu().numpy()
    labels = torch.cat(all_labels).cpu().numpy()

    out_s, out_p, out_l = [], [], []
    for c in np.unique(labels):  # class-wise Soft-NMS
        sel = labels == c
        s, p = soft_nms(segs[sel], scores[sel], sigma, min_score, max_num)
        out_s.append(s), out_p.append(p), out_l.append(np.full(len(p), c))
    if not out_s:
        return np.zeros((0, 2)), np.zeros(0), np.zeros(0, int)
    segs, scores, labels = np.concatenate(out_s), np.concatenate(out_p), np.concatenate(out_l)
    order = scores.argsort()[::-1][:max_num]
    return segs[order], scores[order], labels[order]
