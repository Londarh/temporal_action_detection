"""Target assignment and losses (sigmoid focal + 1D DIoU)."""
import torch
import torch.nn.functional as F


def assign_targets(points, segments, labels, num_classes, center_radius=1.5):
    """Label every pyramid point of one video.

    points: [N, 4] (position, range_lo, range_hi, stride) concatenated over levels.
    segments: [G, 2] ground-truth segments on the feature grid; labels: [G].
    Returns cls_targets [N, K] (multi-label 0/1) and reg_targets [N, 2] (distances to
    start/end divided by the level stride).
    """
    N, G = points.shape[0], segments.shape[0]
    if G == 0:
        return points.new_zeros(N, num_classes), points.new_zeros(N, 2)

    pos, lo, hi, stride = points[:, 0:1], points[:, 1:2], points[:, 2:3], points[:, 3:4]
    s, e = segments[None, :, 0], segments[None, :, 1]  # [1, G]
    left, right = pos - s, e - pos  # [N, G]
    reg = torch.stack([left, right], -1)

    # centre sampling: only points near the action centre are positives
    ctr = 0.5 * (s + e)
    t_min = torch.maximum(ctr - stride * center_radius, s)
    t_max = torch.minimum(ctr + stride * center_radius, e)
    inside = torch.minimum(pos - t_min, t_max - pos) > 0

    # each pyramid level handles one range of action lengths
    max_dist = reg.max(-1).values
    in_range = (max_dist >= lo) & (max_dist <= hi)

    lens = (e - s).expand(N, G).clone()
    lens[~(inside & in_range)] = float("inf")
    min_len, min_idx = lens.min(1)
    # several actions with (almost) the same extent -> keep all their labels (multi-label)
    same = (lens <= min_len[:, None] + 1e-3) & torch.isfinite(lens)
    cls_t = (same.float() @ F.one_hot(labels, num_classes).float()).clamp(max=1.0)
    reg_t = reg[torch.arange(N), min_idx] / stride
    return cls_t, reg_t


def sigmoid_focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce * (1 - p_t) ** gamma
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (alpha_t * loss).sum()


def diou_loss_1d(pred, target, eps=1e-8):
    """DIoU loss for segments sharing the same anchor point, given (left, right) offsets."""
    lp, rp = pred[:, 0], pred[:, 1]
    lg, rg = target[:, 0], target[:, 1]
    inter = torch.min(lp, lg) + torch.min(rp, rg)
    union = (lp + rp) + (lg + rg) - inter
    iou = inter / union.clamp(min=eps)
    enclosing = torch.max(lp, lg) + torch.max(rp, rg)
    centre_dist = 0.5 * (rp - lp - rg + lg)
    return (1 - iou + (centre_dist / enclosing.clamp(min=eps)) ** 2).sum()


class DetectionLoss:
    def __init__(self, num_classes, init_norm=100.0, momentum=0.9, reg_weight=1.0):
        self.num_classes, self.norm, self.momentum, self.reg_weight = num_classes, init_norm, momentum, reg_weight

    def __call__(self, model, cls_out, reg_out, masks, batch):
        lengths = [c.shape[1] for c in cls_out]
        points = torch.cat(model.points(lengths, cls_out[0].device))
        cls_t, reg_t = [], []
        for b in batch:
            c, r = assign_targets(points, b["segments"].to(points.device), b["labels"].to(points.device), self.num_classes)
            cls_t.append(c)
            reg_t.append(r)
        cls_t, reg_t = torch.stack(cls_t), torch.stack(reg_t)
        valid = torch.cat(masks, 1)
        pos = (cls_t.sum(-1) > 0) & valid
        num_pos = pos.sum().item()
        self.norm = self.momentum * self.norm + (1 - self.momentum) * max(num_pos, 1)

        cls_logits, offsets = torch.cat(cls_out, 1), torch.cat(reg_out, 1)
        cls_loss = sigmoid_focal_loss(cls_logits[valid], cls_t[valid]) / self.norm
        reg_loss = diou_loss_1d(offsets[pos], reg_t[pos]) / self.norm if num_pos else offsets.sum() * 0
        return cls_loss + self.reg_weight * reg_loss, {"cls": cls_loss.item(), "reg": reg_loss.item()}
