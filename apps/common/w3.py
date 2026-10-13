"""W3 Detection: Faster R-CNN with cloud-side I/O, preprocessing and post-processing, edge-side inference.

The default is the ResNet-50 model on 600x600 frames (decoded with DecodeBox + per-class
NMS); HEDICES_W3_MODEL=mobilenet selects the torchvision MobileNetV3-320 variant on 320x320 frames.
"""
import os

import numpy as np

from apps.common import assets as A

MODEL = os.environ.get("HEDICES_W3_MODEL", "resnet50")
SIZE = 600 if MODEL == "resnet50" else 320
NUM_CLASSES = 20
STD = (0.1, 0.1, 0.2, 0.2)


@A.once
def load_detector():
    import torchvision

    if MODEL == "resnet50":
        return A.torch().jit.load(str(A.ASSETS / "script_model_cpu.pt"), map_location="cpu").eval()
    return A.seeded(lambda: torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(
        weights=None, weights_backbone=None, num_classes=NUM_CLASSES + 1,
        box_score_thresh=0.0, box_nms_thresh=1.0, box_detections_per_img=100))


def load_image(frame_idx):
    return A.frame(frame_idx, (SIZE, SIZE))


def preprocess_input(image):
    x = image.astype(np.float32) / 255.0
    return np.ascontiguousarray(np.transpose(x, (2, 0, 1))[None])


def frcnn_infer(tensor, model=None):
    t = A.torch()
    model = model or load_detector()
    x = t.from_numpy(tensor)
    with t.no_grad():
        if MODEL == "resnet50":
            cls_locs, scores, rois, _ = model(x)
            return MODEL, (cls_locs.numpy(), scores.numpy(), rois.numpy())
        out = model([x[0]])[0]
        return MODEL, (out["boxes"].numpy(), out["scores"].numpy(), out["labels"].numpy())


def loc2bbox(t, src, loc):
    w = src[:, 2:3] - src[:, 0:1]
    h = src[:, 3:4] - src[:, 1:2]
    cx = src[:, 0:1] + 0.5 * w
    cy = src[:, 1:2] + 0.5 * h
    dx, dy, dw, dh = loc[:, 0::4], loc[:, 1::4], loc[:, 2::4], loc[:, 3::4]
    ncx, ncy = dx * w + cx, dy * h + cy
    nw, nh = t.exp(dw) * w, t.exp(dh) * h
    out = t.zeros_like(loc)
    out[:, 0::4], out[:, 1::4] = ncx - 0.5 * nw, ncy - 0.5 * nh
    out[:, 2::4], out[:, 3::4] = ncx + 0.5 * nw, ncy + 0.5 * nh
    return out


def decode_boxes(raw, confidence=0.5, nms_iou=0.3):
    t = A.torch()
    from torchvision.ops import nms

    cls_locs, scores, rois = (t.from_numpy(a) for a in raw)
    n = NUM_CLASSES + 1
    loc = (cls_locs[0] * t.tensor(STD).repeat(n)).view(-1, n, 4)
    roi = rois.view(-1, 1, 4).expand_as(loc)
    boxes = loc2bbox(t, roi.reshape(-1, 4), loc.reshape(-1, 4)).view(-1, n, 4)
    prob = t.softmax(scores[0], dim=-1)
    out = []
    for c in range(1, n):
        m = prob[:, c] > confidence
        if m.any():
            b, s = boxes[m, c], prob[m, c]
            for k in nms(b, s, nms_iou).tolist():
                out.append([*b[k].tolist(), float(s[k]), c - 1])
    return np.array(out, np.float32).reshape(-1, 6)


def filter_candidates(raw, confidence=0.02, nms_iou=0.3):
    t = A.torch()
    from torchvision.ops import batched_nms

    boxes, scores, labels = (t.from_numpy(a) for a in raw)
    m = scores > confidence
    boxes, scores, labels = boxes[m], scores[m], labels[m]
    keep = batched_nms(boxes, scores, labels, nms_iou)
    return t.cat([boxes[keep], scores[keep, None], (labels[keep, None] - 1).float()], 1).numpy()


def postprocess(raw):
    kind, tensors = raw
    return decode_boxes(tensors) if kind == "resnet50" else filter_candidates(tensors)


def count_classes(dets):
    classes, counts = np.unique(dets[:, 5].astype(int), return_counts=True) if len(dets) else ([], [])
    return {int(c): int(n) for c, n in zip(classes, counts)}


def inputs(n=8):
    return [(f"w3-{i}", i) for i in range(n)]
