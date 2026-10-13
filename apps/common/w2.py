"""W2 Tracking: YOLOX detect-then-track on 224x224 frames with periodic re-detection.

Frames are read on the cloud side; every frame is preprocessed and tracked on the edge, every
REDETECT_EVERY-th frame is re-detected with YOLOX, and track summaries are aggregated in the
cloud and pushed back to the edge track table.
"""
import numpy as np

from apps.common import assets as A
from apps.common import yolox as Y

REDETECT_EVERY = 3
INPUT = 320  # 224x224 frames are letterboxed into the 320x320 YOLOX-nano input
MAX_TRACK_AGE = 2 * REDETECT_EVERY


def new_detector(threads=None):
    return Y.yolox_session("yoloxna-320.onnx", threads)


load_detector = A.once(new_detector)


def load_frame(frame_idx):
    return A.frame(frame_idx, (224, 224))


def preprocess(frame):
    return Y.letterbox(frame, INPUT)


def nms(boxes, scores, thr):
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order, keep = scores.argsort()[::-1], []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1 + 1) * np.maximum(0.0, yy2 - yy1 + 1)
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(ovr <= thr)[0] + 1]
    return keep


def detect(tensor, sess=None, score_thr=0.25, nms_thr=0.45):
    x, ratio = tensor
    sess = sess or load_detector()
    pred = Y.yolox_decode(sess.run(None, {sess.get_inputs()[0].name: x})[0], INPUT)[0]
    scores = pred[:, 4:5] * pred[:, 5:]
    cls = scores.argmax(1)
    conf = scores[np.arange(len(cls)), cls]
    m = conf > score_thr
    if not m.any():
        return np.zeros((0, 6), np.float32)
    b = pred[m, :4]
    xyxy = np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2,
                     b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], 1) / ratio
    keep = nms(xyxy, conf[m], nms_thr)
    return np.concatenate([xyxy[keep], conf[m][keep, None], cls[m][keep, None]], 1).astype(np.float32)


def redetect(tensor, frame_idx, sess=None):
    """Re-detect on every REDETECT_EVERY-th frame; other frames keep the existing tracks."""
    return None if frame_idx % REDETECT_EVERY else detect(tensor, sess)


def update_tracks(frame, dets, frame_idx):
    """Appearance descriptors (colour histograms) for each tracked box."""
    if dets is None or len(dets) == 0:
        h, w = frame.shape[:2]
        dets = np.array([[0, 0, w, h, 0.0, -1]], np.float32)
    tracks = []
    for k, (x1, y1, x2, y2, conf, cls) in enumerate(dets):
        patch = frame[max(int(y1), 0):max(int(y2), 1), max(int(x1), 0):max(int(x2), 1)]
        if patch.size == 0:
            continue
        hist = np.concatenate([np.histogram(patch[..., c], bins=16, range=(0, 255))[0] for c in range(3)])
        tracks.append((k, int(cls), float(conf), hist.astype(np.float32) / max(patch.size, 1)))
    return {"frame": frame_idx, "redetect": frame_idx % REDETECT_EVERY == 0, "tracks": tracks}


def summarize_tracks(tracks):
    return {"frame": tracks["frame"], "n": len(tracks["tracks"]),
            "classes": sorted({c for _, c, _, _ in tracks["tracks"] if c >= 0})}


def track_update(summary):
    """Edge-side track-table update derived from the cloud summary: active classes and expiry."""
    return {"frame": summary["frame"], "active": summary["classes"],
            "expire_before": summary["frame"] - MAX_TRACK_AGE, "reset": summary["n"] == 0}


def inputs(n=8):
    return [(f"w2-{i}", i) for i in range(n)]
