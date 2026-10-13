"""W2 Tracking: YOLOX detect-then-track on 224x224 frames with periodic re-detection.

Camera frames are read, preprocessed and tracked on the edge. About 30% of the frames are
re-detected with YOLOX in the cloud (retrack); on the other 70% the edge updates its track table
(track update). Both branches are aggregated in the cloud into one per-frame track summary.
"""
import numpy as np

from apps.common import assets as A
from apps.common import yolox as Y

REDETECT_PER_10 = 3  # frames out of every 10 that are re-detected in the cloud
INPUT = 320  # 224x224 frames are letterboxed into the 320x320 YOLOX-nano input
MAX_TRACK_AGE = 6
GRID = 2  # track windows per side


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


def track(frame, frame_idx):
    """Edge-side appearance tracking: colour-histogram descriptors of the track windows."""
    h, w = frame.shape[:2]
    windows = []
    for k in range(GRID * GRID):
        r, c = divmod(k, GRID)
        patch = frame[r * h // GRID:(r + 1) * h // GRID, c * w // GRID:(c + 1) * w // GRID]
        hist = np.concatenate([np.histogram(patch[..., ch], bins=16, range=(0, 255))[0] for ch in range(3)])
        windows.append(hist.astype(np.float32) / patch.size)
    return {"frame": frame_idx, "redetect": frame_idx % 10 < REDETECT_PER_10, "windows": np.stack(windows)}


def retrack(tensor, tracks, sess=None):
    """Cloud-side re-detection of the ~30% frames flagged by the tracker; None on the others."""
    return detect(tensor, sess) if tracks["redetect"] else None


def track_update(tracks):
    """Edge-side track-table update on the ~70% frames that are not re-detected."""
    if tracks["redetect"]:
        return None
    return {"frame": tracks["frame"], "active": len(tracks["windows"]),
            "expire_before": tracks["frame"] - MAX_TRACK_AGE}


def summarize_tracks(tracks, dets, update):
    """Cloud-side aggregation of whichever branch ran into one per-frame track summary."""
    if dets is not None:
        return {"frame": tracks["frame"], "source": "retrack", "n": len(dets),
                "classes": sorted({int(c) for c in dets[:, 5]})}
    return {"frame": tracks["frame"], "source": "track_update", "n": update["active"],
            "expire_before": update["expire_before"]}


def inputs(n=8):
    return [(f"w2-{i}", i) for i in range(n)]
