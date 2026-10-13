"""W4 Fusion: branching camera + IMU pipeline with a cloud-side 4 MB fusion CNN."""
import numpy as np

from apps.common import assets as A
from apps.common import yolox as Y

INPUT = 320
GRID = 64
ACTIVITIES = ["idle", "walking", "running", "cycling", "driving", "stairs", "falling", "sitting",
              "standing", "lying"]


def new_detector(threads=None):
    return Y.yolox_session("yoloxna-320.onnx", threads)


load_detector = A.once(new_detector)


def _fusion_arch():
    nn = A.torch().nn
    return nn.Sequential(
        nn.Conv2d(24, 64, 3, padding=1), nn.ReLU(),
        nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
        nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU(),
        nn.Conv2d(256, 256, 3, stride=2, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, len(ACTIVITIES)))


@A.once
def load_fusion():
    return A.seeded(_fusion_arch)


def preprocess_frame(frame):
    return Y.letterbox(frame, INPUT)


def detect(tensor, sess=None, score_thr=0.25):
    x, ratio = tensor
    sess = sess or load_detector()
    pred = Y.yolox_decode(sess.run(None, {sess.get_inputs()[0].name: x})[0], INPUT)[0]
    scores = pred[:, 4:5] * pred[:, 5:]
    cls = scores.argmax(1)
    conf = scores[np.arange(len(cls)), cls]
    m = conf > score_thr
    b = pred[m, :4]
    xyxy = np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2,
                     b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], 1) / ratio
    return np.concatenate([xyxy, conf[m, None], cls[m, None]], 1).astype(np.float32)


def filter_and_normalize(imu_sample):
    k = np.full(9, 1 / 9, np.float32)
    smooth = np.stack([np.convolve(imu_sample[:, c], k, mode="same") for c in range(6)], 1)
    z = (smooth - smooth.mean(0)) / (smooth.std(0) + 1e-6)
    energy = (np.abs(np.fft.rfft(z, axis=0)[1:9]) ** 2).mean(0)
    return np.concatenate([smooth.mean(0), smooth.std(0), energy[:4]]).astype(np.float32)


def fusion_model(objects, imu_state, model=None):
    """Rasterize detections into per-class occupancy maps, stack IMU features, return logits."""
    t = A.torch()
    x = np.zeros((24, GRID, GRID), np.float32)
    for x1, y1, x2, y2, conf, cls in objects:
        c = int(cls) % 8
        a, b = int(max(x1, 0) / INPUT * GRID), int(max(y1, 0) / INPUT * GRID)
        e, f = int(min(x2, INPUT - 1) / INPUT * GRID) + 1, int(min(y2, INPUT - 1) / INPUT * GRID) + 1
        x[c, b:f, a:e] = np.maximum(x[c, b:f, a:e], conf)
    x[8:24] = imu_state[:, None, None]
    with t.no_grad():
        return (model or load_fusion())(t.from_numpy(x)[None])[0].numpy()


def aggregate_decision(logits):
    p = np.exp(logits - logits.max())
    p /= p.sum()
    k = int(p.argmax())
    return {"activity": ACTIVITIES[k], "confidence": round(float(p[k]), 4)}


def imu_window(idx, n=200):
    rng = np.random.default_rng(idx)
    t = np.arange(n, dtype=np.float32)[:, None]
    base = np.sin(t / 15.0 + rng.uniform(0, 6.28, (1, 6))).astype(np.float32)
    return base + 0.3 * rng.standard_normal((n, 6)).astype(np.float32)


def inputs(n=8):
    return [(f"w4-{i}", A.frame(i, (INPUT, INPUT)), imu_window(i)) for i in range(n)]
