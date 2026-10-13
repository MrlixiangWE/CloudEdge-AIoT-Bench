"""YOLOX-nano helpers shared by W2 and W4: input letterboxing, ONNX session, output decoding."""
import os

import numpy as np

from apps.common.assets import ASSETS


def letterbox(img, size):
    """Resize-and-pad an RGB frame into a normalized NCHW tensor for YOLOX."""
    from PIL import Image

    r = min(size / img.shape[0], size / img.shape[1])
    h, w = int(img.shape[0] * r), int(img.shape[1] * r)
    out = np.full((size, size, 3), 114, dtype=np.uint8)
    out[:h, :w] = np.asarray(Image.fromarray(img).resize((w, h), Image.BILINEAR))
    x = out[:, :, ::-1].astype(np.float32) / 255.0
    x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None]), r


def yolox_session(path, threads=None):
    """ONNX Runtime session; threads=None keeps the library default (all cores)."""
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = threads or int(os.environ.get("OMP_NUM_THREADS", "0"))
    so.inter_op_num_threads = 1
    return ort.InferenceSession(str(ASSETS / path), so, providers=["CPUExecutionProvider"])


def yolox_decode(outputs, size):
    grids, strides = [], []
    for stride in (8, 16, 32):
        n = size // stride
        xv, yv = np.meshgrid(np.arange(n), np.arange(n))
        grids.append(np.stack((xv, yv), 2).reshape(1, -1, 2))
        strides.append(np.full((1, n * n, 1), stride))
    grids, strides = np.concatenate(grids, 1), np.concatenate(strides, 1)
    outputs[..., :2] = (outputs[..., :2] + grids) * strides
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * strides
    return outputs
