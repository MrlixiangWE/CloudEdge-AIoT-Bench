"""Node-local model/data assets shared by the four applications, loaded once per process."""
import os
import threading
from pathlib import Path

import numpy as np

ASSETS = Path(os.environ.get("HEDICES_ASSETS", Path(__file__).resolve().parents[2] / "assets"))
_lock = threading.RLock()


def once(build):
    """Build an asset at most once per worker process, even under concurrent callers."""
    def get():
        if not hasattr(get, "value"):
            with _lock:
                if not hasattr(get, "value"):
                    get.value = build()
        return get.value
    return get


@once
def torch():
    import torch as t

    if os.environ.get("OMP_NUM_THREADS"):
        t.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    return t


def seeded(arch):
    """Instantiate a network with fixed initial weights (no trained checkpoint is shipped)."""
    torch().manual_seed(0)
    return arch().eval()


@once
def frames():
    z = np.load(ASSETS / "pics40.npz", allow_pickle=True)
    return [np.ascontiguousarray(f) for f in z["img"]]


def frame(idx, size):
    from PIL import Image

    f = frames()[idx % len(frames())]
    return np.asarray(Image.fromarray(f).resize(size, Image.BICUBIC))
