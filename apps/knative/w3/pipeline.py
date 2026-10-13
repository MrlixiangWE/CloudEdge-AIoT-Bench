import os
import pickle
from concurrent.futures import ThreadPoolExecutor

import requests

from apps.common import w3 as L

URLS = {s: os.environ.get(f"{s.upper()}_URL", f"http://w3-{s.replace('_', '-')}.default.svc.cluster.local")
        for s in ("load_image", "preprocess", "infer", "postprocess", "aggregate")}
session = requests.Session()


def call(stage, *args, attempts=3):
    """Invoke one stage service; a call lost with a failed pod is retried on another replica."""
    for i in range(attempts):
        try:
            r = session.post(f"{URLS[stage]}/{stage}", data=pickle.dumps(args), timeout=120)
            r.raise_for_status()
            return pickle.loads(r.content)
        except requests.RequestException:
            if i == attempts - 1:
                raise


def workflow(req_id, frame_idx):
    image = call("load_image", frame_idx)
    tensor = call("preprocess", image)
    raw = call("infer", tensor)
    dets = call("postprocess", raw)
    return call("aggregate", dets)


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=8) as pool:
        for out in pool.map(lambda args: workflow(*args), L.inputs()):
            print(out)
