import os
import pickle
from concurrent.futures import ThreadPoolExecutor

import requests

from apps.common import w2 as L

URLS = {s: os.environ.get(f"{s.upper()}_URL", f"http://w2-{s.replace('_', '-')}.default.svc.cluster.local")
        for s in ("read_frame", "preprocess", "track", "retrack", "track_update", "aggregate")}
session = requests.Session()
branches = ThreadPoolExecutor(max_workers=8)


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
    frame = call("read_frame", frame_idx)
    tensor = call("preprocess", frame)
    tracks = call("track", frame, frame_idx)
    dets = branches.submit(call, "retrack", tensor, tracks)  # cloud retrack and edge track
    update = call("track_update", tracks)  # update run concurrently
    return call("aggregate", tracks, dets.result(), update)


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=8) as pool:
        for out in pool.map(lambda args: workflow(*args), L.inputs()):
            print(out)
