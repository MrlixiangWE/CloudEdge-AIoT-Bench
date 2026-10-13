import os
import pickle
from concurrent.futures import ThreadPoolExecutor

import requests

from apps.common import w4 as L

URLS = {s: os.environ.get(f"{s.upper()}_URL", f"http://w4-{s.replace('_', '-')}.default.svc.cluster.local")
        for s in ("preproc_frame", "detect_objects", "preproc_imu", "fuse", "aggregate")}
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


def workflow(req_id, frame, imu_sample):
    objects = branches.submit(lambda: call("detect_objects", call("preproc_frame", frame)))  # camera and IMU
    imu_state = call("preproc_imu", imu_sample)  # branches run concurrently
    logits = call("fuse", objects.result(), imu_state)
    return call("aggregate", logits)


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=8) as pool:
        for out in pool.map(lambda args: workflow(*args), L.inputs()):
            print(out)
