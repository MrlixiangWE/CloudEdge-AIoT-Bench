import os
import pickle
from concurrent.futures import ThreadPoolExecutor

import requests

from apps.common import w1 as L

URLS = {s: os.environ.get(f"{s.upper()}_URL", f"http://w1-{s.replace('_', '-')}.default.svc.cluster.local")
        for s in ("load_signal", "preproc", "recognize", "aggregate")}
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


def workflow(req_id, signal):
    trace = call("load_signal", signal)
    features = call("preproc", trace)
    label = call("recognize", features)
    return call("aggregate", label)


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=8) as pool:
        for out in pool.map(lambda args: workflow(*args), L.inputs()):
            print(out)
