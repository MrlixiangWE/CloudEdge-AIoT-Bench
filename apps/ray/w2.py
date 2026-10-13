import itertools

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

import apps
from apps.common import w2 as L

EDGE, CLOUD = {"edge": 1}, {"cloud": 1}
RUNTIME_ENV = {"py_modules": [apps], "pip": ["numpy", "pillow", "onnxruntime"]}  # ship app code and deps


@ray.remote(num_cpus=1, memory=1 << 30, resources=CLOUD)
def read_frame(frame_idx):
    return L.load_frame(frame_idx)


@ray.remote(num_cpus=1, memory=512 << 20, resources=EDGE)
def preprocess(frame):
    return L.preprocess(frame)


@ray.remote(memory=1 << 30, resources=EDGE, max_restarts=-1, max_task_retries=-1)
class Retracker:
    """Keeps the YOLOX session resident (a task would reload it whenever Ray recycles its worker
    process); restarted with its session on another node after a failure."""

    def __init__(self, threads):
        self.sess = L.new_detector(threads)  # native threads sized to the CPUs this replica holds

    def retrack(self, tensor, frame_idx):
        return L.redetect(tensor, frame_idx, self.sess)


@ray.remote(num_cpus=1, memory=512 << 20, resources=EDGE)
def track(frame, dets, frame_idx):
    return L.update_tracks(frame, dets, frame_idx)


@ray.remote(num_cpus=1, memory=512 << 20, resources=CLOUD)
def aggregate(tracks):
    return L.summarize_tracks(tracks)


@ray.remote(num_cpus=1, memory=256 << 20, resources=EDGE)
def track_update(summary):
    return L.track_update(summary)


def replicas(actor, tier, cpus, share):
    """Per-device replicas: each node of the tier gets as many as its CPU share allows, pinned there."""
    pool = []
    for node in ray.nodes():
        if node["Alive"] and tier in node["Resources"]:
            pin = NodeAffinitySchedulingStrategy(node["NodeID"], soft=True)
            for _ in range(max(1, int(node["Resources"]["CPU"] * share) // cpus)):
                pool.append(actor.options(num_cpus=cpus, scheduling_strategy=pin).remote(cpus))
    return itertools.cycle(pool)


def workflow(retrackers, req_id, frame_idx):
    frame = read_frame.remote(frame_idx)
    tensor = preprocess.remote(frame)
    dets = next(retrackers).retrack.remote(tensor, frame_idx)
    tracks = track.remote(frame, dets, frame_idx)
    summary = aggregate.remote(tracks)
    return track_update.remote(summary)


if __name__ == "__main__":
    ray.init(runtime_env=RUNTIME_ENV)
    retrackers = replicas(Retracker, "edge", cpus=2, share=0.5)
    refs = [workflow(retrackers, *args) for args in L.inputs()]
    for out in ray.get(refs):
        print(out)
