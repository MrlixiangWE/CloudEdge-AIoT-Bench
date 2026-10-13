import itertools

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

import apps
from apps.common import assets as A
from apps.common import w3 as L

EDGE, CLOUD = {"edge": 1}, {"cloud": 1}
RUNTIME_ENV = {"py_modules": [apps], "pip": ["numpy", "pillow", "torch", "torchvision"]}  # ship app code and deps


@ray.remote(num_cpus=1, memory=1 << 30, resources=CLOUD)
def load_image(frame_idx):
    return L.load_image(frame_idx)


@ray.remote(num_cpus=1, memory=1 << 30, resources=CLOUD)
def preprocess(image):
    return L.preprocess_input(image)


@ray.remote(memory=2 << 30, resources=EDGE, max_restarts=-1, max_task_retries=-1)
class Detector:
    """Keeps Faster R-CNN resident (a task would reload it whenever Ray recycles its worker process);
    restarted with its model on another node after a failure."""

    def __init__(self, threads):
        A.torch().set_num_threads(threads)  # native threads sized to the CPUs this replica holds
        self.model = L.load_detector()

    def infer(self, tensor):
        return L.frcnn_infer(tensor, self.model)


@ray.remote(num_cpus=2, memory=1 << 30, resources=CLOUD)
def postprocess(raw):
    return L.postprocess(raw)


@ray.remote(num_cpus=1, memory=512 << 20, resources=CLOUD)
def aggregate(dets):
    return L.count_classes(dets)


def replicas(actor, tier, cpus, share):
    """Per-device replicas: each node of the tier gets as many as its CPU share allows, pinned there."""
    pool = []
    for node in ray.nodes():
        if node["Alive"] and tier in node["Resources"]:
            pin = NodeAffinitySchedulingStrategy(node["NodeID"], soft=True)
            for _ in range(max(1, int(node["Resources"]["CPU"] * share) // cpus)):
                pool.append(actor.options(num_cpus=cpus, scheduling_strategy=pin).remote(cpus))
    return itertools.cycle(pool)


def workflow(detectors, req_id, frame_idx):
    image = load_image.remote(frame_idx)
    tensor = preprocess.remote(image)
    raw = next(detectors).infer.remote(tensor)
    dets = postprocess.remote(raw)
    return aggregate.remote(dets)


if __name__ == "__main__":
    ray.init(runtime_env=RUNTIME_ENV)
    detectors = replicas(Detector, "edge", cpus=2, share=1.0)
    refs = [workflow(detectors, *args) for args in L.inputs()]
    for out in ray.get(refs):
        print(out)
