import itertools

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

import apps
from apps.common import assets as A
from apps.common import w1 as L

EDGE, CLOUD = {"edge": 1}, {"cloud": 1}
RUNTIME_ENV = {"py_modules": [apps], "pip": ["numpy", "torch"]}  # ship app code and deps to every node


@ray.remote(num_cpus=1, memory=1 << 30, resources=EDGE)
def load_signal(signal):
    return L.decode_power_trace(signal)


@ray.remote(num_cpus=2, memory=1 << 30, resources=EDGE)
def preproc(trace):
    return L.normalize_window(trace)


@ray.remote(memory=2 << 30, resources=EDGE, max_restarts=-1, max_task_retries=-1)
class Recognizer:
    """Keeps the LSTM resident (a task would reload it whenever Ray recycles its worker process);
    restarted with its model on another node after a failure."""

    def __init__(self, threads):
        A.torch().set_num_threads(threads)  # native threads sized to the CPUs this replica holds
        self.model = L.load_recognizer()

    def infer(self, features):
        return L.lstm_infer(features, self.model)


@ray.remote(num_cpus=1, memory=512 << 20, resources=CLOUD)
def aggregate(label):
    return L.majority_vote(label)


def replicas(actor, tier, cpus, share):
    """Per-device replicas: each node of the tier gets as many as its CPU share allows, pinned there."""
    pool = []
    for node in ray.nodes():
        if node["Alive"] and tier in node["Resources"]:
            pin = NodeAffinitySchedulingStrategy(node["NodeID"], soft=True)
            for _ in range(max(1, int(node["Resources"]["CPU"] * share) // cpus)):
                pool.append(actor.options(num_cpus=cpus, scheduling_strategy=pin).remote(cpus))
    return itertools.cycle(pool)


def workflow(recognizers, req_id, signal):
    trace = load_signal.remote(signal)
    features = preproc.remote(trace)
    label = next(recognizers).infer.remote(features)
    return aggregate.remote(label)


if __name__ == "__main__":
    ray.init(runtime_env=RUNTIME_ENV)
    recognizers = replicas(Recognizer, "edge", cpus=2, share=0.5)
    refs = [workflow(recognizers, *args) for args in L.inputs()]
    for out in ray.get(refs):
        print(out)
