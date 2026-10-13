import shutil
from pathlib import Path

from dask.distributed import Client, PipInstall, WorkerPlugin, get_worker

import apps
from apps.common import assets as A
from apps.common import w4 as L

GB = 1 << 30


class Models(WorkerPlugin):
    """Loads YOLOX on edge workers and the fusion CNN on cloud workers before tasks arrive, with
    native threads limited to the CPUs one task reserves (workers run several tasks concurrently)."""

    name = "w4-models"

    def setup(self, worker):
        tiers = worker.state.total_resources
        if "edge" in tiers:
            self.detector = L.new_detector(RESOURCES[detect_objects]["CPU"])
        if "cloud" in tiers:
            A.torch().set_num_threads(RESOURCES[fuse]["CPU"])
            self.fusion = L.load_fusion()


def preproc_frame(frame):
    return L.preprocess_frame(frame)


def detect_objects(tensor):
    return L.detect(tensor, get_worker().plugins[Models.name].detector)


def preproc_imu(imu_sample):
    return L.filter_and_normalize(imu_sample)


def fuse(objects, imu_state):
    return L.fusion_model(objects, imu_state, get_worker().plugins[Models.name].fusion)


def aggregate(logits):
    return L.aggregate_decision(logits)


# placement tier and resource demand of each stage (workers advertise their tier, CPU and MEM)
RESOURCES = {preproc_frame: {"edge": 1, "CPU": 1, "MEM": GB // 2},
             detect_objects: {"edge": 1, "CPU": 2, "MEM": 2 * GB},
             preproc_imu: {"edge": 1, "CPU": 1, "MEM": GB // 2},
             fuse: {"cloud": 1, "CPU": 4, "MEM": 4 * GB},
             aggregate: {"cloud": 1, "CPU": 1, "MEM": GB // 2}}


def submit(client, stage, req_id, *args):
    # explicit keys: Dask would otherwise merge repeated triggers that carry identical inputs
    return client.submit(stage, *args, key=f"{stage.__name__}-{req_id}", resources=RESOURCES[stage])


def workflow(client, req_id, frame, imu_sample):
    tensor = submit(client, preproc_frame, req_id, frame)
    objects = submit(client, detect_objects, req_id, tensor)
    imu_state = submit(client, preproc_imu, req_id, imu_sample)
    logits = submit(client, fuse, req_id, objects, imu_state)
    return submit(client, aggregate, req_id, logits)


def deploy(client, packages):
    """Install dependencies and ship the application package to every worker."""
    client.register_plugin(PipInstall(packages))
    client.upload_file(shutil.make_archive("apps", "zip", Path(apps.__file__).parent.parent, "apps"))


if __name__ == "__main__":
    client = Client()
    deploy(client, ["numpy", "pillow", "onnxruntime", "torch"])
    client.register_plugin(Models())
    futures = [workflow(client, *args) for args in L.inputs()]
    for out in client.gather(futures):
        print(out)
