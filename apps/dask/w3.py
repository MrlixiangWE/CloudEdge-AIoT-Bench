import shutil
from pathlib import Path

from dask.distributed import Client, PipInstall, WorkerPlugin, get_worker

import apps
from apps.common import assets as A
from apps.common import w3 as L

GB = 1 << 30


class Models(WorkerPlugin):
    """Loads Faster R-CNN once on every edge worker before tasks arrive, with native threads
    limited to the CPUs one infer task reserves (workers run several tasks concurrently)."""

    name = "w3-models"

    def setup(self, worker):
        if "edge" in worker.state.total_resources:
            A.torch().set_num_threads(RESOURCES[infer]["CPU"])
            self.detector = L.load_detector()


def load_image(frame_idx):
    return L.load_image(frame_idx)


def preprocess(image):
    return L.preprocess_input(image)


def infer(tensor):
    return L.frcnn_infer(tensor, get_worker().plugins[Models.name].detector)


def postprocess(raw):
    return L.postprocess(raw)


def aggregate(dets):
    return L.count_classes(dets)


# placement tier and resource demand of each stage (workers advertise their tier, CPU and MEM)
RESOURCES = {load_image: {"cloud": 1, "CPU": 1, "MEM": GB},
             preprocess: {"cloud": 1, "CPU": 1, "MEM": GB},
             infer: {"edge": 1, "CPU": 2, "MEM": 2 * GB},
             postprocess: {"edge": 1, "CPU": 2, "MEM": GB},
             aggregate: {"cloud": 1, "CPU": 1, "MEM": GB // 2}}


def submit(client, stage, req_id, *args):
    # explicit keys: Dask would otherwise merge repeated triggers that carry identical inputs
    return client.submit(stage, *args, key=f"{stage.__name__}-{req_id}", resources=RESOURCES[stage])


def workflow(client, req_id, frame_idx):
    image = submit(client, load_image, req_id, frame_idx)
    tensor = submit(client, preprocess, req_id, image)
    raw = submit(client, infer, req_id, tensor)
    dets = submit(client, postprocess, req_id, raw)
    return submit(client, aggregate, req_id, dets)


def deploy(client, packages):
    """Install dependencies and ship the application package to every worker."""
    client.register_plugin(PipInstall(packages))
    client.upload_file(shutil.make_archive("apps", "zip", Path(apps.__file__).parent.parent, "apps"))


if __name__ == "__main__":
    client = Client()
    deploy(client, ["numpy", "pillow", "torch", "torchvision"])
    client.register_plugin(Models())
    futures = [workflow(client, *args) for args in L.inputs()]
    for out in client.gather(futures):
        print(out)
