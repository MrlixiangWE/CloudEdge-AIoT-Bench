import shutil
from pathlib import Path

from dask.distributed import Client, PipInstall, WorkerPlugin, get_worker

import apps
from apps.common import assets as A
from apps.common import w1 as L

GB = 1 << 30


class Models(WorkerPlugin):
    """Loads the LSTM once on every edge worker before tasks arrive, with native threads limited
    to the CPUs one recognize task reserves (workers run several tasks concurrently)."""

    name = "w1-models"

    def setup(self, worker):
        if "edge" in worker.state.total_resources:
            A.torch().set_num_threads(RESOURCES[recognize]["CPU"])
            self.recognizer = L.load_recognizer()


def load_signal(signal):
    return L.decode_power_trace(signal)


def preproc(trace):
    return L.normalize_window(trace)


def recognize(features):
    return L.lstm_infer(features, get_worker().plugins[Models.name].recognizer)


def aggregate(label):
    return L.majority_vote(label)


# placement tier and resource demand of each stage (workers advertise their tier, CPU and MEM)
RESOURCES = {load_signal: {"edge": 1, "CPU": 1, "MEM": GB},
             preproc: {"edge": 1, "CPU": 2, "MEM": GB},
             recognize: {"edge": 1, "CPU": 2, "MEM": 2 * GB},
             aggregate: {"cloud": 1, "CPU": 1, "MEM": GB // 2}}


def submit(client, stage, req_id, *args):
    # explicit keys: Dask would otherwise merge repeated triggers that carry identical inputs
    return client.submit(stage, *args, key=f"{stage.__name__}-{req_id}", resources=RESOURCES[stage])


def workflow(client, req_id, signal):
    trace = submit(client, load_signal, req_id, signal)
    features = submit(client, preproc, req_id, trace)
    label = submit(client, recognize, req_id, features)
    return submit(client, aggregate, req_id, label)


def deploy(client, packages):
    """Install dependencies and ship the application package to every worker."""
    client.register_plugin(PipInstall(packages))
    client.upload_file(shutil.make_archive("apps", "zip", Path(apps.__file__).parent.parent, "apps"))


if __name__ == "__main__":
    client = Client()
    deploy(client, ["numpy", "torch"])
    client.register_plugin(Models())
    futures = [workflow(client, *args) for args in L.inputs()]
    for out in client.gather(futures):
        print(out)
