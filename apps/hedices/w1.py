from hedices import NoRely, Stem, Workflow
from hedices.master import Master
from apps.common import w1 as L


@Stem("edge", cpus=1, mem="1GB")
def load_signal(req_id, signal):
    return L.decode_power_trace(signal)


@Stem("edge", cpus=2, mem="1GB")
@NoRely(flag="req_id")
def preproc(req_id, trace):
    return L.normalize_window(trace)


@Stem("edge", cpus=2, mem="2GB")
@NoRely(flag="req_id")
def recognize(req_id, features):
    return L.lstm_infer(features)


@Stem("cloud", cpus=1, mem="512MB")
def aggregate(req_id, label):
    return L.majority_vote(label)


def workflow(req_id, signal):
    trace = load_signal(req_id, signal)
    features = preproc(req_id, trace)
    label = recognize(req_id, features)
    return aggregate(req_id, label)


if __name__ == "__main__":
    with Master() as stem:
        wf = Workflow(workflow)
        futures = [stem.submit(wf, *args) for args in L.inputs()]
        for f in futures:
            print(f.result())
