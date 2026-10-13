from hedices import NoRely, Stem, Workflow
from hedices.master import Master
from apps.common import w4 as L


@Stem("edge", cpus=1, mem="512MB")
@NoRely(flag="req_id")
def preproc_frame(req_id, frame):
    return L.preprocess_frame(frame)


@Stem("edge", cpus=2, mem="2GB")
@NoRely(flag="req_id")
def detect_objects(req_id, tensor):
    return L.detect(tensor)


@Stem("edge", cpus=1, mem="512MB")
@NoRely(flag="req_id")
def preproc_imu(req_id, imu_sample):
    return L.filter_and_normalize(imu_sample)


@Stem("cloud", cpus=4, mem="4GB")
def fuse(req_id, objects, imu_state):
    return L.fusion_model(objects, imu_state)


@Stem("cloud", cpus=1, mem="512MB")
def aggregate(req_id, logits):
    return L.aggregate_decision(logits)


def workflow(req_id, frame, imu_sample):
    tensor = preproc_frame(req_id, frame)
    objects = detect_objects(req_id, tensor)
    imu_state = preproc_imu(req_id, imu_sample)
    logits = fuse(req_id, objects, imu_state)
    return aggregate(req_id, logits)


if __name__ == "__main__":
    with Master() as stem:
        wf = Workflow(workflow)
        futures = [stem.submit(wf, *args) for args in L.inputs()]
        for f in futures:
            print(f.result())
