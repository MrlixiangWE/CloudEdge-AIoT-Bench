from hedices import NoRely, Stem, Workflow
from hedices.master import Master
from apps.common import w3 as L


@Stem("cloud", cpus=1, mem="1GB")
def load_image(req_id, frame_idx):
    return L.load_image(frame_idx)


@Stem("cloud", cpus=1, mem="1GB")
def preprocess(req_id, image):
    return L.preprocess_input(image)


@Stem("edge", cpus=2, mem="2GB")
@NoRely(flag="req_id")
def infer(req_id, tensor):
    return L.frcnn_infer(tensor)


@Stem("cloud", cpus=2, mem="1GB")
def postprocess(req_id, raw):
    return L.postprocess(raw)


@Stem("cloud", cpus=1, mem="512MB")
def aggregate(req_id, dets):
    return L.count_classes(dets)


def workflow(req_id, frame_idx):
    image = load_image(req_id, frame_idx)
    tensor = preprocess(req_id, image)
    raw = infer(req_id, tensor)
    dets = postprocess(req_id, raw)
    return aggregate(req_id, dets)


if __name__ == "__main__":
    with Master() as stem:
        wf = Workflow(workflow)
        futures = [stem.submit(wf, *args) for args in L.inputs()]
        for f in futures:
            print(f.result())
