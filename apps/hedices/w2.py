from hedices import NoRely, Stem, Workflow
from hedices.master import Master
from apps.common import w2 as L


@Stem("cloud", cpus=1, mem="1GB")
def read_frame(req_id, frame_idx):
    return L.load_frame(frame_idx)


@Stem("edge", cpus=1, mem="512MB")
@NoRely(flag="req_id")
def preprocess(req_id, frame):
    return L.preprocess(frame)


@Stem("edge", cpus=2, mem="1GB")
@NoRely(flag="req_id")
def retrack(req_id, tensor, frame_idx):
    return L.redetect(tensor, frame_idx)


@Stem("edge", cpus=1, mem="512MB")
@NoRely(flag="req_id")
def track(req_id, frame, dets, frame_idx):
    return L.update_tracks(frame, dets, frame_idx)


@Stem("cloud", cpus=1, mem="512MB")
def aggregate(req_id, tracks):
    return L.summarize_tracks(tracks)


@Stem("edge", cpus=1, mem="256MB")
def track_update(req_id, summary):
    return L.track_update(summary)


def workflow(req_id, frame_idx):
    frame = read_frame(req_id, frame_idx)
    tensor = preprocess(req_id, frame)
    dets = retrack(req_id, tensor, frame_idx)
    tracks = track(req_id, frame, dets, frame_idx)
    summary = aggregate(req_id, tracks)
    return track_update(req_id, summary)


if __name__ == "__main__":
    with Master() as stem:
        wf = Workflow(workflow)
        futures = [stem.submit(wf, *args) for args in L.inputs()]
        for f in futures:
            print(f.result())
