import os
import pickle

from flask import Flask, request

from apps.common import w2 as L

STAGES = {"read_frame": L.load_frame,
          "preprocess": L.preprocess,
          "retrack": L.redetect,
          "track": L.update_tracks,
          "aggregate": L.summarize_tracks,
          "track_update": L.track_update}
MODELS = {"retrack": L.load_detector}
STAGE = os.environ["STAGE"]  # each Deployment/Service runs one stage
app = Flask(__name__)


@app.post(f"/{STAGE}")
def run():
    return pickle.dumps(STAGES[STAGE](*pickle.loads(request.data)))


if __name__ == "__main__":
    if STAGE in MODELS:
        MODELS[STAGE]()  # load before the port opens, so the readiness probe gates traffic
    # a model replica serves one call at a time, keeping its native threads within its CPU request
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8080")), threaded=STAGE not in MODELS)
