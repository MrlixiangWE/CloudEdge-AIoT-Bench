import os
import pickle

from flask import Flask, request

from apps.common import w2 as L

STAGES = {"read_frame": L.load_frame,
          "preprocess": L.preprocess,
          "track": L.track,
          "retrack": L.retrack,
          "track_update": L.track_update,
          "aggregate": L.summarize_tracks}
MODELS = {"retrack": L.load_detector}
STAGE = os.environ["STAGE"]  # each Deployment/Service runs one stage
app = Flask(__name__)


@app.post(f"/{STAGE}")
def run():
    return pickle.dumps(STAGES[STAGE](*pickle.loads(request.data)))


if __name__ == "__main__":
    if STAGE in MODELS:
        MODELS[STAGE]()  # load before the port opens, so the queue-proxy probe gates traffic
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8080")), threaded=True)
