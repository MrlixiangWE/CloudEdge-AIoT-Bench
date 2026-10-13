import os
import pickle

from flask import Flask, request

from apps.common import w4 as L

STAGES = {"preproc_frame": L.preprocess_frame,
          "detect_objects": L.detect,
          "preproc_imu": L.filter_and_normalize,
          "fuse": L.fusion_model,
          "aggregate": L.aggregate_decision}
MODELS = {"detect_objects": L.load_detector,
          "fuse": L.load_fusion}
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
