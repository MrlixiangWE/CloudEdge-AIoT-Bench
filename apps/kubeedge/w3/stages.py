import os
import pickle

from flask import Flask, request

from apps.common import w3 as L

STAGES = {"load_image": L.load_image,
          "preprocess": L.preprocess_input,
          "infer": L.frcnn_infer,
          "postprocess": L.postprocess,
          "aggregate": L.count_classes}
MODELS = {"infer": L.load_detector}
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
