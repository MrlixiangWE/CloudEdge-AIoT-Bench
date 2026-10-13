import os
import pickle

from flask import Flask, request

from apps.common import w1 as L

STAGES = {"load_signal": L.decode_power_trace,
          "preproc": L.normalize_window,
          "recognize": L.lstm_infer,
          "aggregate": L.majority_vote}
MODELS = {"recognize": L.load_recognizer}
STAGE = os.environ["STAGE"]  # each Deployment/Service runs one stage
app = Flask(__name__)


@app.post(f"/{STAGE}")
def run():
    return pickle.dumps(STAGES[STAGE](*pickle.loads(request.data)))


if __name__ == "__main__":
    if STAGE in MODELS:
        MODELS[STAGE]()  # load before the port opens, so the queue-proxy probe gates traffic
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8080")), threaded=True)
