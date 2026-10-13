"""W1 Recognition: appliance-state recognition from a household current trace (UK-DALE)."""
import numpy as np

from apps.common import assets as A

APPLIANCES = ["Lamp1", "Lamp2", "Humidifier", "Television", "AirCleaner", "Roomba", "MicrowaveOven",
              "Refrigerator", "Monitor", "Touchpad1", "Charger", "Laptop", "Display", "Touchpad2", "PC"]


@A.once
def current_windows():
    return np.load(A.ASSETS / "w1" / "TotalCurrent.npy", mmap_mode="r")


def _recognizer_arch():
    nn = A.torch().nn

    class Recognizer(nn.Module):
        """2-layer LSTM over a 320-sample window, on/off logits for 15 appliances."""

        def __init__(self, n_apps=15, steps=20, feat=16, hidden=64):
            super().__init__()
            self.steps, self.feat = steps, feat
            self.lstm = nn.LSTM(feat, hidden, num_layers=2, batch_first=True)
            self.head = nn.Linear(hidden, n_apps * 2)

        def forward(self, x):
            h, _ = self.lstm(x.view(x.shape[0], self.steps, self.feat))
            return self.head(h[:, -1]).view(x.shape[0], -1, 2)

    return Recognizer()


@A.once
def load_recognizer():
    return A.seeded(_recognizer_arch)


def decode_power_trace(signal):
    return np.frombuffer(signal, dtype=np.int32).astype(np.float32)


def normalize_window(trace):
    x = (trace - trace.mean()) / (trace.std() + 1e-6)
    return np.convolve(x, np.full(5, 0.2, np.float32), mode="same").astype(np.float32)


def lstm_infer(features, model=None):
    t = A.torch()
    x = t.from_numpy(np.ascontiguousarray(features)).view(1, -1)
    with t.no_grad():
        return (model or load_recognizer())(x).argmax(-1).numpy()[0]


def majority_vote(label):
    return sorted(APPLIANCES[i] for i, on in enumerate(label) if on)


def inputs(n=8):
    w = current_windows()
    return [(f"w1-{i}", np.asarray(w[i % len(w)], dtype=np.int32).tobytes()) for i in range(n)]
