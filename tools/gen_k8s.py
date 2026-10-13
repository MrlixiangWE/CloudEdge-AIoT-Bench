"""Generate the KubeEdge / Knative deployment files of W1-W4.

Each stage keeps the tier, CPU and memory it declares under Hedices (@Stem); edge stages get 4
replicas and cloud stages 2 (KubeEdge replicas, Knative min-scale).  Images are split by native
runtime (base / onnx / torch) so an edge pod pulls only what its stage imports.  Model stages
preload their model before the port opens, bound native threads with OMP_NUM_THREADS and serve
one call at a time (Flask single-threaded under KubeEdge, containerConcurrency under Knative).
  python tools/gen_k8s.py
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "apps"
# stage, tier, cpu, memory, image, model stage
SPEC = {
    "w1": [("load_signal", "edge", 1, "1Gi", "base", False),
           ("preproc", "edge", 2, "1Gi", "base", False),
           ("recognize", "edge", 2, "2Gi", "torch", True),
           ("aggregate", "cloud", 1, "512Mi", "base", False)],
    "w2": [("read_frame", "cloud", 1, "1Gi", "base", False),
           ("preprocess", "edge", 1, "512Mi", "base", False),
           ("retrack", "edge", 2, "1Gi", "onnx", True),
           ("track", "edge", 1, "512Mi", "base", False),
           ("aggregate", "cloud", 1, "512Mi", "base", False),
           ("track_update", "edge", 1, "256Mi", "base", False)],
    "w3": [("load_image", "cloud", 1, "1Gi", "base", False),
           ("preprocess", "cloud", 1, "1Gi", "base", False),
           ("infer", "edge", 2, "2Gi", "torch", True),
           ("postprocess", "cloud", 2, "1Gi", "torch", False),
           ("aggregate", "cloud", 1, "512Mi", "base", False)],
    "w4": [("preproc_frame", "edge", 1, "512Mi", "base", False),
           ("detect_objects", "edge", 2, "2Gi", "onnx", True),
           ("preproc_imu", "edge", 1, "512Mi", "base", False),
           ("fuse", "cloud", 4, "4Gi", "torch", True),
           ("aggregate", "cloud", 1, "512Mi", "base", False)],
}
# pip packages and assets of each image (base also carries the stage server and the app package)
IMAGES = {
    "w1": {"base": (["numpy", "flask"], []), "torch": (["torch"], [])},
    "w2": {"base": (["numpy", "pillow", "flask"], ["pics40.npz"]), "onnx": (["onnxruntime"], ["yoloxna-320.onnx"])},
    "w3": {"base": (["numpy", "pillow", "flask"], ["pics40.npz"]),
           "torch": (["torch", "torchvision"], ["script_model_cpu.pt"])},
    "w4": {"base": (["numpy", "pillow", "flask"], []), "onnx": (["onnxruntime"], ["yoloxna-320.onnx"]),
           "torch": (["torch"], [])},
}
TOLERATIONS = """\
      tolerations:  # leave a failed node after 5 s instead of the default 300 s
      - {key: node.kubernetes.io/unreachable, operator: Exists, effect: NoExecute, tolerationSeconds: 5}
      - {key: node.kubernetes.io/not-ready, operator: Exists, effect: NoExecute, tolerationSeconds: 5}
"""


def replicas(tier):
    return 4 if tier == "edge" else 2


def env(stage, cpu, model, indent):
    lines = [f"{indent}env:", f"{indent}- {{name: STAGE, value: {stage}}}"]
    if model:
        lines.append(f'{indent}- {{name: OMP_NUM_THREADS, value: "{cpu}"}}  # native threads = CPU request')
    return "\n".join(lines) + "\n"


def kubeedge(w):
    docs = []
    for stage, tier, cpu, mem, image, model in SPEC[w]:
        name = f"{w}-{stage.replace('_', '-')}"
        probe = ("        readinessProbe:  # ready once the model is loaded and the port opens\n"
                 "          tcpSocket: {port: 8080}\n"
                 "          periodSeconds: 2\n") if model else ""
        docs.append(f"""\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {name}
  labels: {{app: {name}}}
spec:
  replicas: {replicas(tier)}
  selector:
    matchLabels: {{app: {name}}}
  template:
    metadata:
      labels: {{app: {name}}}
    spec:
      nodeSelector:
        hedices.io/tier: {tier}
{TOLERATIONS}      containers:
      - name: stage
        image: aiot-bench/{w}-{image}:latest
        imagePullPolicy: IfNotPresent
{env(stage, cpu, model, "        ")}        ports:
        - containerPort: 8080
        resources:
          requests: {{cpu: "{cpu}", memory: {mem}}}
{probe}---
apiVersion: v1
kind: Service
metadata:
  name: {name}
spec:
  selector: {{app: {name}}}
  ports:
  - port: 80
    targetPort: 8080
""")
    return "---\n".join(docs)


def knative(w):
    docs = []
    for stage, tier, cpu, mem, image, model in SPEC[w]:
        name = f"{w}-{stage.replace('_', '-')}"
        cc = "      containerConcurrency: 1  # one call at a time per model replica\n" if model else ""
        docs.append(f"""\
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: {name}
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/min-scale: "{replicas(tier)}"
    spec:
{cc}      nodeSelector:
        hedices.io/tier: {tier}
{TOLERATIONS}      containers:
      - image: aiot-bench/{w}-{image}:latest
{env(stage, cpu, model, "        ")}        ports:
        - containerPort: 8080
        resources:
          requests: {{cpu: "{cpu}", memory: {mem}}}
""")
    return "---\n".join(docs)


def dockerfile(fw, w):
    targets = list(IMAGES[w])
    out = [f"# docker build --target <{'|'.join(targets)}> -t aiot-bench/{w}-<target> ."]
    for t in targets:
        pkgs, assets = IMAGES[w][t]
        if t == "base":
            out += ["FROM python:3.12-slim AS base", "WORKDIR /app",
                    f"RUN pip install --no-cache-dir {' '.join(pkgs)}", "COPY apps/ apps/"]
            out += [f"COPY {' '.join('assets/' + a for a in assets)} assets/"] if assets else []
            out += ["ENV PORT=8080", "EXPOSE 8080", f'CMD ["python", "-m", "apps.{fw}.{w}.stages"]']
        else:
            out += ["", f"FROM base AS {t}", f"RUN pip install --no-cache-dir {' '.join(pkgs)}"]
            out += [f"COPY {' '.join('assets/' + a for a in assets)} assets/"] if assets else []
    return "\n".join(out) + "\n"


def main():
    for w in SPEC:
        (ROOT / "kubeedge" / w / "deploy.yaml").write_text(kubeedge(w))
        (ROOT / "knative" / w / "service.yaml").write_text(knative(w))
        for fw in ("kubeedge", "knative"):
            (ROOT / fw / w / "Dockerfile").write_text(dockerfile(fw, w))


if __name__ == "__main__":
    main()
