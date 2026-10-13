# docker build --target <base|onnx> -t aiot-bench/w2-<target> .
FROM python:3.12-slim AS base
WORKDIR /app
RUN pip install --no-cache-dir numpy pillow flask
COPY apps/ apps/
COPY assets/pics40.npz assets/
ENV PORT=8080
EXPOSE 8080
CMD ["python", "-m", "apps.knative.w2.stages"]

FROM base AS onnx
RUN pip install --no-cache-dir onnxruntime
COPY assets/yoloxna-320.onnx assets/
