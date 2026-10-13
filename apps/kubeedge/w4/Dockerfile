# docker build --target <base|onnx|torch> -t aiot-bench/w4-<target> .
FROM python:3.12-slim AS base
WORKDIR /app
RUN pip install --no-cache-dir numpy pillow flask
COPY apps/ apps/
ENV PORT=8080
EXPOSE 8080
CMD ["python", "-m", "apps.kubeedge.w4.stages"]

FROM base AS onnx
RUN pip install --no-cache-dir onnxruntime
COPY assets/yoloxna-320.onnx assets/

FROM base AS torch
RUN pip install --no-cache-dir torch
