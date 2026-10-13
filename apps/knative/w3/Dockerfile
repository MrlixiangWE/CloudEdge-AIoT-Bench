# docker build --target <base|torch> -t aiot-bench/w3-<target> .
FROM python:3.12-slim AS base
WORKDIR /app
RUN pip install --no-cache-dir numpy pillow flask
COPY apps/ apps/
COPY assets/pics40.npz assets/
ENV PORT=8080
EXPOSE 8080
CMD ["python", "-m", "apps.knative.w3.stages"]

FROM base AS torch
RUN pip install --no-cache-dir torch torchvision
COPY assets/script_model_cpu.pt assets/
