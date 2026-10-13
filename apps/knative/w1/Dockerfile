# docker build --target <base|torch> -t aiot-bench/w1-<target> .
FROM python:3.12-slim AS base
WORKDIR /app
RUN pip install --no-cache-dir numpy flask
COPY apps/ apps/
ENV PORT=8080
EXPOSE 8080
CMD ["python", "-m", "apps.knative.w1.stages"]

FROM base AS torch
RUN pip install --no-cache-dir torch
