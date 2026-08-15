#!/usr/bin/env bash
# Build + run the all-in-one alienware analyzer (Ollama + model + FastAPI).
# Requires the NVIDIA container toolkit for GPU offload.
set -euo pipefail
cd "$(dirname "$0")"

if ! docker info >/dev/null 2>&1; then
    echo "Docker not running" >&2
    exit 1
fi

docker build -f Dockerfile.allinone -t alien-analyzer-allinone:local .

docker rm -f alien-analyzer 2>/dev/null || true
docker run -d --gpus all \
    --name alien-analyzer \
    -p 31027:8000 \
    -p 11434:11434 \
    -v "$PWD/shared_volume:/data/shared" \
    -e OLLAMA_HOST=http://localhost:11434 \
    alien-analyzer-allinone:local

echo "Waiting for health..."
for i in $(seq 1 30); do
    if curl -fsS localhost:31027/health >/dev/null 2>&1; then
        curl -fsS localhost:31027/health
        echo
        echo "Ready: POST /analyze on :31027"
        exit 0
    fi
    sleep 2
done
echo "Timed out waiting for :31027; check: docker logs alien-analyzer" >&2
exit 1