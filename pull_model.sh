#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-qwen2.5vl:3b}"

echo "Waiting for the ollama container to be ready..."
for i in $(seq 1 60); do
    if docker exec ollama ollama list >/dev/null 2>&1; then
        break
    fi
    sleep 2
    if [ "$i" -eq 60 ]; then
        echo "Error: ollama container not ready after 120s." >&2
        exit 1
    fi
done

echo "Pulling model: $MODEL"
docker exec -it ollama ollama pull "$MODEL"

echo "Done. Model list:"
docker exec ollama ollama list