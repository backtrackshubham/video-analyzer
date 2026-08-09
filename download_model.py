"""Download an HF repo to a local dir, with mirror + Xet fallbacks."""
import os
import sys

from huggingface_hub import snapshot_download

ENDPOINTS = [
    os.environ.get("HF_ENDPOINT", "https://huggingface.co"),
    "https://hf-mirror.com",
]


def main():
    repo = os.environ["HF_REPO"]
    target = os.environ["HF_TARGET"]
    for endpoint in ENDPOINTS:
        os.environ["HF_ENDPOINT"] = endpoint
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        print(f"[download] endpoint={endpoint} repo={repo}", flush=True)
        try:
            out = snapshot_download(repo_id=repo, local_dir=target)
        except Exception as e:
            print(f"[download] FAILED via {endpoint}: {type(e).__name__}: {e}", flush=True)
            continue
        print(f"[download] OK -> {out}", flush=True)
        return 0
    print("[download] All endpoints failed.", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
