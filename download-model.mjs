#!/usr/bin/env node
// Download the OpenVINO GenAI VLM checkpoint for the analyzer (Node 18+).
// Big files first, concurrent downloads, per-file timing.
// Env: HF_REPO, HF_TARGET, HF_ENDPOINT, CONCURRENCY
import { createWriteStream } from "node:fs";
import { mkdir, rename } from "node:fs/promises";
import { pipeline } from "node:stream/promises";
import { Readable } from "node:stream";
import path from "node:path";

const REPO = process.env.HF_REPO || "0ldev/Qwen2.5-VL-3B-Instruct-ov-nf4-npu";
const DEST = process.env.HF_TARGET || path.join(process.cwd(), "models", REPO.split("/").pop());
const ENDPOINTS = [process.env.HF_ENDPOINT || "https://huggingface.co", "https://hf-mirror.com"];
const CONCURRENCY = parseInt(process.env.CONCURRENCY || "4", 10);
const RETRIES = 3;

const FILES = [
  "openvino_language_model.bin",
  "openvino_vision_embeddings_merger_model.bin",
  "openvino_text_embeddings_model.bin",
  "tokenizer.json",
  "openvino_tokenizer.bin",
  "openvino_language_model.xml",
  "vocab.json",
  "openvino_detokenizer.bin",
  "openvino_vision_embeddings_merger_model.xml",
  "merges.txt",
  "openvino_vision_embeddings_model.bin",
  "openvino_tokenizer.xml",
  "openvino_detokenizer.xml",
  "openvino_vision_embeddings_model.xml",
  "openvino_text_embeddings_model.xml",
  "tokenizer_config.json",
  "config.json",
  "openvino_config.json",
  "chat_template.jinja",
  "special_tokens_map.json",
  "added_tokens.json",
  "preprocessor_config.json",
  "generation_config.json",
];

async function downloadFile(endpoint, name) {
  const url = `${endpoint}/${REPO}/resolve/main/${name}?download=true`;
  const res = await fetch(url, {
    redirect: "follow",
    headers: { "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36" },
  });
  if (!res.ok) throw new Error(`HTTP ${res.status} for ${url}`);
  const out = path.join(DEST, name);
  const tmp = `${out}.part`;
  await pipeline(Readable.fromWeb(res.body), createWriteStream(tmp));
  await rename(tmp, out);
}

async function fetchFile(name) {
  let lastErr;
  for (const endpoint of ENDPOINTS) {
    for (let i = 1; i <= RETRIES; i++) {
      const t0 = Date.now();
      try {
        await downloadFile(endpoint, name);
        const secs = ((Date.now() - t0) / 1000).toFixed(1);
        console.log(`[ok]  ${name}  (${secs}s) via ${endpoint.replace("https://", "")}`);
        return true;
      } catch (e) {
        lastErr = e;
        console.log(`[try ${i}] ${name} via ${endpoint}: ${e.message}`);
      }
    }
  }
  console.error(`[FAIL] ${name}: ${lastErr.message}`);
  return false;
}

async function main() {
  await mkdir(DEST, { recursive: true });
  console.log(`Dest: ${DEST}  concurrency=${CONCURRENCY}`);
  let done = 0;
  for (let i = 0; i < FILES.length; i += CONCURRENCY) {
    const batch = FILES.slice(i, i + CONCURRENCY);
    const results = await Promise.all(batch.map(fetchFile));
    done += results.filter(Boolean).length;
  }
  console.log(`\nDone: ${done}/${FILES.length} files -> ${DEST}`);
  process.exit(done === FILES.length ? 0 : 1);
}

main();