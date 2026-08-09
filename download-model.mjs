#!/usr/bin/env node
// Download the OpenVINO GenAI VLM checkpoint for the analyzer (Node 18+).
// Uses undici fetch (node's global fetch). Set HF_REPO/HF_TARGET to override.
import { createWriteStream } from "node:fs";
import { mkdir, rename } from "node:fs/promises";
import { pipeline } from "node:stream/promises";
import { Readable } from "node:stream";
import path from "node:path";

const REPO = process.env.HF_REPO || "0ldev/Qwen2.5-VL-3B-Instruct-ov-nf4-npu";
const DEST = process.env.HF_TARGET || path.join(process.cwd(), "models", REPO.split("/").pop());
const ENDPOINTS = [process.env.HF_ENDPOINT || "https://huggingface.co", "https://hf-mirror.com"];
const RETRIES = 3;

const FILES = [
  "added_tokens.json", "chat_template.jinja", "config.json", "generation_config.json",
  "merges.txt", "openvino_config.json", "openvino_detokenizer.bin", "openvino_detokenizer.xml",
  "openvino_language_model.bin", "openvino_language_model.xml",
  "openvino_text_embeddings_model.bin", "openvino_text_embeddings_model.xml",
  "openvino_tokenizer.bin", "openvino_tokenizer.xml",
  "openvino_vision_embeddings_merger_model.bin", "openvino_vision_embeddings_merger_model.xml",
  "openvino_vision_embeddings_model.bin", "openvino_vision_embeddings_model.xml",
  "preprocessor_config.json", "special_tokens_map.json", "tokenizer.json",
  "tokenizer_config.json", "vocab.json",
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

async function tryDownload(name) {
  let lastErr;
  for (const endpoint of ENDPOINTS) {
    for (let i = 1; i <= RETRIES; i++) {
      try {
        await downloadFile(endpoint, name);
        console.log(`[ok] ${name}`);
        return true;
      } catch (e) {
        lastErr = e;
        console.log(`[try ${i}] ${name} via ${endpoint}: ${e.code || e.message}`);
      }
    }
  }
  console.error(`[FAIL] ${name}: ${lastErr}`);
  return false;
}

async function main() {
  await mkdir(DEST, { recursive: true });
  console.log(`Dest: ${DEST}`);
  let ok = 0;
  for (const f of FILES) if (await tryDownload(f)) ok++;
  console.log(`\nDone: ${ok}/${FILES.length} files -> ${DEST}`);
  process.exit(ok === FILES.length ? 0 : 1);
}

main();