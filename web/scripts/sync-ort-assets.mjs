// Copies onnxruntime-web's WASM/JSEP artifacts into public/ort/ so the app
// self-hosts them instead of pulling from a CDN. Self-hosting pins the runtime
// to the exact version in package-lock.json and keeps the page CSP-clean.
import { cp, mkdir, readdir } from "node:fs/promises";
import { existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const src = join(root, "node_modules", "onnxruntime-web", "dist");
const dest = join(root, "public", "ort");

if (!existsSync(src)) {
  console.warn("[sync-ort] onnxruntime-web not installed yet; skipping.");
  process.exit(0);
}

await mkdir(dest, { recursive: true });
// The default onnxruntime-web bundle registers the cpu/wasm/webgpu/webnn EPs and
// loads ort-wasm-simd-threaded.jsep.* for all of them. The asyncify/jspi variants
// are for other bundles we do not import, so shipping them would waste ~42MB.
const wanted = (f) => /^ort-wasm-simd-threaded\.jsep\.(wasm|mjs)$/.test(f);
const files = (await readdir(src)).filter(wanted);
for (const f of files) await cp(join(src, f), join(dest, f));
console.log(`[sync-ort] copied ${files.length} runtime files -> public/ort/`);
