// Copies the model and class names the Python function needs into api/, next to the
// function itself.
//
// They cannot be referenced from public/ or data/: `includeFiles` takes a single
// string glob, and the brace form silently matched nothing under public/ -- the
// function then fell back to fetching the model over HTTP, got Vercel's access-
// protection login page, and wrote an HTML page to /tmp as a .onnx. Files that sit
// beside the entrypoint in api/ are bundled without any of that ceremony.
import { cp, mkdir, writeFile, readFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const api = join(root, "api");
const MODEL_ID = "coco80-n";

await mkdir(api, { recursive: true });
await cp(
  join(root, "public", "models", `${MODEL_ID}-int8.onnx`),
  join(api, "model-int8.onnx"),
);

const manifest = JSON.parse(await readFile(join(root, "data", "models.json"), "utf-8"));
const entry = manifest.models.find((m) => m.id === MODEL_ID);
if (!entry) throw new Error(`${MODEL_ID} missing from data/models.json`);
await writeFile(
  join(api, "labels.json"),
  JSON.stringify({ id: entry.id, precision: "int8", classes: entry.classes }, null, 2) + "\n",
);
console.log(`[sync-fn] api/model-int8.onnx + api/labels.json for ${MODEL_ID}`);
