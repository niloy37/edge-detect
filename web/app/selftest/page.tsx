"use client";

import { useEffect, useState } from "react";
import * as ort from "onnxruntime-web";

import { decodeYolo } from "@/lib/decode";
import { Preprocessor } from "@/lib/letterbox";
import { nonMaxSuppression } from "@/lib/nms";
import { MODELS, getVariant } from "@/lib/models";
import { DEFAULT_DETECT_OPTIONS } from "@/lib/types";

/**
 * Pipeline self-test: runs preprocess -> session -> decode -> NMS on the main thread
 * against a known image, and dumps the intermediate tensor statistics.
 *
 * It exists because "the demo shows no boxes" has at least four possible causes
 * (preprocessing, the model, the decode, the worker plumbing) and they are not
 * distinguishable from the outside. Reusing the exact same lib/ modules the worker
 * uses means a green result here narrows the fault to the plumbing.
 */
export default function SelfTestPage() {
  const [log, setLog] = useState<string[]>([]);

  useEffect(() => {
    let cancelled = false;
    const lines: string[] = [];
    const say = (line: string) => {
      lines.push(line);
      if (!cancelled) setLog([...lines]);
    };

    (async () => {
      try {
        ort.env.wasm.wasmPaths = "/ort/";
        ort.env.wasm.numThreads = 1;

        const model = MODELS[0];
        const variant = getVariant(model, "int8") ?? model.variants[0];
        say(`model: ${model.id} ${variant.precision} ${variant.file}`);

        const session = await ort.InferenceSession.create(variant.file, {
          executionProviders: ["wasm"],
          graphOptimizationLevel: "all",
        });
        say(`session ok. inputs=${session.inputNames} outputs=${session.outputNames}`);

        // Not HTMLImageElement.decode(): it never settles in a backgrounded tab.
        const response = await fetch(model.sample?.image ?? "/samples/coco80-n.jpg");
        const bitmap = await createImageBitmap(await response.blob());
        say(`image ${bitmap.width}x${bitmap.height}`);
        const pre = new Preprocessor(640);
        const { data, transform } = pre.run(bitmap);
        let dmin = Infinity;
        let dmax = -Infinity;
        let dsum = 0;
        for (let i = 0; i < data.length; i++) {
          if (data[i] < dmin) dmin = data[i];
          if (data[i] > dmax) dmax = data[i];
          dsum += data[i];
        }
        say(
          `tensor len=${data.length} min=${dmin.toFixed(3)} max=${dmax.toFixed(3)} mean=${(dsum / data.length).toFixed(3)}`,
        );
        say(
          `transform scale=${transform.scale.toFixed(4)} padX=${transform.padX} padY=${transform.padY}`,
        );

        const tensor = new ort.Tensor("float32", data, [1, 3, 640, 640]);
        const out = await session.run({ [session.inputNames[0]]: tensor });
        const raw = out[session.outputNames[0]];
        const values = raw.data as Float32Array;
        say(`output dims=[${raw.dims.join(",")}] type=${raw.type} len=${values.length}`);

        const [, channels, anchors] = raw.dims as [number, number, number];
        let boxMin = Infinity;
        let boxMax = -Infinity;
        for (let i = 0; i < 4 * anchors; i++) {
          if (values[i] < boxMin) boxMin = values[i];
          if (values[i] > boxMax) boxMax = values[i];
        }
        let scoreMax = 0;
        for (let i = 4 * anchors; i < channels * anchors; i++) {
          if (values[i] > scoreMax) scoreMax = values[i];
        }
        say(`boxes[min=${boxMin.toFixed(2)} max=${boxMax.toFixed(2)}] scoreMax=${scoreMax.toFixed(5)}`);
        say(
          `inFP=${data[0].toFixed(4)},${data[400000].toFixed(4)},${data[1228799].toFixed(4)}` +
            ` | outType=${raw.type} ctor=${values.constructor.name} len=${values.length}` +
            ` | box0=${values[0].toFixed(2)},${values[1].toFixed(2)} score0=${values[4 * anchors].toFixed(4)},${values[4 * anchors + 1].toFixed(4)}`,
        );

        // Repeat the run: a single successful inference proves the pipeline decodes
        // correctly, but says nothing about whether the session survives being used
        // again -- which is exactly where the worker was stalling.
        for (let iteration = 1; iteration <= 5; iteration++) {
          const started = performance.now();
          const again = await session.run({ [session.inputNames[0]]: tensor });
          const rawAgain = again[session.outputNames[0]];
          const dets = nonMaxSuppression(
            decodeYolo(
              rawAgain.data as Float32Array,
              channels,
              anchors,
              transform,
              model.classes,
              DEFAULT_DETECT_OPTIONS,
            ),
            DEFAULT_DETECT_OPTIONS,
          );
          say(
            `  repeat ${iteration}: ${(performance.now() - started).toFixed(0)}ms, ${dets.length} detections`,
          );
        }

        const candidates = decodeYolo(
          values,
          channels,
          anchors,
          transform,
          model.classes,
          DEFAULT_DETECT_OPTIONS,
        );
        say(`candidates above ${DEFAULT_DETECT_OPTIONS.scoreThreshold}: ${candidates.length}`);

        const detections = nonMaxSuppression(candidates, DEFAULT_DETECT_OPTIONS);
        say(`after NMS: ${detections.length}`);
        for (const d of detections) {
          say(
            `  ${d.label} ${(d.score * 100).toFixed(0)}%  x=${d.x.toFixed(0)} y=${d.y.toFixed(0)} w=${d.w.toFixed(0)} h=${d.h.toFixed(0)}`,
          );
        }
        const mainThreadOk = detections.length > 0;

        // --- worker path -------------------------------------------------------
        // Drives the real worker directly, without the render loop. requestAnimationFrame
        // does not fire in a backgrounded tab, so an rAF-driven check silently proves
        // nothing there; this pushes frames by hand and is therefore honest anywhere.
        say("");
        say("worker path:");
        const workerDetections = await new Promise<number[]>((resolve) => {
          const results: number[] = [];
          const worker = new Worker(
            new URL("../../workers/detector.worker.ts", import.meta.url),
            { type: "module" },
          );
          const finish = () => {
            worker.postMessage({ type: "dispose" });
            worker.terminate();
            resolve(results);
          };
          const timer = setTimeout(() => {
            say("  worker timed out");
            finish();
          }, 45000);

          let sent = 0;
          const FRAMES = 3;
          const pushFrame = async () => {
            const clone = await createImageBitmap(bitmap);
            worker.postMessage(
              {
                type: "frame",
                bitmap: clone,
                seq: sent++,
                options: DEFAULT_DETECT_OPTIONS,
              },
              [clone],
            );
          };

          worker.onerror = (event) => {
            say(`  worker error: ${event.message}`);
            clearTimeout(timer);
            finish();
          };
          worker.onmessage = (event: MessageEvent<{ type: string; [k: string]: unknown }>) => {
            const message = event.data;
            if (message.type === "ready") {
              say(`  ready: ep=${message.ep} threads=${message.threads} warmup=${Math.round(message.warmupMs as number)}ms`);
              void pushFrame();
            } else if (message.type === "result") {
              const timing = message.timing as { inference: number; total: number };
              const count = (message.detections as unknown[]).length;
              results.push(count);
              say(`  frame ${message.seq}: ${count} detections, inference ${timing.inference.toFixed(0)}ms`);
              if (results.length >= FRAMES) {
                clearTimeout(timer);
                finish();
              } else {
                void pushFrame();
              }
            } else if (message.type === "error") {
              say(`  worker reported: ${message.message}`);
              clearTimeout(timer);
              finish();
            }
          };

          worker.postMessage({
            type: "init",
            modelUrl: variant.file,
            ep: "wasm",
            inputSize: 640,
            labels: model.classes,
            threads: navigator.hardwareConcurrency > 4 ? 4 : 1,
          });
        });

        const workerOk =
          workerDetections.length === 3 && workerDetections.every((n) => n === detections.length);
        say("");
        say(`main thread: ${detections.length} detections`);
        say(`worker:      [${workerDetections.join(", ")}]`);
        say(
          mainThreadOk && workerOk
            ? "RESULT: PASS (worker agrees with main thread on every frame)"
            : `RESULT: FAIL (${!mainThreadOk ? "main thread found nothing" : "worker disagrees with main thread"})`,
        );
      } catch (error) {
        say(`THREW: ${error instanceof Error ? error.stack ?? error.message : String(error)}`);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <main className="p-6">
      <h1 className="mb-3 font-mono text-sm text-neutral-200">pipeline self-test</h1>
      <pre
        id="selftest-output"
        className="whitespace-pre-wrap font-mono text-xs leading-relaxed text-neutral-300"
      >
        {log.join("\n")}
      </pre>
    </main>
  );
}
