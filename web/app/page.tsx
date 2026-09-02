import { DemoClient } from "@/components/DemoClient";
import { SiteHeader } from "@/components/SiteHeader";
import { MANIFEST } from "@/lib/models";

export default function Page() {
  const modelCount = MANIFEST.models.filter((m) => m.status === "ready").length;

  return (
    <>
      <SiteHeader active="/" />
      <main className="mx-auto max-w-6xl px-4 py-6">
        <div className="mb-5 max-w-2xl">
          <h1 className="text-xl font-semibold tracking-tight text-neutral-100">
            Real-time object detection, running on your device
          </h1>
          <p className="mt-1.5 text-sm leading-relaxed text-neutral-400">
            A YOLO11 detector quantized to INT8 and executed in this tab through ONNX Runtime Web —
            WebGPU where the browser supports it, multi-threaded WASM everywhere else. No frames
            leave your machine, and there is no inference server to pay for. The panel on the right
            reports what it actually costs, per stage, per frame.
          </p>
        </div>

        <DemoClient />

        <p className="mt-6 max-w-3xl text-xs leading-relaxed text-neutral-600">
          {modelCount} model{modelCount === 1 ? "" : "s"} available. Weights, quantization
          calibration and every published number are produced by the scripts in{" "}
          <code className="text-neutral-500">ml/</code> and recorded in a generated manifest, so
          nothing quoted here is hand-written.
        </p>
      </main>
    </>
  );
}
