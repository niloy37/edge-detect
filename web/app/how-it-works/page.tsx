import { SiteHeader } from "@/components/SiteHeader";

export const metadata = {
  title: "How it works — edge-detect",
  description:
    "The pipeline behind the demo: letterbox preprocessing, a decode/NMS implementation kept out of the ONNX graph, INT8 quantization and the head-tail bug it caused.",
};

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="space-y-2">
      <h2 className="text-sm font-semibold tracking-tight text-neutral-100">{title}</h2>
      <div className="space-y-3 text-sm leading-relaxed text-neutral-400">{children}</div>
    </section>
  );
}

function Code({ children }: { children: React.ReactNode }) {
  return (
    <pre className="overflow-x-auto rounded-lg border border-neutral-800 bg-neutral-950 p-3 font-mono text-[11px] leading-relaxed text-neutral-300">
      {children}
    </pre>
  );
}

export default function HowItWorksPage() {
  return (
    <>
      <SiteHeader active="/how-it-works" />
      <main className="mx-auto max-w-3xl space-y-8 px-4 py-8">
        <header>
          <h1 className="text-xl font-semibold tracking-tight text-neutral-100">How it works</h1>
          <p className="mt-1.5 text-sm leading-relaxed text-neutral-400">
            The interesting parts of this project are not the model. They are the four or five
            decisions between a checkpoint and something that runs at speed in a stranger&rsquo;s
            browser.
          </p>
        </header>

        <Section title="The pipeline">
          <Code>{`camera frame (1280x720)
  |
  |  createImageBitmap  -- snapshot off the main thread, transferable
  v
Web Worker
  |  letterbox to NxN, pad 114, RGBA -> CHW float32 /255   [lib/letterbox.ts]
  |  session.run()  via WebGPU or multi-threaded WASM      [onnxruntime-web]
  |  decode [1, 84, anchors] -> candidates                 [lib/decode.ts]
  |  class-wise NMS                                        [lib/nms.ts]
  v
main thread: draw last known boxes every animation frame   [DetectorStage]`}</Code>
          <p>
            The render loop and the detection loop are independent. Every animation frame redraws
            the video and the most recent boxes; frames are only handed to the model when it is
            idle, and dropped otherwise. Coupling the two — drawing only when a detection returns —
            is why so many browser demos play back like a slideshow even though the camera feed
            itself is smooth.
          </p>
        </Section>

        <Section title="Why NMS is not in the graph">
          <p>
            Ultralytics can export with non-maximum suppression baked in. This project exports with{" "}
            <code className="text-neutral-300">nms=False</code> and implements decode and NMS in
            TypeScript instead.
          </p>
          <p>
            An in-graph <code className="text-neutral-300">NonMaxSuppression</code> node is not
            supported by the WebGPU execution provider, so ONNX Runtime partitions the model and
            falls back to CPU for the tail — which means a GPU-to-CPU sync on every single frame.
            Keeping the exported graph to pure tensor operations lets it stay GPU-resident. The cost
            is that the decode and NMS become ours to get right, which is what{" "}
            <code className="text-neutral-300">ml/parity_test.py</code> is for: it pins the
            TypeScript implementation against the numpy one on fixed inputs.
          </p>
        </Section>

        <Section title="The INT8 bug worth reading about">
          <p>
            Static INT8 quantization succeeded, produced a model 3x smaller, ran without error — and
            detected absolutely nothing. No exception, no warning.
          </p>
          <p>The graph ends like this:</p>
          <Code>{`Concat_26( Mul_5    -> box xywh, pixel units, range 0..640
           Sigmoid  -> class scores,        range 0..1  ) -> output0`}</Code>
          <p>
            QDQ quantization assigns <em>one</em> scale per tensor, and a Concat output is one
            tensor. Calibrated over the union range [0, 640] that scale is roughly 2.5 — so every
            class score, all of which are below 1.0, rounds to exactly zero. The box channels are
            large enough to survive, which is what makes the failure so quiet: the model still emits
            perfectly plausible geometry.
          </p>
          <Code>{`fp32  boxes[min=2.57 max=637.18]  scores[max=0.93992]  -> 5 detections
int8  boxes[min=0.00 max=640.05]  scores[max=0.00000]  -> 0 detections`}</Code>
          <p>
            The fix is to leave the detection head&rsquo;s elementwise tail in float32 — the
            sigmoid, the DFL softmax and projection, the box arithmetic, and that final concat —
            while all 88 backbone and neck convolutions stay INT8. Those convolutions are where the
            time actually goes, so the speedup survives the fix. Afterwards both precisions find the
            same objects on the same image, with scores within 0.02.
          </p>
        </Section>

        <Section title="Cross-origin isolation">
          <p>
            ONNX Runtime&rsquo;s WASM backend can use multiple threads, but only through{" "}
            <code className="text-neutral-300">SharedArrayBuffer</code>, which browsers gate behind
            cross-origin isolation. The deployed build sends:
          </p>
          <Code>{`Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: credentialless`}</Code>
          <p>
            <code className="text-neutral-300">credentialless</code> rather than{" "}
            <code className="text-neutral-300">require-corp</code>: the stricter value blocks every
            cross-origin subresource that does not opt in with CORP headers, which breaks embedded
            media for no benefit here. Without isolation the runtime silently drops to a single
            thread and the WASM numbers look about three times worse than the machine can do — so
            the demo shows the isolation state in the badge row rather than leaving it hidden.
          </p>
        </Section>

        <Section title="One model file, three resolutions">
          <p>
            The ONNX is exported with dynamic spatial dimensions, so a single file serves 320, 480
            and 640 inference and the resolution control is a real latency/accuracy trade-off rather
            than three downloads. The trade-off is that the first inference at a new size re-pays
            shape-dependent setup — a WebGPU shader recompile — which shows up as a one-off spike in
            the p95 immediately after switching.
          </p>
          <p>
            It also broke quantization tooling: ONNX Runtime&rsquo;s symbolic shape inference cannot
            resolve dynamic H/W and raises{" "}
            <code className="text-neutral-300">Incomplete symbolic shape inference</code>, so the
            quantizer falls back to the optimizer-only preprocessing path.
          </p>
        </Section>

        <Section title="The square tensor was wasting 44% of every frame">
          <p>
            The letterbox fits the frame into a square and pads the remainder grey. A 16:9 camera
            frame therefore fills 640×360 of a 640×640 tensor — and the model pays for all 409,600
            pixels, including the 179,200 that are only padding.
          </p>
          <Code>{`640x640  24.54ms   8400 anchors   100% of the tensor
384x640  17.48ms   5040 anchors    60%   (16:9)
480x640  20.36ms   6300 anchors    75%   (4:3)`}</Code>
          <p>
            The exported graph has dynamic height and width, so matching the input to the source
            aspect needed no second model — only a stride-aligned shape computed per frame. It is
            about 29% cheaper on a 16:9 source, and it is also <em>more accurate</em>: with less of
            the tensor spent on padding the subject survives at a larger scale. On the sample image
            the square input found 4 objects and the aspect-matched input found 5, with every score
            higher — the bus went 92% to 96%, and the marginal person 45% to 65%.
          </p>
          <p>
            The one cost is that changing shape re-pays shape-dependent setup in the runtime, so
            pointing the demo at a differently-shaped source produces a one-off spike in the p95.
          </p>
        </Section>

        <Section title="Jitter is not an accuracy problem">
          <p>
            Boxes shimmered around motionless objects. That reads like model uncertainty, but every
            frame is decoded completely independently — nothing carries between them, so
            near-identical input produces slightly different output forever. No amount of mAP fixes
            it, because it is not a measurement error; it is the absence of any temporal model.
          </p>
          <p>
            Detections are now matched across frames by IoU within a class, and matched coordinates
            are exponentially smoothed. A track may coast for a single frame, which removes the
            blink that happens when confidence dips just under the threshold. It is deliberately
            not a Kalman filter — there is no motion model and no prediction, because at four
            frames a second a wrong guess about where a box is heading looks far worse than a box
            that lags slightly behind. Toggle it in the panel and watch a still object.
          </p>
        </Section>

        <Section title="What the benchmarks actually said">
          <p>
            Three results changed decisions rather than decorating a page. All measured on an RTX
            5070 at batch size 1, 50 warm-up iterations discarded, 300 timed.
          </p>
          <Code>{`FP32  CPU   320px 11.45ms   640px 37.95ms
FP32  CUDA  320px 13.10ms   640px 13.88ms
FP32  TRT   320px  4.27ms   640px  5.29ms   (static shapes: 3.15ms)
INT8  CPU   320px 10.74ms   640px 34.31ms
INT8  CUDA  320px 31.75ms   640px 31.04ms
INT8  TRT   engine build rejected -- see Benchmarks`}</Code>
          <p>
            <strong>CUDA barely notices the difference between 320 and 640</strong> (13.10 →
            13.88 ms) while CPU nearly quadruples. A nano model does not saturate this GPU — it is
            bound by kernel launch overhead, not arithmetic. That is also why the CPU is
            <em> faster</em> than CUDA at 320px.
          </p>
          <p>
            <strong>INT8 is 2.3× slower than FP32 under the CUDA provider.</strong> The
            quantize/dequantize nodes are not fused into INT8 tensor-core kernels there, so the
            casts cost real time and buy nothing. INT8 pays off on memory-bound targets — CPU and
            WASM, which is where this demo runs — and not on bare CUDA. Adopting INT8 everywhere on
            the general principle that &ldquo;quantization is faster&rdquo; would have made the GPU
            path 2.3× worse.
          </p>
          <p>
            <strong>Dynamic shapes cost TensorRT about 40%</strong> — 5.29 ms against 3.15 ms for a
            fixed-shape export. That is the price of the resolution control on this page, paid on a
            deployment target this page does not use. Both numbers are published rather than the
            flattering one.
          </p>
        </Section>

        <Section title="Why the live demo is not a server">
          <p>
            There is a serverless endpoint in this project at{" "}
            <code className="text-neutral-300">/api/py/detect</code>, running the same INT8 weights
            through onnxruntime on CPU. It exists to make the comparison concrete, not because it is
            the right way to serve video.
          </p>
          <p>
            Per-frame server inference pays a network round trip and, on a serverless platform, an
            occasional cold start measured in seconds. On-device inference pays neither, costs
            nothing to host, and never transmits the frames at all. For a real-time detector that
            argument is decisive — which is why the video path runs in your tab and the endpoint
            accepts single images.
          </p>
        </Section>
      </main>
    </>
  );
}
