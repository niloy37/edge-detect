import { SiteHeader } from "@/components/SiteHeader";
import { BENCH, providerLabel, okRuns, type BenchRun } from "@/lib/bench";

export const metadata = {
  title: "Benchmarks — edge-detect",
  description:
    "Measured latency for the same ONNX detector across CPU, CUDA and TensorRT execution providers, with the methodology and the failures.",
};

function StatusPill({ run }: { run: BenchRun }) {
  if (run.status === "ok") return null;
  const tone =
    run.status === "unavailable"
      ? "border-amber-900 bg-amber-950/40 text-amber-300"
      : "border-red-900 bg-red-950/40 text-red-300";
  return <span className={`rounded border px-1.5 py-0.5 text-[10px] uppercase ${tone}`}>{run.status}</span>;
}

export default function BenchmarksPage() {
  const runs = BENCH.runs;
  const successes = okRuns(runs);
  const slowest = Math.max(...successes.map((r) => r.p50Ms ?? 0), 1);
  const failures = runs.filter((r) => r.status !== "ok");

  return (
    <>
      <SiteHeader active="/benchmarks" />
      <main className="mx-auto max-w-6xl space-y-8 px-4 py-8">
        <header className="max-w-2xl">
          <h1 className="text-xl font-semibold tracking-tight text-neutral-100">Benchmarks</h1>
          <p className="mt-1.5 text-sm leading-relaxed text-neutral-400">
            One ONNX file, every execution provider this machine offers. Latency is{" "}
            <code className="text-neutral-300">session.run()</code> alone — preprocessing and NMS are
            reported separately by the live demo&rsquo;s HUD, because folding them in is how vendors
            quote numbers you cannot reproduce.
          </p>
        </header>

        {successes.length === 0 ? (
          <p className="rounded-lg border border-amber-900 bg-amber-950/30 p-4 text-sm text-amber-200">
            No successful runs recorded yet. Run <code>python ml/bench.py</code> to populate this page.
          </p>
        ) : (
          <section>
            <h2 className="mb-3 text-xs font-semibold uppercase tracking-wider text-neutral-400">
              Latency, batch size 1
            </h2>
            <div className="overflow-x-auto rounded-lg border border-neutral-800">
              <table className="w-full min-w-[640px] border-collapse text-sm">
                <thead>
                  <tr className="border-b border-neutral-800 bg-neutral-950 text-left">
                    <th className="px-3 py-2 text-xs font-medium text-neutral-400">Provider</th>
                    <th className="px-3 py-2 text-xs font-medium text-neutral-400">Precision</th>
                    <th className="px-3 py-2 text-xs font-medium text-neutral-400">Input</th>
                    <th className="px-3 py-2 text-right text-xs font-medium text-neutral-400">p50 ms</th>
                    <th className="px-3 py-2 text-right text-xs font-medium text-neutral-400">p95 ms</th>
                    <th className="px-3 py-2 text-right text-xs font-medium text-neutral-400">fps</th>
                    <th className="px-3 py-2 text-xs font-medium text-neutral-400">relative</th>
                  </tr>
                </thead>
                <tbody>
                  {successes
                    .slice()
                    .sort((a, b) => (a.p50Ms ?? 0) - (b.p50Ms ?? 0))
                    .map((run, index) => (
                      <tr key={index} className="border-b border-neutral-900 last:border-0">
                        <td className="px-3 py-2 font-medium text-neutral-200">
                          {providerLabel(run.provider)}
                        </td>
                        <td className="px-3 py-2 font-mono text-xs text-neutral-400">{run.precision}</td>
                        <td className="px-3 py-2 font-mono text-xs text-neutral-400">
                          {run.inputSize}
                        </td>
                        <td className="px-3 py-2 text-right font-mono tabular-nums text-neutral-100">
                          {run.p50Ms?.toFixed(2)}
                        </td>
                        <td className="px-3 py-2 text-right font-mono tabular-nums text-neutral-400">
                          {run.p95Ms?.toFixed(2)}
                        </td>
                        <td className="px-3 py-2 text-right font-mono tabular-nums text-neutral-100">
                          {run.fps?.toFixed(1)}
                        </td>
                        <td className="w-40 px-3 py-2">
                          <div className="h-1.5 w-full overflow-hidden rounded-full bg-neutral-900">
                            <div
                              className="h-full rounded-full bg-violet-500"
                              style={{ width: `${(((run.p50Ms ?? 0) / slowest) * 100).toFixed(1)}%` }}
                            />
                          </div>
                        </td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
          </section>
        )}

        {failures.length > 0 ? (
          <section>
            <h2 className="mb-1 text-xs font-semibold uppercase tracking-wider text-neutral-400">
              Providers that did not run
            </h2>
            <p className="mb-3 max-w-2xl text-xs leading-relaxed text-neutral-500">
              Listed rather than hidden. ONNX Runtime falls back to CPU without raising when a
              provider fails to initialise, so a benchmark harness that does not assert on the{" "}
              <em>active</em> provider will happily publish CPU timings under a TensorRT heading.
            </p>
            <div className="space-y-2">
              {failures.map((run, index) => (
                <div
                  key={index}
                  className="rounded-lg border border-neutral-800 bg-neutral-950/60 p-3"
                >
                  <div className="mb-1 flex flex-wrap items-center gap-2">
                    <span className="text-sm font-medium text-neutral-200">
                      {providerLabel(run.provider)}
                    </span>
                    <span className="font-mono text-[10px] text-neutral-500">
                      {run.precision} @{run.inputSize}
                    </span>
                    <StatusPill run={run} />
                  </div>
                  <p className="font-mono text-[11px] leading-relaxed text-neutral-500">{run.error}</p>
                </div>
              ))}
            </div>
          </section>
        ) : null}

        <section className="grid gap-4 md:grid-cols-2">
          <div className="rounded-lg border border-neutral-800 bg-neutral-950/60 p-4">
            <h2 className="mb-2 text-xs font-semibold uppercase tracking-wider text-neutral-400">
              Methodology
            </h2>
            <dl className="space-y-1 text-xs text-neutral-400">
              <div className="flex justify-between gap-4">
                <dt>Batch size</dt>
                <dd className="font-mono text-neutral-200">{BENCH.methodology.batchSize}</dd>
              </div>
              <div className="flex justify-between gap-4">
                <dt>Warm-up iterations (discarded)</dt>
                <dd className="font-mono text-neutral-200">{BENCH.methodology.warmupIterations}</dd>
              </div>
              <div className="flex justify-between gap-4">
                <dt>Timed iterations</dt>
                <dd className="font-mono text-neutral-200">{BENCH.methodology.timedIterations}</dd>
              </div>
            </dl>
            <p className="mt-2 text-[11px] leading-relaxed text-neutral-500">
              {BENCH.methodology.note}
            </p>
          </div>

          <div className="rounded-lg border border-neutral-800 bg-neutral-950/60 p-4">
            <h2 className="mb-2 text-xs font-semibold uppercase tracking-wider text-neutral-400">
              Machine
            </h2>
            <dl className="space-y-1 text-xs text-neutral-400">
              {Object.entries(BENCH.environment)
                .filter(([, value]) => value && !Array.isArray(value))
                .map(([key, value]) => (
                  <div key={key} className="flex justify-between gap-4">
                    <dt className="shrink-0">{key}</dt>
                    <dd className="truncate font-mono text-neutral-200" title={String(value)}>
                      {String(value)}
                    </dd>
                  </div>
                ))}
            </dl>
            <p className="mt-2 font-mono text-[10px] text-neutral-600">
              measured {BENCH.generatedAt}
            </p>
          </div>
        </section>
      </main>
    </>
  );
}
