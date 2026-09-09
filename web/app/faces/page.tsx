import { FacesClient } from "@/components/FacesClient";
import { SiteHeader } from "@/components/SiteHeader";

export const metadata = {
  title: "Faces — edge-detect",
  description:
    "On-device face detection with a privacy blur, running in the browser through ONNX Runtime Web. Detection only: where faces are, not whose they are.",
};

export default function FacesPage() {
  return (
    <>
      <SiteHeader active="/faces" />
      <main className="mx-auto max-w-6xl px-4 py-6">
        <div className="mb-5 max-w-2xl">
          <h1 className="text-xl font-semibold tracking-tight text-neutral-100">
            Find faces, then hide them — without uploading anything
          </h1>
          <p className="mt-1.5 text-sm leading-relaxed text-neutral-400">
            The same worker, decoder and NMS as the object detector, running different weights:
            YOLO11n fine-tuned on Open Images &ldquo;Human face&rdquo; by the scripts in{" "}
            <code className="text-neutral-300">ml/</code>. Retargeting the pipeline to a new domain
            is a config file, which is the claim the rest of this site is making.
          </p>
        </div>

        <FacesClient />

        <p className="mt-6 max-w-3xl text-xs leading-relaxed text-neutral-600">
          Trained on Open Images rather than WIDER FACE, the usual choice, because WIDER FACE ships
          under CC-BY-NC-ND and a trained model is arguably a derivative work. Open Images
          annotations are CC BY 4.0 and its images CC BY 2.0, so these weights can be published.
        </p>
      </main>
    </>
  );
}
