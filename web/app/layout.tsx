import type { Metadata } from "next";

import "./globals.css";

export const metadata: Metadata = {
  title: "edge-detect — real-time object detection in your browser",
  description:
    "A custom-trained YOLO detector quantized to INT8 and run on-device via ONNX Runtime Web (WebGPU/WASM), with the TensorRT and CPU benchmarks that justify the design.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-neutral-950 font-sans antialiased">{children}</body>
    </html>
  );
}
