/** @type {import('next').NextConfig} */
const nextConfig = {
  async headers() {
    return [
      {
        // Cross-origin isolation is what unlocks SharedArrayBuffer, which is what
        // lets ONNX Runtime's WASM backend use more than one thread. Without these
        // two headers the WASM path silently runs single-threaded and looks ~3x
        // slower than the hardware is capable of.
        //
        // COEP is `credentialless` rather than `require-corp` on purpose:
        // require-corp blocks every cross-origin subresource that does not send
        // CORP headers, which breaks embedded media and third-party images for no
        // benefit here.
        source: "/:path*",
        headers: [
          { key: "Cross-Origin-Opener-Policy", value: "same-origin" },
          { key: "Cross-Origin-Embedder-Policy", value: "credentialless" },
        ],
      },
      {
        // The runtime and the weights are content-addressed by version and never
        // mutate in place, so they can be cached hard. This is the difference
        // between a 3MB download on every visit and one.
        source: "/:path(models|ort)/:file*",
        headers: [{ key: "Cache-Control", value: "public, max-age=31536000, immutable" }],
      },
    ];
  },
  async rewrites() {
    return [
      // Python serverless functions live in web/api/*.py. Routing them under
      // /api/py/ keeps them from colliding with Next's own route handlers.
      { source: "/api/py/:path*", destination: "/api/:path*" },
    ];
  },
};

export default nextConfig;
