// TypeScript's lib.dom does not yet declare the WebGPU surface, and lib/ep.ts has to
// touch navigator.gpu directly to probe for a usable adapter.
/// <reference types="@webgpu/types" />
