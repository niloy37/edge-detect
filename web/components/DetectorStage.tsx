"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { drawDetections } from "@/lib/draw";
import { FpsCounter } from "@/lib/telemetry";
import type { DetectOptions, Detection } from "@/lib/types";

export type SourceKind = "webcam" | "sample" | "image";

export interface StageSource {
  kind: SourceKind;
  /** Video URL for "sample", object URL for "image". Ignored for "webcam". */
  url?: string;
}

interface Props {
  source: StageSource;
  ready: boolean;
  paused: boolean;
  options: DetectOptions;
  beginFrame: () => boolean;
  sendFrame: (bitmap: ImageBitmap, options: DetectOptions) => void;
  abortFrame: () => void;
  detectionsRef: React.RefObject<Detection[]>;
  onRenderFps: (fps: number) => void;
  onSourceError: (message: string | null) => void;
  onSourceReady: () => void;
}

/**
 * The display half of the detector.
 *
 * The render loop and the detection loop are deliberately independent: every
 * animation frame redraws the source and the most recent boxes, while frames are
 * only handed to the model when it is idle. Coupling them -- drawing only when a
 * detection returns -- is what makes most browser demos look like a slideshow at
 * 8fps even though the video itself is smooth.
 */
export function DetectorStage({
  source,
  ready,
  paused,
  options,
  beginFrame,
  sendFrame,
  abortFrame,
  detectionsRef,
  onRenderFps,
  onSourceError,
  onSourceReady,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const imageRef = useRef<ImageBitmap | null>(null);
  const rafRef = useRef<number>(0);
  const fpsRef = useRef(new FpsCounter());
  const lastFpsReport = useRef(0);
  const optionsRef = useRef(options);
  const readyRef = useRef(ready);
  const pausedRef = useRef(paused);
  // Callbacks live in refs so that the source-acquisition effect below depends only
  // on the source itself. Its cleanup closes the ImageBitmap the render loop draws,
  // so any spurious re-run tears down a perfectly good source mid-flight.
  const sourceErrorRef = useRef(onSourceError);
  const sourceReadyRef = useRef(onSourceReady);
  const renderFpsRef = useRef(onRenderFps);
  const [dimensions, setDimensions] = useState<{ width: number; height: number } | null>(null);

  optionsRef.current = options;
  readyRef.current = ready;
  pausedRef.current = paused;
  sourceErrorRef.current = onSourceError;
  sourceReadyRef.current = onSourceReady;
  renderFpsRef.current = onRenderFps;

  // --- source acquisition -------------------------------------------------
  useEffect(() => {
    let cancelled = false;
    let stream: MediaStream | null = null;
    sourceErrorRef.current(null);
    setDimensions(null);
    imageRef.current = null;
    videoRef.current = null;

    async function attachWebcam() {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          // Rear camera when there is one: a hiring manager opening this on a phone
          // wants to point it at the room, not at their own face.
          video: { facingMode: { ideal: "environment" }, width: { ideal: 1280 } },
          audio: false,
        });
      } catch (error) {
        if (!cancelled) {
          sourceErrorRef.current(
            error instanceof DOMException && error.name === "NotAllowedError"
              ? "Camera permission denied. Try a sample video instead."
              : `Camera unavailable: ${error instanceof Error ? error.message : String(error)}`,
          );
        }
        return;
      }
      if (cancelled || !stream) return;
      const video = document.createElement("video");
      video.srcObject = stream;
      video.muted = true;
      video.playsInline = true;
      await video.play().catch(() => undefined);
      if (cancelled) return;
      videoRef.current = video;
      setDimensions({ width: video.videoWidth, height: video.videoHeight });
      sourceReadyRef.current();
    }

    async function attachVideo(url: string) {
      const video = document.createElement("video");
      video.src = url;
      video.loop = true;
      video.muted = true;
      video.playsInline = true;
      video.crossOrigin = "anonymous";
      try {
        await new Promise<void>((resolve, reject) => {
          video.onloadeddata = () => resolve();
          video.onerror = () => reject(new Error("could not load sample video"));
        });
        await video.play();
      } catch (error) {
        if (!cancelled) sourceErrorRef.current(error instanceof Error ? error.message : String(error));
        return;
      }
      if (cancelled) return;
      videoRef.current = video;
      setDimensions({ width: video.videoWidth, height: video.videoHeight });
      sourceReadyRef.current();
    }

    async function attachImage(url: string) {
      // Decoded via fetch -> blob -> createImageBitmap rather than an <img> and
      // HTMLImageElement.decode(). decode() does not resolve in a backgrounded Chrome
      // tab -- it simply never settles, with no error -- which stalls the whole demo
      // for anyone who opens the link in a tab that is not in front. createImageBitmap
      // has no such dependency, and it yields exactly the type the worker wants.
      try {
        const response = await fetch(url);
        if (!response.ok) throw new Error(`image request failed (${response.status})`);
        const bitmap = await createImageBitmap(await response.blob());
        if (cancelled) {
          bitmap.close();
          return;
        }
        imageRef.current = bitmap;
        setDimensions({ width: bitmap.width, height: bitmap.height });
        sourceReadyRef.current();
      } catch (error) {
        if (!cancelled) {
          sourceErrorRef.current(error instanceof Error ? error.message : "could not decode that image");
        }
      }
    }

    if (source.kind === "webcam") void attachWebcam();
    else if (source.kind === "sample" && source.url) void attachVideo(source.url);
    else if (source.kind === "image" && source.url) void attachImage(source.url);

    return () => {
      cancelled = true;
      stream?.getTracks().forEach((track) => track.stop());
      const video = videoRef.current;
      if (video) {
        video.pause();
        video.srcObject = null;
        video.removeAttribute("src");
      }
      videoRef.current = null;
      imageRef.current?.close();
      imageRef.current = null;
    };
  }, [source.kind, source.url]);

  // --- render + submit loop -----------------------------------------------
  const loop = useCallback(() => {
    rafRef.current = requestAnimationFrame(loop);

    const canvas = canvasRef.current;
    const media = videoRef.current ?? imageRef.current;
    if (!canvas || !media) return;

    const sourceWidth = videoRef.current?.videoWidth ?? imageRef.current?.width ?? 0;
    const sourceHeight = videoRef.current?.videoHeight ?? imageRef.current?.height ?? 0;
    if (!sourceWidth || !sourceHeight) return;

    if (canvas.width !== sourceWidth || canvas.height !== sourceHeight) {
      canvas.width = sourceWidth;
      canvas.height = sourceHeight;
    }

    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.drawImage(media, 0, 0, sourceWidth, sourceHeight);
    drawDetections(ctx, detectionsRef.current ?? [], 1);

    fpsRef.current.tick();
    // Report at ~5Hz, not once per animation frame. Calling a React setState 200+
    // times a second re-renders the whole panel on every frame just to move a
    // readout, which is a measurable tax on the loop it is measuring.
    const now = performance.now();
    if (now - lastFpsReport.current > 200) {
      lastFpsReport.current = now;
      renderFpsRef.current(fpsRef.current.fps);
    }

    // Claim the model's single in-flight slot *before* starting the capture. The
    // capture is async, so checking availability and then awaiting would let this
    // loop start a new capture on every animation frame while the first is still
    // resolving -- dozens of bitmaps in flight against a model that wanted one.
    if (!readyRef.current || pausedRef.current || !beginFrame()) return;
    // createImageBitmap snapshots off the main thread so the worker never reads a
    // <video> that has since advanced to a different frame.
    createImageBitmap(media).then(
      (bitmap) => sendFrame(bitmap, optionsRef.current),
      // Release the claim, or one failed capture wedges the loop forever.
      () => abortFrame(),
    );
  }, [abortFrame, beginFrame, detectionsRef, sendFrame]);

  useEffect(() => {
    rafRef.current = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(rafRef.current);
  }, [loop]);

  return (
    <div className="relative overflow-hidden rounded-xl border border-neutral-800 bg-neutral-950">
      <canvas
        ref={canvasRef}
        className="block h-auto w-full"
        style={{ aspectRatio: dimensions ? `${dimensions.width}/${dimensions.height}` : "16/9" }}
      />
      {!dimensions ? (
        <div className="absolute inset-0 flex items-center justify-center text-sm text-neutral-500">
          {source.kind === "webcam" ? "Requesting camera…" : "Loading source…"}
        </div>
      ) : null}
      {dimensions ? (
        <div className="pointer-events-none absolute bottom-2 right-2 rounded bg-black/60 px-2 py-1 font-mono text-[10px] text-neutral-400">
          {dimensions.width}×{dimensions.height}
        </div>
      ) : null}
    </div>
  );
}
