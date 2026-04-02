# coding=utf-8
from __future__ import annotations

import argparse

import uvicorn

from qwen_tts.server.app import create_app
from qwen_tts.server.service import ServerSettings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qwen-tts-serve",
        description="Run a production-style FastAPI service for Qwen3-TTS CustomVoice.",
    )
    parser.add_argument(
        "checkpoint",
        nargs="?",
        default="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        help="CustomVoice checkpoint path or Hugging Face repo id.",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0).")
    parser.add_argument("--port", type=int, default=8001, help="Bind port (default: 8001).")
    parser.add_argument("--device", default="cuda:0", help="Device for model loading.")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "bf16", "float16", "fp16", "float32", "fp32"],
        help="Torch dtype for model loading.",
    )
    parser.add_argument(
        "--flash-attn/--no-flash-attn",
        dest="flash_attn",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Enable FlashAttention-2 when available.",
    )
    parser.add_argument("--workers", type=int, default=1, help="Uvicorn worker count.")
    parser.add_argument("--max-concurrency", type=int, default=3, help="Max concurrent synthesis jobs.")
    parser.add_argument("--executor-workers", type=int, default=2, help="Background worker threads for streaming.")
    parser.add_argument("--emit-every-frames", type=int, default=8, help="Streaming emit frequency in codec frames.")
    parser.add_argument("--decode-window-frames", type=int, default=80, help="Streaming decoder window.")
    parser.add_argument("--overlap-samples", type=int, default=0, help="Crossfade overlap for stream chunks.")
    parser.add_argument("--use-compile", action="store_true", help="Enable streaming compile optimizations.")
    parser.add_argument("--compile-mode", default="reduce-overhead", help="torch.compile mode for streaming decode.")
    parser.add_argument("--max-queue-size", type=int, default=8, help="Max queued requests before 429.")
    parser.add_argument("--no-preload-model", dest="preload_model", action="store_false", help="Disable startup model preload.")
    parser.add_argument("--no-warmup", dest="warmup_enabled", action="store_false", help="Disable startup warmup request.")
    parser.add_argument("--warmup-text", default="Hello, this is a warmup request.", help="Warmup text.")
    parser.add_argument("--warmup-speaker", default="Ryan", help="Warmup speaker.")
    parser.set_defaults(preload_model=True, warmup_enabled=True)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    settings = ServerSettings(
        model_path=args.checkpoint,
        device=args.device,
        dtype=args.dtype,
        flash_attn=args.flash_attn,
        max_workers=args.executor_workers,
        max_concurrency=args.max_concurrency,
        use_compile=args.use_compile,
        compile_mode=args.compile_mode,
        decode_window_frames=args.decode_window_frames,
        emit_every_frames=args.emit_every_frames,
        overlap_samples=args.overlap_samples,
        preload_model=args.preload_model,
        warmup_enabled=args.warmup_enabled,
        warmup_text=args.warmup_text,
        warmup_speaker=args.warmup_speaker,
        max_queue_size=args.max_queue_size,
    )
    app = create_app(settings)
    uvicorn.run(app, host=args.host, port=args.port, workers=args.workers)


if __name__ == "__main__":
    main()
