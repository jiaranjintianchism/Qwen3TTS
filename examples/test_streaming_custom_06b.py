# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch

from qwen_tts import Qwen3TTSModel

torch.set_float32_matmul_precision("high")


def env_flag(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip().lower() not in {"0", "false", "no", "off"}


def format_seconds(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}s"


@dataclass
class RuntimeConfig:
    model_path: str
    requested_attn_impl: str
    effective_attn_impl: str
    text: str
    warmup_text: str
    language: str
    speaker: str
    emit_every_frames: int
    decode_window_frames: int
    overlap_samples: int
    max_frames: int
    warmup_max_frames: int
    enable_stream_opt: bool
    use_compile: bool
    use_cuda_graphs: bool
    compile_mode: str
    compile_talker: bool
    compile_codebook_predictor: bool
    benchmark_mode: bool
    benchmark_runs: int
    save_benchmark_audio: bool
    service_mode: bool
    warmup_enabled: bool
    output_dir: Path
    device_map: str
    dtype: torch.dtype


def build_runtime_config() -> RuntimeConfig:
    has_cuda = torch.cuda.is_available()
    device_map = "cuda:0" if has_cuda else "cpu"
    if has_cuda:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32

    requested_attn_impl = os.environ.get("QWEN3_TTS_ATTN_IMPL", "flash_attention_2")
    use_compile_default = "1" if has_cuda else "0"

    return RuntimeConfig(
        model_path=os.environ.get(
            "QWEN3_TTS_MODEL_PATH",
            r"e:\Qwen3-TTS\models\Qwen3-TTS-12Hz-0.6B-CustomVoice",
        ),
        requested_attn_impl=requested_attn_impl,
        effective_attn_impl=requested_attn_impl,
        text=os.environ.get(
            "QWEN3_TTS_TEXT",
            "你好，这是一段用于测试零点六B自定义音色流式合成效果的中文语音。",
        ),
        warmup_text=os.environ.get(
            "QWEN3_TTS_WARMUP_TEXT",
            "你好，这是启动预热。",
        ),
        language=os.environ.get("QWEN3_TTS_LANGUAGE", "Chinese"),
        speaker=os.environ.get("QWEN3_TTS_SPEAKER", "Ryan"),
        emit_every_frames=int(os.environ.get("QWEN3_TTS_EMIT_EVERY_FRAMES", "16")),
        decode_window_frames=int(os.environ.get("QWEN3_TTS_DECODE_WINDOW_FRAMES", "128")),
        overlap_samples=int(os.environ.get("QWEN3_TTS_OVERLAP_SAMPLES", "0")),
        max_frames=int(os.environ.get("QWEN3_TTS_MAX_FRAMES", "10000")),
        warmup_max_frames=int(os.environ.get("QWEN3_TTS_WARMUP_MAX_FRAMES", "128")),
        enable_stream_opt=env_flag("QWEN3_TTS_ENABLE_STREAM_OPT", "1"),
        use_compile=env_flag("QWEN3_TTS_USE_COMPILE", use_compile_default),
        use_cuda_graphs=env_flag("QWEN3_TTS_USE_CUDA_GRAPHS", "0"),
        compile_mode=os.environ.get("QWEN3_TTS_COMPILE_MODE", "reduce-overhead"),
        compile_talker=env_flag("QWEN3_TTS_COMPILE_TALKER", "0"),
        compile_codebook_predictor=env_flag("QWEN3_TTS_COMPILE_CODEBOOK_PREDICTOR", "0"),
        benchmark_mode=env_flag("QWEN3_TTS_BENCHMARK_MODE", "1"),
        benchmark_runs=int(os.environ.get("QWEN3_TTS_BENCHMARK_RUNS", "3")),
        save_benchmark_audio=env_flag("QWEN3_TTS_SAVE_BENCHMARK_AUDIO", "0"),
        service_mode=env_flag("QWEN3_TTS_SERVICE_MODE", "0"),
        warmup_enabled=env_flag("QWEN3_TTS_WARMUP", "1"),
        output_dir=Path(os.environ.get("QWEN3_TTS_OUTPUT_DIR", ".")),
        device_map=device_map,
        dtype=dtype,
    )


def load_model(config: RuntimeConfig) -> Qwen3TTSModel:
    try:
        model = Qwen3TTSModel.from_pretrained(
            config.model_path,
            device_map=config.device_map,
            dtype=config.dtype,
            attn_implementation=config.requested_attn_impl,
        )
        config.effective_attn_impl = config.requested_attn_impl
        return model
    except Exception as exc:
        if config.requested_attn_impl != "flash_attention_2":
            raise
        print(
            "[Model] flash_attention_2 load failed; falling back to sdpa. "
            f"Reason: {exc}"
        )
        model = Qwen3TTSModel.from_pretrained(
            config.model_path,
            device_map=config.device_map,
            dtype=config.dtype,
            attn_implementation="sdpa",
        )
        config.effective_attn_impl = "sdpa"
        return model


def configure_streaming(model: Qwen3TTSModel, config: RuntimeConfig) -> None:
    if not config.enable_stream_opt:
        print("[Startup] Streaming optimizations disabled")
        return

    model.enable_streaming_optimizations(
        decode_window_frames=config.decode_window_frames,
        use_compile=config.use_compile,
        use_cuda_graphs=config.use_cuda_graphs,
        compile_mode=config.compile_mode,
        compile_codebook_predictor=config.compile_codebook_predictor,
        compile_talker=config.compile_talker,
    )


def run_request(
    model: Qwen3TTSModel,
    config: RuntimeConfig,
    text: str,
    output_path: Optional[Path] = None,
    max_frames: Optional[int] = None,
) -> dict:
    start = time.time()
    chunks = []
    sample_rate = 24000
    first_chunk_latency = None

    for chunk, chunk_sr in model.stream_generate_custom_voice(
        text=text,
        language=config.language,
        speaker=config.speaker,
        emit_every_frames=config.emit_every_frames,
        decode_window_frames=config.decode_window_frames,
        overlap_samples=config.overlap_samples,
        max_frames=config.max_frames if max_frames is None else max_frames,
    ):
        if first_chunk_latency is None:
            first_chunk_latency = time.time() - start
        chunks.append(chunk)
        sample_rate = chunk_sr

    audio = np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)
    if output_path is not None:
        sf.write(output_path, audio, sample_rate)

    total_time = time.time() - start
    audio_duration = len(audio) / sample_rate if sample_rate > 0 else 0.0
    rtf = total_time / audio_duration if audio_duration > 0 else 0.0
    return {
        "first_chunk_latency": first_chunk_latency,
        "total_time": total_time,
        "audio_duration": audio_duration,
        "rtf": rtf,
        "num_chunks": len(chunks),
        "sample_rate": sample_rate,
        "saved": str(output_path) if output_path is not None else "",
    }


def warmup_model(model: Qwen3TTSModel, config: RuntimeConfig) -> None:
    if not config.warmup_enabled:
        print("[Warmup] Skipped")
        return

    print("[Warmup] Running startup warmup request...")
    result = run_request(
        model,
        config,
        text=config.warmup_text,
        output_path=None,
        max_frames=config.warmup_max_frames,
    )
    print(
        "[Warmup] complete: "
        f"first_chunk={format_seconds(result['first_chunk_latency'])}, "
        f"total={result['total_time']:.2f}s, "
        f"rtf={result['rtf']:.2f}, "
        f"chunks={result['num_chunks']}"
    )


def print_runtime_summary(config: RuntimeConfig) -> None:
    print("[Startup] Service ready")
    print(f"  model_path={config.model_path}")
    print(f"  requested_attn_impl={config.requested_attn_impl}")
    print(f"  effective_attn_impl={config.effective_attn_impl}")
    print(f"  device_map={config.device_map}")
    print(f"  dtype={config.dtype}")
    print(f"  enable_stream_opt={config.enable_stream_opt}")
    print(f"  use_compile={config.use_compile}")
    print(f"  use_cuda_graphs={config.use_cuda_graphs}")
    print(f"  compile_mode={config.compile_mode}")
    print(f"  compile_talker={config.compile_talker}")
    print(f"  compile_codebook_predictor={config.compile_codebook_predictor}")
    print(f"  benchmark_mode={config.benchmark_mode}")
    print(f"  benchmark_runs={config.benchmark_runs}")
    print(f"  save_benchmark_audio={config.save_benchmark_audio}")
    print(f"  service_mode={config.service_mode}")
    print(f"  emit_every_frames={config.emit_every_frames}")
    print(f"  decode_window_frames={config.decode_window_frames}")
    print(f"  output_dir={config.output_dir}")


def run_single_request(model: Qwen3TTSModel, config: RuntimeConfig) -> None:
    output_path = config.output_dir / "output_streaming_custom_06b.wav"
    result = run_request(model, config, text=config.text, output_path=output_path)
    print(f"first_chunk_latency={format_seconds(result['first_chunk_latency'])}")
    print(f"total_time={result['total_time']:.2f}s")
    print(f"audio_duration={result['audio_duration']:.2f}s")
    print(f"rtf={result['rtf']:.2f}")
    print(f"num_chunks={result['num_chunks']}")
    print(f"attn_impl={config.effective_attn_impl}")
    print(f"device_map={config.device_map}")
    print(f"dtype={config.dtype}")
    print(f"emit_every_frames={config.emit_every_frames}")
    print(f"decode_window_frames={config.decode_window_frames}")
    print(f"saved={output_path}")


def run_service_loop(model: Qwen3TTSModel, config: RuntimeConfig) -> None:
    request_id = 1
    print("[Service] Enter text and press Enter. Use ':quit' to exit.")
    while True:
        try:
            text = input("tts> ").strip()
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print()
            break

        if not text:
            continue
        if text in {":quit", ":exit", "quit", "exit"}:
            break

        output_path = config.output_dir / f"output_streaming_custom_06b_{request_id:04d}.wav"
        print(f"[Request {request_id}] Generating...")
        result = run_request(model, config, text=text, output_path=output_path)
        print(
            f"[Request {request_id}] first_chunk={format_seconds(result['first_chunk_latency'])}, "
            f"total={result['total_time']:.2f}s, "
            f"audio={result['audio_duration']:.2f}s, "
            f"rtf={result['rtf']:.2f}, "
            f"chunks={result['num_chunks']}, "
            f"saved={output_path}"
        )
        request_id += 1


def run_benchmark(model: Qwen3TTSModel, config: RuntimeConfig) -> None:
    results = []
    print(f"[Benchmark] Running {config.benchmark_runs} request(s)...")
    for request_id in range(1, config.benchmark_runs + 1):
        output_path = None
        if config.save_benchmark_audio:
            output_path = config.output_dir / f"output_streaming_custom_06b_bench_{request_id:04d}.wav"

        result = run_request(model, config, text=config.text, output_path=output_path)
        results.append(result)
        saved_suffix = f", saved={output_path}" if output_path is not None else ""
        print(
            f"[Benchmark {request_id}] first_chunk={format_seconds(result['first_chunk_latency'])}, "
            f"total={result['total_time']:.2f}s, "
            f"audio={result['audio_duration']:.2f}s, "
            f"rtf={result['rtf']:.2f}, "
            f"chunks={result['num_chunks']}{saved_suffix}"
        )

    if not results:
        return

    first_chunk_values = [r["first_chunk_latency"] for r in results if r["first_chunk_latency"] is not None]
    total_times = [r["total_time"] for r in results]
    audio_durations = [r["audio_duration"] for r in results]
    rtfs = [r["rtf"] for r in results]

    avg_first_chunk = sum(first_chunk_values) / len(first_chunk_values) if first_chunk_values else None
    avg_total_time = sum(total_times) / len(total_times)
    avg_audio_duration = sum(audio_durations) / len(audio_durations)
    avg_rtf = sum(rtfs) / len(rtfs)
    overall_rtf = sum(total_times) / sum(audio_durations) if sum(audio_durations) > 0 else 0.0

    print("[Benchmark] Summary")
    print(f"  runs={len(results)}")
    print(f"  avg_first_chunk_latency={format_seconds(avg_first_chunk)}")
    print(f"  min_first_chunk_latency={format_seconds(min(first_chunk_values) if first_chunk_values else None)}")
    print(f"  max_first_chunk_latency={format_seconds(max(first_chunk_values) if first_chunk_values else None)}")
    print(f"  avg_total_time={avg_total_time:.2f}s")
    print(f"  avg_audio_duration={avg_audio_duration:.2f}s")
    print(f"  avg_rtf={avg_rtf:.2f}")
    print(f"  min_rtf={min(rtfs):.2f}")
    print(f"  max_rtf={max(rtfs):.2f}")
    print(f"  overall_rtf={overall_rtf:.2f}")
    print(f"  attn_impl={config.effective_attn_impl}")
    print(f"  device_map={config.device_map}")
    print(f"  dtype={config.dtype}")
    print(f"  emit_every_frames={config.emit_every_frames}")
    print(f"  decode_window_frames={config.decode_window_frames}")


def main() -> None:
    config = build_runtime_config()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(config)
    configure_streaming(model, config)
    warmup_model(model, config)
    print_runtime_summary(config)

    if config.benchmark_mode:
        run_benchmark(model, config)
    elif config.service_mode:
        run_service_loop(model, config)
    else:
        run_single_request(model, config)


if __name__ == "__main__":
    main()
