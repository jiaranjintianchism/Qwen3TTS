# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0

import time
import os

import numpy as np
import soundfile as sf
import torch

from qwen_tts import Qwen3TTSModel


def main():
    model_path = os.environ.get(
        "QWEN3_TTS_MODEL_PATH",
        r"e:\Qwen3-TTS\models\Qwen3-TTS-12Hz-0.6B-CustomVoice",
    )
    attn_impl = os.environ.get("QWEN3_TTS_ATTN_IMPL", "flash_attention_2")
    test_text = os.environ.get(
        "QWEN3_TTS_TEXT",
        "Streaming generation for 0.6B CustomVoice is running in the local repository.",
    )
    language = os.environ.get("QWEN3_TTS_LANGUAGE", "English")
    speaker = os.environ.get("QWEN3_TTS_SPEAKER", "Ryan")
    emit_every_frames = int(os.environ.get("QWEN3_TTS_EMIT_EVERY_FRAMES", "16"))
    decode_window_frames = int(os.environ.get("QWEN3_TTS_DECODE_WINDOW_FRAMES", "128"))
    overlap_samples = int(os.environ.get("QWEN3_TTS_OVERLAP_SAMPLES", "0"))
    max_frames = int(os.environ.get("QWEN3_TTS_MAX_FRAMES", "10000"))
    enable_stream_opt = os.environ.get("QWEN3_TTS_ENABLE_STREAM_OPT", "1") != "0"
    has_cuda = torch.cuda.is_available()
    device_map = "cuda:0" if has_cuda else "cpu"
    if has_cuda:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32

    try:
        model = Qwen3TTSModel.from_pretrained(
            model_path,
            device_map=device_map,
            dtype=dtype,
            attn_implementation=attn_impl,
        )
    except Exception:
        if attn_impl == "flash_attention_2":
            model = Qwen3TTSModel.from_pretrained(
                model_path,
                device_map=device_map,
                dtype=dtype,
                attn_implementation="sdpa",
            )
        else:
            raise

    if enable_stream_opt:
        model.enable_streaming_optimizations(
            decode_window_frames=decode_window_frames,
            use_compile=has_cuda,
            use_cuda_graphs=has_cuda,
            compile_mode="reduce-overhead",
            compile_codebook_predictor=True,
            compile_talker=True,
        )

    start = time.time()
    chunks = []
    sample_rate = 24000
    first_chunk_latency = None

    for chunk, chunk_sr in model.stream_generate_custom_voice(
        text=test_text,
        language=language,
        speaker=speaker,
        emit_every_frames=emit_every_frames,
        decode_window_frames=decode_window_frames,
        overlap_samples=overlap_samples,
        max_frames=max_frames,
    ):
        if first_chunk_latency is None:
            first_chunk_latency = time.time() - start
        chunks.append(chunk)
        sample_rate = chunk_sr

    audio = np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)
    sf.write("output_streaming_custom_06b.wav", audio, sample_rate)

    total_time = time.time() - start
    audio_duration = len(audio) / sample_rate if sample_rate > 0 else 0.0
    rtf = total_time / audio_duration if audio_duration > 0 else 0.0
    print(f"first_chunk_latency={first_chunk_latency:.2f}s")
    print(f"total_time={total_time:.2f}s")
    print(f"audio_duration={audio_duration:.2f}s")
    print(f"rtf={rtf:.2f}")
    print(f"num_chunks={len(chunks)}")
    print(f"attn_impl={attn_impl}")
    print(f"device_map={device_map}")
    print(f"dtype={dtype}")
    print(f"emit_every_frames={emit_every_frames}")
    print(f"decode_window_frames={decode_window_frames}")
    print("saved=output_streaming_custom_06b.wav")


if __name__ == "__main__":
    main()
