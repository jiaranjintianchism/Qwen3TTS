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
    attn_impl = os.environ.get("QWEN3_TTS_ATTN_IMPL", "sdpa")
    has_cuda = torch.cuda.is_available()
    device_map = "cuda:0" if has_cuda else "cpu"
    dtype = torch.bfloat16 if has_cuda else torch.float32

    model = Qwen3TTSModel.from_pretrained(
        model_path,
        device_map=device_map,
        dtype=dtype,
        attn_implementation=attn_impl,
    )

    start = time.time()
    chunks = []
    sample_rate = 24000
    first_chunk_latency = None

    for chunk, chunk_sr in model.stream_generate_custom_voice(
        text="Streaming generation for 0.6B CustomVoice is running in the local repository.",
        language="English",
        speaker="Ryan",
        emit_every_frames=8,
        decode_window_frames=80,
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
    print("saved=output_streaming_custom_06b.wav")


if __name__ == "__main__":
    main()
