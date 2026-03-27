# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0

import time

import numpy as np
import soundfile as sf
import torch

from qwen_tts import Qwen3TTSModel

torch.set_float32_matmul_precision("high")


def run_stream(model, prompt, text, label, emit_every_frames=4, decode_window_frames=80):
    start = time.time()
    first_chunk_latency = None
    chunks = []
    sample_rate = 24000

    for chunk, chunk_sr in model.stream_generate_voice_clone(
        text=text,
        language="English",
        voice_clone_prompt=prompt,
        emit_every_frames=emit_every_frames,
        decode_window_frames=decode_window_frames,
        use_optimized_decode=True,
    ):
        if first_chunk_latency is None:
            first_chunk_latency = time.time() - start
        chunks.append(chunk)
        sample_rate = chunk_sr

    audio = np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)
    total_time = time.time() - start
    sf.write(f"{label}.wav", audio, sample_rate)
    return {
        "label": label,
        "first_chunk_latency": first_chunk_latency,
        "total_time": total_time,
        "audio_duration": len(audio) / sample_rate if sample_rate > 0 else 0,
    }


def main():
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    ref_audio = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone.wav"
    ref_text = (
        "Okay. Yeah. I resent you. I love you. I respect you. But you know what? "
        "You blew it! And thanks to you."
    )
    voice_clone_prompt = model.create_voice_clone_prompt(
        ref_audio=ref_audio,
        ref_text=ref_text,
    )

    test_text = "This example compares baseline streaming with the optimized streaming path."

    baseline = run_stream(model, voice_clone_prompt, test_text, "output_streaming_baseline")

    model.enable_streaming_optimizations(
        decode_window_frames=80,
        use_compile=True,
        use_cuda_graphs=False,
        compile_mode="reduce-overhead",
        use_fast_codebook=True,
        compile_codebook_predictor=True,
        compile_talker=True,
    )

    run_stream(model, voice_clone_prompt, "Warmup run for compilation.", "output_streaming_warmup")
    optimized = run_stream(model, voice_clone_prompt, test_text, "output_streaming_optimized")

    for result in [baseline, optimized]:
        rtf = result["total_time"] / result["audio_duration"] if result["audio_duration"] > 0 else 0.0
        print(
            f"{result['label']}: first_chunk={result['first_chunk_latency']:.2f}s, "
            f"total={result['total_time']:.2f}s, rtf={rtf:.2f}"
        )


if __name__ == "__main__":
    main()
