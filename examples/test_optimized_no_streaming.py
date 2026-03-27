# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0

import time

import soundfile as sf
import torch

from qwen_tts import Qwen3TTSModel

torch.set_float32_matmul_precision("high")


def run_generation(model, prompt, text, label):
    start = time.time()
    wavs, sample_rate = model.generate_voice_clone(
        text=text,
        language="English",
        voice_clone_prompt=prompt,
    )
    total_time = time.time() - start
    audio = wavs[0]
    sf.write(f"{label}.wav", audio, sample_rate)
    return {
        "label": label,
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

    test_text = "This example benchmarks offline generation after enabling the local inference optimizations."
    baseline = run_generation(model, voice_clone_prompt, test_text, "output_baseline")

    model.enable_streaming_optimizations(
        decode_window_frames=300,
        use_compile=True,
        use_cuda_graphs=False,
        compile_mode="max-autotune",
        use_fast_codebook=True,
        compile_codebook_predictor=True,
        compile_talker=True,
    )

    run_generation(model, voice_clone_prompt, "Warmup run for compilation.", "output_warmup")
    optimized = run_generation(model, voice_clone_prompt, test_text, "output_optimized")

    for result in [baseline, optimized]:
        rtf = result["total_time"] / result["audio_duration"] if result["audio_duration"] > 0 else 0.0
        print(f"{result['label']}: total={result['total_time']:.2f}s, rtf={rtf:.2f}")


if __name__ == "__main__":
    main()
