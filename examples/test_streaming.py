# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0

import time

import numpy as np
import soundfile as sf
import torch

from qwen_tts import Qwen3TTSModel


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

    start = time.time()
    chunks = []
    sample_rate = 24000
    first_chunk_latency = None

    for chunk, chunk_sr in model.stream_generate_voice_clone(
        text="Streaming generation is now available in the local Qwen3-TTS repository.",
        language="English",
        voice_clone_prompt=voice_clone_prompt,
        emit_every_frames=8,
        decode_window_frames=80,
    ):
        if first_chunk_latency is None:
            first_chunk_latency = time.time() - start
        chunks.append(chunk)
        sample_rate = chunk_sr

    audio = np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)
    sf.write("output_streaming.wav", audio, sample_rate)

    total_time = time.time() - start
    print(f"first_chunk_latency={first_chunk_latency:.2f}s")
    print(f"total_time={total_time:.2f}s")
    print(f"num_chunks={len(chunks)}")
    print("saved=output_streaming.wav")


if __name__ == "__main__":
    main()
