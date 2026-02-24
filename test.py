#!/usr/bin/env python3
"""
End-to-end pipeline (macOS-safe):
1) MeloTTS -> synthesize text to WAV
2) OpenVoice v2 -> extract source/target speaker embeddings (SE)
3) ToneColorConverter -> convert TTS wav into reference voice
4) Save output WAVs to ./output

Run:
  conda activate openvoice
  cd /Users/noahkhaloo/Desktop/OpenVoice
  python pipeline_e2e_melo.py
"""

from __future__ import annotations

import os
import sys
import inspect
from pathlib import Path

import torch

from openvoice import se_extractor
from openvoice.api import ToneColorConverter

# MeloTTS (MyShell)
from melo.api import TTS  # pip/conda package must already be installed in this env


TEXT = "I like to eat cake for my birthday"


def find_reference_wav(repo_root: Path) -> Path:
    """
    User-provided path: /Users/noahkhaloo/Desktop/OpenVoice/reference/practive.wav
    Often it's actually practice.wav; handle both so you don't get path errors.
    """
    candidates = [
        Path("/Users/noahkhaloo/Desktop/OpenVoice/reference/practive.wav"),
        Path("/Users/noahkhaloo/Desktop/OpenVoice/reference/practice.wav"),
        repo_root / "reference" / "practive.wav",
        repo_root / "reference" / "practice.wav",
    ]
    for c in candidates:
        c = c.expanduser().resolve()
        if c.exists():
            return c
    raise FileNotFoundError(
        "Reference WAV not found. Tried:\n" + "\n".join(f"  - {str(x)}" for x in candidates)
    )


def call_converter_convert(converter, src_audio_path: str, src_se, tgt_se, out_wav_path: str):
    """
    OpenVoice forks differ slightly in ToneColorConverter.convert() signature.
    Try common call patterns.
    """
    # Common: positional
    try:
        return converter.convert(src_audio_path, src_se, tgt_se, out_wav_path)
    except TypeError:
        pass

    # Common keyword variants
    for kw in (
        dict(audio_src_path=src_audio_path, src_se=src_se, tgt_se=tgt_se, output_path=out_wav_path),
        dict(src_path=src_audio_path, src_se=src_se, tgt_se=tgt_se, out_path=out_wav_path),
        dict(src_audio_path=src_audio_path, src_se=src_se, tgt_se=tgt_se, output_wav_path=out_wav_path),
        dict(wav_src=src_audio_path, src_se=src_se, tgt_se=tgt_se, wav_out=out_wav_path),
    ):
        try:
            return converter.convert(**kw)
        except TypeError:
            continue

    sig = None
    try:
        sig = inspect.signature(converter.convert)
    except Exception:
        pass

    raise TypeError(
        "Could not call ToneColorConverter.convert() with known signatures.\n"
        f"convert() signature: {sig}\n"
        "If you paste the signature output, I’ll adapt this exactly:\n"
        "  import inspect; print(inspect.signature(tone_color_converter.convert))"
    )


def main() -> None:
    repo_root = Path("/Users/noahkhaloo/Desktop/OpenVoice").resolve()
    os.chdir(repo_root)

    out_dir = repo_root / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[info] repo_root:", repo_root)
    print("[info] cwd:", os.getcwd())
    print("[info] python:", sys.executable)

    # macOS: CPU
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("[info] torch.cuda.is_available():", torch.cuda.is_available(), "-> device:", device)

    # Reference voice
    reference_wav = find_reference_wav(repo_root)
    print("[info] reference_wav:", reference_wav)

    # OpenVoice checkpoints
    ckpt_converter_dir = repo_root / "checkpoints_v2" / "converter"
    config_json = ckpt_converter_dir / "config.json"
    checkpoint_pth = ckpt_converter_dir / "checkpoint.pth"

    if not config_json.exists():
        raise FileNotFoundError(f"Missing converter config: {config_json}")
    if not checkpoint_pth.exists():
        raise FileNotFoundError(f"Missing converter checkpoint: {checkpoint_pth}")

    # ------------------------------------------------------------------
    # 1) MeloTTS -> synthesize to WAV
    # ------------------------------------------------------------------
    tts_raw = out_dir / "tts_raw.wav"

    # MeloTTS usage pattern: create TTS(language='EN', device=...), then tts_to_file(text, speaker_id, output_path)
    # The model card shows speaker_ids = model.hps.data.spk2id and uses speaker_ids['EN-Default'] etc. :contentReference[oaicite:1]{index=1}
    print("[info] Loading MeloTTS model...")
    melo_device = "auto"  # MeloTTS supports 'auto' in examples :contentReference[oaicite:2]{index=2}
    melo = TTS(language="EN", device=melo_device)

    speaker_ids = melo.hps.data.spk2id
    # Use default English accent unless you want a specific one (EN-US, EN-BR, EN_INDIA, EN-AU).
    spk_key = "EN-Default" if "EN-Default" in speaker_ids else next(iter(speaker_ids.keys()))
    spk_id = speaker_ids[spk_key]

    speed = 1.0
    print(f"[info] Synthesizing with MeloTTS speaker='{spk_key}' -> {tts_raw}")
    melo.tts_to_file(TEXT, spk_id, str(tts_raw), speed=speed)

    if not tts_raw.exists() or tts_raw.stat().st_size == 0:
        raise RuntimeError("MeloTTS did not create tts_raw.wav (file missing or empty).")

    # ------------------------------------------------------------------
    # 2) Load OpenVoice converter
    # ------------------------------------------------------------------
    print("[info] Loading OpenVoice ToneColorConverter...")
    tone_color_converter = ToneColorConverter(str(config_json), device=device)
    tone_color_converter.load_ckpt(str(checkpoint_pth))
    print("[success] Converter loaded")

    # Reset cached Whisper model inside se_extractor (avoids stale global state)
    if hasattr(se_extractor, "model"):
        se_extractor.model = None

    # ------------------------------------------------------------------
    # 3) Extract SEs
    # ------------------------------------------------------------------
    print("[info] Extracting SOURCE SE from:", tts_raw)
    source_se, source_name = se_extractor.get_se(str(tts_raw), tone_color_converter, vad=False)

    print("[info] Extracting TARGET SE from:", reference_wav)
    target_se, target_name = se_extractor.get_se(str(reference_wav), tone_color_converter, vad=False)

    print("[info] source_name:", source_name, "source_se shape:", getattr(source_se, "shape", None))
    print("[info] target_name:", target_name, "target_se shape:", getattr(target_se, "shape", None))

    # ------------------------------------------------------------------
    # 4) Convert + write output
    # ------------------------------------------------------------------
    final_out = out_dir / "final_converted.wav"
    print("[info] Converting ->", final_out)

    call_converter_convert(
        tone_color_converter,
        str(tts_raw),
        source_se,
        target_se,
        str(final_out),
    )

    if not final_out.exists() or final_out.stat().st_size == 0:
        raise RuntimeError("Conversion did not create final_converted.wav (file missing or empty).")

    # Optional: keep a normalized copy (16k mono) using Python-only (no ffmpeg dependency)
    # If the output is already fine, you can ignore this file.
    try:
        import soundfile as sf
        import numpy as np
        import librosa

        y, sr = sf.read(str(final_out))
        if y.ndim > 1:
            y = y.mean(axis=1)  # mono
        if sr != 16000:
            y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=16000)
            sr = 16000

        final_norm = out_dir / "final_converted_16k_mono.wav"
        sf.write(str(final_norm), y, sr)
        print("[success] Normalized output ->", final_norm)
    except Exception as e:
        print("[warn] Could not write normalized file (soundfile/librosa issue). Output is still at:", final_out)
        print("[warn] Details:", repr(e))

    print("\n[DONE]")
    print("[output] tts_raw:", tts_raw)
    print("[output] final:", final_out)


if __name__ == "__main__":
    main()

