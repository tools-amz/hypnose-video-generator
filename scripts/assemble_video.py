"""Assembliert das finale Hypnose-Video aus Audio, Musik und Visual."""

from __future__ import annotations

import os
import subprocess
import numpy as np
from scipy.io import wavfile


def mix_audio(
    voice_path: str,
    music_path: str,
    output_path: str,
    music_volume: float = 0.15,
    fade_in_sec: float = 5.0,
    fade_out_sec: float = 10.0,
) -> str:
    """Mischt Stimme und Hintergrundmusik mit numpy."""

    # WAV-Dateien laden
    voice_sr, voice_data = wavfile.read(voice_path)
    music_sr, music_data = wavfile.read(music_path)

    # Zu float konvertieren
    if voice_data.dtype == np.int16:
        voice_data = voice_data.astype(np.float32) / 32767.0
    if music_data.dtype == np.int16:
        music_data = music_data.astype(np.float32) / 32767.0

    # Mono zu Stereo konvertieren falls noetig
    if voice_data.ndim == 1:
        voice_data = np.column_stack([voice_data, voice_data])
    if music_data.ndim == 1:
        music_data = np.column_stack([music_data, music_data])

    # Sample-Rate angleichen (einfaches Resampling)
    if music_sr != voice_sr:
        ratio = voice_sr / music_sr
        new_len = int(len(music_data) * ratio)
        x_old = np.linspace(0, 1, len(music_data))
        x_new = np.linspace(0, 1, new_len)
        music_left = np.interp(x_new, x_old, music_data[:, 0])
        music_right = np.interp(x_new, x_old, music_data[:, 1])
        music_data = np.column_stack([music_left, music_right])

    # Musik auf Stimm-Laenge bringen (loopen falls noetig)
    voice_len = len(voice_data)
    if len(music_data) < voice_len:
        repeats = (voice_len // len(music_data)) + 1
        music_data = np.tile(music_data, (repeats, 1))
    music_data = music_data[:voice_len]

    # Musik-Lautstaerke anpassen
    music_data = music_data * music_volume

    # Fade-in fuer Musik
    fade_in_samples = int(fade_in_sec * voice_sr)
    if fade_in_samples > 0 and fade_in_samples < voice_len:
        fade_in = np.linspace(0, 1, fade_in_samples).reshape(-1, 1)
        music_data[:fade_in_samples] *= fade_in

    # Fade-out fuer Musik
    fade_out_samples = int(fade_out_sec * voice_sr)
    if fade_out_samples > 0 and fade_out_samples < voice_len:
        fade_out = np.linspace(1, 0, fade_out_samples).reshape(-1, 1)
        music_data[-fade_out_samples:] *= fade_out

    # Mixen
    combined = voice_data + music_data

    # Gesamt-Fade
    if fade_in_samples > 0 and fade_in_samples < voice_len:
        fade_in = np.linspace(0, 1, fade_in_samples).reshape(-1, 1)
        combined[:fade_in_samples] *= fade_in
    if fade_out_samples > 0 and fade_out_samples < voice_len:
        fade_out = np.linspace(1, 0, fade_out_samples).reshape(-1, 1)
        combined[-fade_out_samples:] *= fade_out

    # Normalisieren
    max_val = np.max(np.abs(combined))
    if max_val > 0:
        combined = combined / max_val * 0.85

    # Speichern
    audio_16bit = (np.clip(combined, -1.0, 1.0) * 32767).astype(np.int16)
    wavfile.write(output_path, voice_sr, audio_16bit)

    duration = voice_len / voice_sr
    print(f"  Audio gemischt: {duration:.1f}s ({output_path})")
    return output_path


def assemble_video(
    audio_path: str,
    visual_loop_path: str,
    output_path: str,
    resolution: tuple = (1920, 1080),
) -> str:
    """
    Erstellt das finale Video: kombiniert Visual und Audio mit ffmpeg.
    """
    # FFmpeg: Video + Audio zusammenfuegen
    cmd = [
        "ffmpeg", "-y",
        "-i", visual_loop_path,
        "-i", audio_path,
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "20",
        "-c:a", "aac",
        "-b:a", "192k",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-shortest",
        output_path,
    ]

    print(f"  Video wird assembliert...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  FFmpeg stderr: {result.stderr[:500]}")
        raise RuntimeError(f"FFmpeg Fehler: {result.stderr[:200]}")

    # Dateigroesse
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Video fertig: {size_mb:.1f} MB ({output_path})")

    return output_path


if __name__ == "__main__":
    print("Bitte ueber main.py ausfuehren.")
