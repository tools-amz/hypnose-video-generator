"""Assembliert das finale Hypnose-Video aus Audio, Musik und Visual (ohne FFmpeg)."""

from __future__ import annotations

import os
import wave
import numpy as np
from scipy.io import wavfile
from moviepy import AudioFileClip, VideoFileClip, concatenate_videoclips


def mix_audio(
    voice_path: str,
    music_path: str,
    output_path: str,
    music_volume: float = 0.15,
    fade_in_sec: float = 5.0,
    fade_out_sec: float = 10.0,
) -> str:
    """Mischt Stimme und Hintergrundmusik mit numpy (kein pydub/ffmpeg)."""

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
        music_sr = voice_sr

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
    Erstellt das finale Video: loopt das Visual und legt Audio darunter.
    Verwendet moviepy statt FFmpeg CLI.
    """
    # Audio laden
    audio_clip = AudioFileClip(audio_path)
    audio_duration = audio_clip.duration
    print(f"  Audio-Dauer: {audio_duration:.1f}s")

    # Visual laden
    visual_clip = VideoFileClip(visual_loop_path)
    visual_duration = visual_clip.duration

    # Visual loopen bis es die Audio-Laenge erreicht
    if visual_duration < audio_duration:
        num_loops = int(audio_duration / visual_duration) + 1
        clips = [visual_clip] * num_loops
        visual_clip = concatenate_videoclips(clips)

    # Auf Audio-Laenge schneiden
    visual_clip = visual_clip.subclipped(0, audio_duration)

    # Audio drauflegen
    final = visual_clip.with_audio(audio_clip)

    # Video schreiben
    print(f"  Video wird assembliert...")
    final.write_videofile(
        output_path,
        fps=30,
        codec="libx264",
        audio_codec="aac",
        bitrate="5000k",
        preset="medium",
        logger="bar",
    )

    # Aufraeumen
    audio_clip.close()
    visual_clip.close()
    final.close()

    # Dateigroesse
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Video fertig: {audio_duration:.0f}s, {size_mb:.1f} MB ({output_path})")

    return output_path


if __name__ == "__main__":
    print("Bitte ueber main.py ausfuehren.")
