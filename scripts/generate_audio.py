"""Generiert Hypnose-Audio mit ElevenLabs TTS (ohne FFmpeg/pydub)."""

from __future__ import annotations

import os
import re
import io
import wave
import struct
import numpy as np
from scipy.io import wavfile
from elevenlabs import ElevenLabs


def _mp3_to_wav_raw(mp3_bytes: bytes) -> np.ndarray:
    """
    Konvertiert MP3-Bytes zu WAV numpy array.
    Nutzt minimp3 über audioop oder direkten Weg über ElevenLabs output_format.
    """
    # Wir umgehen MP3 komplett - ElevenLabs kann direkt PCM liefern
    raise NotImplementedError("Use PCM output format instead")


def _parse_pauses(script: str) -> list:
    """
    Zerlegt das Skript in Segmente mit Text und Pausen.
    [PAUSE 3s] -> 3 Sekunden Stille
    [ATME TIEF EIN... UND AUS] -> 6 Sekunden Stille
    """
    pattern = r"\[PAUSE\s+(\d+)s?\]|\[ATME TIEF EIN\.{3}\s*UND AUS\]"
    segments = []
    last_end = 0

    for match in re.finditer(pattern, script):
        text_before = script[last_end : match.start()].strip()
        if text_before:
            segments.append({"type": "text", "content": text_before})

        if match.group(1):
            pause_seconds = int(match.group(1))
        else:
            pause_seconds = 6  # Atemuebung

        segments.append({"type": "pause", "duration_ms": pause_seconds * 1000})
        last_end = match.end()

    remaining = script[last_end:].strip()
    if remaining:
        segments.append({"type": "text", "content": remaining})

    return segments


def _pcm_bytes_to_numpy(pcm_bytes: bytes, sample_rate: int = 44100) -> np.ndarray:
    """Konvertiert raw PCM 16-bit signed LE bytes zu numpy array."""
    samples = np.frombuffer(pcm_bytes, dtype=np.int16)
    return samples.astype(np.float32) / 32767.0


def _save_wav(data: np.ndarray, output_path: str, sample_rate: int = 44100, channels: int = 1):
    """Speichert numpy array als WAV-Datei."""
    # Auf -1 bis 1 begrenzen
    data = np.clip(data, -1.0, 1.0)
    audio_16bit = (data * 32767).astype(np.int16)

    with wave.open(output_path, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(audio_16bit.tobytes())


def generate_audio(
    script: str,
    output_path: str,
    api_key: str = None,
    voice_id: str = "onwK4e9ZLuTAKqWW03F9",
    model_id: str = "eleven_multilingual_v2",
    stability: float = 0.75,
    similarity_boost: float = 0.75,
    style: float = 0.4,
    speed: float = 0.85,
) -> str:
    """
    Generiert Audio aus dem Hypnose-Skript mit Pausen.
    Nutzt PCM-Output von ElevenLabs (kein FFmpeg noetig).
    """
    key = api_key or os.environ.get("ELEVENLABS_API_KEY")
    if not key:
        raise ValueError("ELEVENLABS_API_KEY nicht gesetzt")

    client = ElevenLabs(api_key=key)
    segments = _parse_pauses(script)

    sample_rate = 44100
    all_audio = []

    for i, segment in enumerate(segments):
        if segment["type"] == "pause":
            # Stille erzeugen
            num_silence_samples = int(segment["duration_ms"] / 1000 * sample_rate)
            silence = np.zeros(num_silence_samples)
            all_audio.append(silence)
            print(f"  Pause: {segment['duration_ms']}ms")
        else:
            text = segment["content"]
            if not text.strip():
                continue

            print(f"  TTS Segment {i + 1}/{len(segments)}: {text[:60]}...")

            # Audio generieren - PCM Format anfordern
            audio_generator = client.text_to_speech.convert(
                voice_id=voice_id,
                text=text,
                model_id=model_id,
                output_format="pcm_44100",  # Raw PCM 16-bit, 44100 Hz
                voice_settings={
                    "stability": stability,
                    "similarity_boost": similarity_boost,
                    "style": style,
                    "use_speaker_boost": True,
                },
            )

            # Generator zu Bytes
            audio_bytes = b"".join(audio_generator)

            # PCM zu numpy
            audio_np = _pcm_bytes_to_numpy(audio_bytes, sample_rate)

            # Geschwindigkeit anpassen durch Resampling
            if speed != 1.0 and len(audio_np) > 0:
                # Einfaches Resampling fuer Geschwindigkeit
                original_len = len(audio_np)
                new_len = int(original_len / speed)
                x_old = np.linspace(0, 1, original_len)
                x_new = np.linspace(0, 1, new_len)
                audio_np = np.interp(x_new, x_old, audio_np)

            all_audio.append(audio_np)

    # Alle Segmente zusammenfuegen
    combined = np.concatenate(all_audio)

    # Als WAV speichern (mono)
    _save_wav(combined, output_path, sample_rate, channels=1)

    duration_sec = len(combined) / sample_rate
    print(f"  Audio generiert: {duration_sec:.1f}s ({output_path})")

    return output_path


if __name__ == "__main__":
    test_script = """
    Willkommen zu dieser Hypnose-Session. [PAUSE 3s]
    Schliesse sanft deine Augen. [PAUSE 5s]
    [ATME TIEF EIN... UND AUS]
    Spuere, wie dein Koerper sich entspannt.
    """
    generate_audio(test_script, "test_audio.wav")
