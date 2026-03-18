"""Generiert harmonische 432 Hz Ambient-Musik für Hypnose-Videos."""

from __future__ import annotations

import numpy as np
from scipy.io import wavfile


def _soft_sine(t, freq, phase=0.0):
    """Weicher Sinuston mit sanften Obertönen."""
    signal = np.sin(2 * np.pi * freq * t + phase)
    # Sanfte Obertöne für wärmeren Klang
    signal += 0.3 * np.sin(2 * np.pi * freq * 2 * t + phase)  # Oktave
    signal += 0.1 * np.sin(2 * np.pi * freq * 3 * t + phase)  # Quinte darüber
    return signal


def _pad_sound(t, freq, sample_rate):
    """Erzeugt einen warmen Pad-Sound (ähnlich Synthesizer-Pad)."""
    # Mehrere leicht verstimmte Oszillatoren für Chorus-Effekt
    detune = 0.5  # Hz
    signal = np.zeros_like(t)
    for d in [-detune, 0, detune]:
        signal += np.sin(2 * np.pi * (freq + d) * t)
    signal /= 3

    # Sanfter Tiefpass-Effekt durch gewichtete Obertöne
    signal += 0.15 * np.sin(2 * np.pi * freq * 2 * t)
    signal += 0.05 * np.sin(2 * np.pi * freq * 3 * t)
    return signal


def generate_432hz_ambient(
    output_path: str,
    duration_seconds: float,
    base_frequency: float = 432.0,
    sample_rate: int = 44100,
    fade_in_sec: float = 5.0,
    fade_out_sec: float = 10.0,
) -> str:
    """
    Generiert harmonische 432 Hz Ambient-Musik.

    Verwendet musikalisch harmonische Intervalle:
    - Grundton 432 Hz
    - Quinte (648 Hz, Verhältnis 3:2)
    - Große Terz (540 Hz, Verhältnis 5:4)
    - Oktave tiefer (216 Hz)
    - Sub-Bass (108 Hz)

    Alles mit langsamen Modulationen für einen meditativen, fließenden Klang.
    """
    num_samples = int(sample_rate * duration_seconds)
    t = np.linspace(0, duration_seconds, num_samples, endpoint=False, dtype=np.float32)

    signal = np.zeros(num_samples, dtype=np.float32)

    # --- Schicht 1: Tiefer Drone (Sub-Bass) ---
    drone_freq = base_frequency / 4  # 108 Hz
    drone = _pad_sound(t, drone_freq, sample_rate)
    # Langsame Lautstärke-Modulation
    drone *= 0.12 * (1.0 + 0.3 * np.sin(2 * np.pi * 0.02 * t))
    signal += drone

    # --- Schicht 2: Grundton-Pad (432 Hz) ---
    root = _pad_sound(t, base_frequency, sample_rate)
    root *= 0.15 * (1.0 + 0.2 * np.sin(2 * np.pi * 0.03 * t))
    signal += root

    # --- Schicht 3: Oktave tiefer (216 Hz) ---
    octave_low = _pad_sound(t, base_frequency / 2, sample_rate)
    octave_low *= 0.12 * (1.0 + 0.25 * np.sin(2 * np.pi * 0.025 * t + 1.0))
    signal += octave_low

    # --- Schicht 4: Sanfte Quinte (324 Hz, eine Oktave tiefer) ---
    fifth = _pad_sound(t, base_frequency * 3 / 4, sample_rate)
    # Ein- und Ausblenden über lange Zyklen (30s)
    fifth_envelope = 0.5 + 0.5 * np.sin(2 * np.pi * (1 / 30) * t)
    fifth *= 0.08 * fifth_envelope
    signal += fifth

    # --- Schicht 5: Große Terz (270 Hz, eine Oktave tiefer) ---
    third = _pad_sound(t, base_frequency * 5 / 8, sample_rate)
    third_envelope = 0.5 + 0.5 * np.sin(2 * np.pi * (1 / 45) * t + 2.0)
    third *= 0.06 * third_envelope
    signal += third

    # --- Schicht 6: Hohe schimmernde Töne ---
    shimmer_freq = base_frequency * 2  # 864 Hz
    shimmer = np.sin(2 * np.pi * shimmer_freq * t)
    shimmer += 0.5 * np.sin(2 * np.pi * shimmer_freq * 1.5 * t)  # Quinte
    shimmer_envelope = 0.5 + 0.5 * np.sin(2 * np.pi * (1 / 20) * t + 0.5)
    shimmer *= 0.03 * shimmer_envelope
    signal += shimmer

    # --- Schicht 7: Binaural-Beat (sanfter Alpha-Wellen Effekt, 10 Hz) ---
    # Links: 432 Hz, Rechts: 442 Hz → 10 Hz Differenz (Alpha)
    binaural_left = 0.04 * np.sin(2 * np.pi * base_frequency * t)
    binaural_right = 0.04 * np.sin(2 * np.pi * (base_frequency + 10) * t)

    # --- Normalisieren ---
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        signal = signal / max_val * 0.6

    # --- Stereo mit Binaural und Räumlichkeit ---
    left = signal + binaural_left
    right = signal + binaural_right

    # Leichte Stereo-Variation für Weite
    stereo_mod = 0.05 * np.sin(2 * np.pi * 0.01 * t)
    left *= (1.0 + stereo_mod)
    right *= (1.0 - stereo_mod)

    # Nochmals normalisieren
    stereo = np.column_stack([left, right])
    max_stereo = np.max(np.abs(stereo))
    if max_stereo > 0:
        stereo = stereo / max_stereo * 0.7

    # --- Fade-in / Fade-out ---
    fade_in_samples = int(fade_in_sec * sample_rate)
    fade_out_samples = int(fade_out_sec * sample_rate)

    if fade_in_samples > 0 and fade_in_samples < num_samples:
        fade_in = np.linspace(0, 1, fade_in_samples)
        stereo[:fade_in_samples, 0] *= fade_in
        stereo[:fade_in_samples, 1] *= fade_in

    if fade_out_samples > 0 and fade_out_samples < num_samples:
        fade_out = np.linspace(1, 0, fade_out_samples)
        stereo[-fade_out_samples:, 0] *= fade_out
        stereo[-fade_out_samples:, 1] *= fade_out

    # Als 16-bit WAV speichern
    audio_16bit = (stereo * 32767).astype(np.int16)
    wavfile.write(output_path, sample_rate, audio_16bit)

    print(f"  432 Hz Ambient-Musik generiert: {duration_seconds:.1f}s ({output_path})")
    return output_path


if __name__ == "__main__":
    generate_432hz_ambient("test_music.wav", duration_seconds=30)
    print("Test-Musik erstellt: test_music.wav")
