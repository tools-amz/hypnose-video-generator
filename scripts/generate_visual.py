"""Generiert Hypnose-Visuals mit DALL-E und nahtlosem Ken-Burns-Effekt."""

from __future__ import annotations

import os
import math
import requests
import numpy as np
from PIL import Image
from moviepy import VideoClip
from openai import OpenAI


# Bild-Prompts je nach Farbschema
IMAGE_PROMPTS = {
    "purple_blue": (
        "Abstract hypnotic spiritual mandala artwork, deep purple and blue "
        "cosmic nebula background, sacred geometry, glowing ethereal light "
        "emanating from center, soft luminous particles, dreamlike atmosphere, "
        "ultra high quality, 4k, smooth gradients, meditation visual"
    ),
    "green_teal": (
        "Abstract hypnotic spiritual mandala artwork, deep emerald green and "
        "teal ocean colors, sacred geometry, bioluminescent glowing patterns, "
        "ethereal underwater light, dreamlike atmosphere, ultra high quality, "
        "4k, smooth gradients, meditation visual"
    ),
    "warm_golden": (
        "Abstract hypnotic spiritual mandala artwork, warm golden and amber "
        "sunset colors, sacred geometry, glowing warm light from center, "
        "soft luminous particles, dreamlike atmosphere, ultra high quality, "
        "4k, smooth gradients, meditation visual"
    ),
}


def _generate_dalle_image(
    prompt: str,
    output_path: str,
    api_key: str = None,
    size: str = "1792x1024",
) -> str:
    """Generiert ein Bild mit DALL-E 3."""
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("OPENAI_API_KEY nicht gesetzt")

    client = OpenAI(api_key=key)

    print(f"  DALL-E Bild wird generiert...")
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        n=1,
        size=size,
        quality="hd",
        style="vivid",
    )

    image_url = response.data[0].url

    # Bild herunterladen
    img_data = requests.get(image_url).content
    with open(output_path, "wb") as f:
        f.write(img_data)

    print(f"  Bild gespeichert: {output_path}")
    return output_path


def _seamless_ken_burns_frame(
    img_array: np.ndarray,
    t: float,
    duration: float,
    target_w: int = 1920,
    target_h: int = 1080,
) -> np.ndarray:
    """
    Erstellt einen Frame mit nahtlosem, fliessend durchgehendem Ken-Burns-Effekt.

    Verwendet mehrere ueberlagerte Sinuswellen mit unterschiedlichen Frequenzen
    (inkommensurabel), sodass sich das Muster nie exakt wiederholt und kein
    Loop-Punkt erkennbar ist.
    """
    src_h, src_w = img_array.shape[:2]

    # Normalisierte Zeit (0 bis 1 ueber gesamte Dauer)
    nt = t / duration

    # === Zoom: ueberlagerte Wellen fuer organisches Gefuehl ===
    # Goldener Schnitt als Frequenz-Verhaeltnis -> nie periodisch
    phi = (1 + math.sqrt(5)) / 2  # 1.618...

    zoom_wave1 = math.sin(2 * math.pi * nt * 1.0) * 0.04
    zoom_wave2 = math.sin(2 * math.pi * nt * phi) * 0.03
    zoom_wave3 = math.sin(2 * math.pi * nt * (phi * phi)) * 0.02
    zoom = 1.0 + 0.05 + zoom_wave1 + zoom_wave2 + zoom_wave3
    # Zoom bewegt sich sanft zwischen ca. 0.96 und 1.14

    # === Position-Drift: langsame, organische Bewegung ===
    drift_x = (
        math.sin(2 * math.pi * nt * 0.7) * 0.025
        + math.sin(2 * math.pi * nt * 1.1 * phi) * 0.015
        + math.sin(2 * math.pi * nt * 2.3) * 0.01
    )
    drift_y = (
        math.cos(2 * math.pi * nt * 0.5) * 0.02
        + math.cos(2 * math.pi * nt * 0.9 * phi) * 0.012
        + math.cos(2 * math.pi * nt * 1.7) * 0.008
    )

    # === Crop berechnen ===
    crop_w = int(src_w / zoom)
    crop_h = int(src_h / zoom)

    # Sicherstellen dass crop nicht groesser als Bild
    crop_w = min(crop_w, src_w)
    crop_h = min(crop_h, src_h)

    # Zentrum + Drift (begrenzt auf sichere Raender)
    max_drift_x = (src_w - crop_w) // 2
    max_drift_y = (src_h - crop_h) // 2

    center_x = src_w // 2 + int(drift_x * max_drift_x * 2) if max_drift_x > 0 else src_w // 2
    center_y = src_h // 2 + int(drift_y * max_drift_y * 2) if max_drift_y > 0 else src_h // 2

    # Crop-Koordinaten
    x1 = max(0, center_x - crop_w // 2)
    y1 = max(0, center_y - crop_h // 2)
    x2 = min(src_w, x1 + crop_w)
    y2 = min(src_h, y1 + crop_h)

    # Bounds-Check
    if x2 - x1 < crop_w:
        x1 = max(0, x2 - crop_w)
    if y2 - y1 < crop_h:
        y1 = max(0, y2 - crop_h)

    # Croppen und auf Zielgroesse skalieren
    cropped = img_array[y1:y2, x1:x2]
    pil_img = Image.fromarray(cropped)
    pil_img = pil_img.resize((target_w, target_h), Image.LANCZOS)

    return np.array(pil_img)


def generate_visual_loop(
    output_path: str,
    loop_duration: float = 10.0,
    width: int = 1920,
    height: int = 1080,
    fps: int = 24,
    style: str = "spiral",
    color_scheme: str = "purple_blue",
    openai_api_key: str = None,
    image_path: str = None,
) -> str:
    """
    Generiert ein Visual-Video ueber die gesamte Dauer:
    1. DALL-E erstellt ein hypnotisches Bild
    2. Nahtloser Ken-Burns-Effekt (Zoom + Drift) ueber die gesamte Laenge
       - Kein erkennbarer Loop-Punkt
       - Fliessende, organische Bewegung
    """
    # Bild generieren oder laden
    if image_path and os.path.exists(image_path):
        print(f"  Verwende vorhandenes Bild: {image_path}")
        bg_image_path = image_path
    else:
        bg_image_path = output_path.replace(".mp4", "_bg.png")
        prompt = IMAGE_PROMPTS.get(color_scheme, IMAGE_PROMPTS["purple_blue"])

        if style and style != "spiral":
            prompt = prompt.replace("mandala", style)

        _generate_dalle_image(
            prompt=prompt,
            output_path=bg_image_path,
            api_key=openai_api_key,
        )

    # Bild laden
    bg_image = np.array(Image.open(bg_image_path).convert("RGB"))

    total_duration = loop_duration
    print(f"  Visual-Dauer: {total_duration:.1f}s ({total_duration / 60:.1f} Min)")

    def make_frame(t):
        return _seamless_ken_burns_frame(
            bg_image, t, total_duration, width, height
        )

    clip = VideoClip(make_frame, duration=total_duration)
    clip = clip.with_fps(fps)

    clip.write_videofile(
        output_path,
        fps=fps,
        codec="libx264",
        preset="medium",
        bitrate="5000k",
        logger="bar",
    )

    print(f"  Visual generiert: {total_duration:.1f}s ({output_path})")
    return output_path


if __name__ == "__main__":
    generate_visual_loop(
        "test_visual.mp4",
        loop_duration=30,
        color_scheme="purple_blue",
        fps=24,
        width=960,
        height=540,
    )
