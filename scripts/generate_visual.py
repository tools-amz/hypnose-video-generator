"""Generiert Hypnose-Visuals mit DALL-E und Ken-Burns-Effekt."""

from __future__ import annotations

import os
import math
import requests
import numpy as np
from PIL import Image, ImageFilter
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


def _ken_burns_frame(
    img_array: np.ndarray,
    t: float,
    duration: float,
    target_w: int = 1920,
    target_h: int = 1080,
) -> np.ndarray:
    """
    Erstellt einen Frame mit sanftem Ken-Burns-Effekt:
    - Langsamer Zoom (rein und raus)
    - Leichte Rotation
    - Sanfte Positionsverschiebung
    """
    src_h, src_w = img_array.shape[:2]

    # Zoom: langsam zwischen 1.0 und 1.15 pendeln
    zoom_cycle = math.sin(2 * math.pi * t / duration) * 0.5 + 0.5  # 0-1
    zoom = 1.0 + 0.15 * zoom_cycle

    # Leichte Position-Drift
    drift_x = math.sin(2 * math.pi * t / (duration * 1.3)) * 0.03  # -3% bis +3%
    drift_y = math.cos(2 * math.pi * t / (duration * 0.9)) * 0.02

    # Crop-Bereich berechnen
    crop_w = int(src_w / zoom)
    crop_h = int(src_h / zoom)

    # Zentrum + Drift
    center_x = src_w // 2 + int(drift_x * src_w)
    center_y = src_h // 2 + int(drift_y * src_h)

    # Crop-Koordinaten (mit Bounds-Check)
    x1 = max(0, center_x - crop_w // 2)
    y1 = max(0, center_y - crop_h // 2)
    x2 = min(src_w, x1 + crop_w)
    y2 = min(src_h, y1 + crop_h)

    # Falls am Rand: zurückschieben
    if x2 - x1 < crop_w:
        x1 = max(0, x2 - crop_w)
    if y2 - y1 < crop_h:
        y1 = max(0, y2 - crop_h)

    # Croppen und auf Zielgröße skalieren
    cropped = img_array[y1:y2, x1:x2]
    pil_img = Image.fromarray(cropped)
    pil_img = pil_img.resize((target_w, target_h), Image.LANCZOS)

    return np.array(pil_img)


def generate_visual_loop(
    output_path: str,
    loop_duration: float = 10.0,
    width: int = 1920,
    height: int = 1080,
    fps: int = 30,
    style: str = "spiral",
    color_scheme: str = "purple_blue",
    openai_api_key: str = None,
    image_path: str = None,
) -> str:
    """
    Generiert ein Visual-Loop-Video:
    1. DALL-E erstellt ein hypnotisches Bild
    2. Ken-Burns-Effekt (Zoom + Drift) macht daraus ein Video

    Args:
        image_path: Optionaler Pfad zu einem vorhandenen Bild (überspringt DALL-E)
    """
    # Bild generieren oder laden
    if image_path and os.path.exists(image_path):
        print(f"  Verwende vorhandenes Bild: {image_path}")
        bg_image_path = image_path
    else:
        bg_image_path = output_path.replace(".mp4", "_bg.png")
        prompt = IMAGE_PROMPTS.get(color_scheme, IMAGE_PROMPTS["purple_blue"])

        # Thema-spezifischen Prompt anpassen falls style-Info vorhanden
        if style and style != "spiral":
            prompt = prompt.replace("mandala", style)

        _generate_dalle_image(
            prompt=prompt,
            output_path=bg_image_path,
            api_key=openai_api_key,
        )

    # Bild laden
    bg_image = np.array(Image.open(bg_image_path).convert("RGB"))

    def make_frame(t):
        return _ken_burns_frame(bg_image, t, loop_duration, width, height)

    clip = VideoClip(make_frame, duration=loop_duration)
    clip = clip.with_fps(fps)

    clip.write_videofile(
        output_path,
        fps=fps,
        codec="libx264",
        preset="medium",
        bitrate="5000k",
        logger="bar",
    )

    print(f"  Visual Loop generiert: {loop_duration}s ({output_path})")
    return output_path


if __name__ == "__main__":
    generate_visual_loop(
        "test_visual.mp4",
        loop_duration=5,
        color_scheme="purple_blue",
        fps=24,
        width=960,
        height=540,
    )
