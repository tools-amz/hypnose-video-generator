"""Generiert Hypnose-Skripte mit der Claude API."""

from __future__ import annotations

import os
import anthropic


def generate_hypnose_script(
    thema: str,
    dauer_minuten: int = 20,
    api_key: str = None,
    model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 8000,
    template_path: str = None,
) -> dict:
    """
    Generiert ein Hypnose-Skript für ein gegebenes Thema.

    Returns:
        dict mit 'script', 'title', 'description', 'tags'
    """
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise ValueError("ANTHROPIC_API_KEY nicht gesetzt")

    client = anthropic.Anthropic(api_key=key)

    # Template laden
    if template_path is None:
        from pathlib import Path
        template_path = str(Path(__file__).parent.parent / "templates" / "hypnose_prompt.txt")

    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()

    hauptteil_minuten = max(dauer_minuten - 10, 5)

    prompt = template.format(
        thema=thema,
        dauer_minuten=dauer_minuten,
        hauptteil_minuten=hauptteil_minuten,
    )

    # Skript generieren
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )

    script_text = response.content[0].text

    # Titel und Beschreibung generieren
    meta_response = client.messages.create(
        model=model,
        max_tokens=500,
        messages=[
            {
                "role": "user",
                "content": (
                    f'Erstelle für dieses Hypnose-Video zum Thema "{thema}" folgendes im JSON-Format:\n'
                    "1. title: Ein YouTube-Titel auf Deutsch (max 80 Zeichen), der Heilende Frequenzen (432 Hz) erwähnt\n"
                    "2. description: Eine YouTube-Beschreibung (ca. 500 Zeichen) mit Emojis und relevanten Keywords\n"
                    "3. tags: Eine Liste von 15 relevanten YouTube-Tags auf Deutsch\n\n"
                    'Antworte NUR mit dem JSON, keine Erklärungen. Format:\n'
                    '{"title": "...", "description": "...", "tags": ["...", "..."]}'
                ),
            }
        ],
    )

    import json

    meta_text = meta_response.content[0].text
    # JSON aus der Antwort extrahieren
    json_start = meta_text.find("{")
    json_end = meta_text.rfind("}") + 1
    if json_start >= 0 and json_end > json_start:
        meta = json.loads(meta_text[json_start:json_end])
    else:
        meta = {
            "title": f"Heilende Frequenzen (432 Hz): Hypnose - {thema}",
            "description": f"Hypnose-Session zum Thema {thema} mit heilenden 432 Hz Frequenzen.",
            "tags": ["Hypnose", "432 Hz", thema, "Meditation", "Entspannung"],
        }

    return {
        "script": script_text,
        "title": meta["title"],
        "description": meta["description"],
        "tags": meta["tags"],
    }


if __name__ == "__main__":
    result = generate_hypnose_script("Raucherentwöhnung", dauer_minuten=20)
    print(f"Titel: {result['title']}")
    print(f"Skript-Länge: {len(result['script'])} Zeichen")
    print(f"\nSkript (Anfang):\n{result['script'][:500]}...")
