"""Hypnose-Video Generator — Streamlit Web App."""

import os
import sys
import time
import shutil
import tempfile
import wave
from pathlib import Path

import streamlit as st

# Projektverzeichnis
BASE_DIR = Path(__file__).parent.resolve()
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from scripts.generate_script import generate_hypnose_script
from scripts.generate_audio import generate_audio
from scripts.generate_music import generate_432hz_ambient
from scripts.generate_visual import generate_visual_loop
from scripts.assemble_video import mix_audio, assemble_video


# ── Helpers ──

def get_secret(key):
    try:
        return st.secrets[key]
    except (KeyError, FileNotFoundError, AttributeError):
        return os.environ.get(key, "")


def get_audio_duration(wav_path):
    with wave.open(wav_path, "rb") as wf:
        return wf.getnframes() / wf.getframerate()


# ── Seiten-Config ──

st.set_page_config(
    page_title="Hypnose-Video Generator",
    page_icon="🌀",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── UI ──

st.title("Hypnose-Video Generator")
st.caption("KI-generierte Hypnose-Videos mit 432 Hz Frequenzen")
st.divider()

thema = st.text_input(
    "Thema",
    placeholder="z.B. Raucherentwöhnung, Besserer Schlaf, Selbstvertrauen...",
)

col1, col2 = st.columns(2)
with col1:
    dauer = st.slider("Dauer (Minuten)", min_value=5, max_value=20, value=10, step=5)
with col2:
    farbschema = st.selectbox(
        "Farbschema",
        options=["purple_blue", "green_teal", "warm_golden"],
        format_func=lambda x: {
            "purple_blue": "Lila / Blau",
            "green_teal": "Grün / Türkis",
            "warm_golden": "Gold / Amber",
        }[x],
    )

st.divider()

# ── Video generieren ──

if st.button("Video erstellen", type="primary", disabled=not thema, use_container_width=True):

    anthropic_key = get_secret("ANTHROPIC_API_KEY")
    elevenlabs_key = get_secret("ELEVENLABS_API_KEY")
    openai_key = get_secret("OPENAI_API_KEY")

    missing = []
    if not anthropic_key:
        missing.append("ANTHROPIC_API_KEY")
    if not elevenlabs_key:
        missing.append("ELEVENLABS_API_KEY")
    if not openai_key:
        missing.append("OPENAI_API_KEY")

    if missing:
        st.error(f"Fehlende API Keys: {', '.join(missing)}")
        st.stop()

    tmp_dir = Path(tempfile.mkdtemp(prefix="hypnose_"))
    safe_name = thema.replace(" ", "_").replace("/", "-")

    progress = st.progress(0)
    status = st.status("Video wird erstellt...", expanded=True)

    try:
        start_time = time.time()

        # Schritt 1: Skript
        status.write("**[1/6] Hypnose-Skript generieren...**")
        progress.progress(0.05)

        template_path = str(BASE_DIR / "templates" / "hypnose_prompt.txt")
        result = generate_hypnose_script(
            thema=thema,
            dauer_minuten=dauer,
            api_key=anthropic_key,
            template_path=template_path,
        )
        status.write(f"Titel: {result['title']}")
        progress.progress(0.15)

        # Schritt 2: Stimme
        status.write("**[2/6] Stimme generieren (ElevenLabs)...**")
        progress.progress(0.20)

        voice_path = str(tmp_dir / "voice.wav")
        generate_audio(
            script=result["script"],
            output_path=voice_path,
            api_key=elevenlabs_key,
        )
        audio_duration_sec = get_audio_duration(voice_path)
        status.write(f"Sprech-Dauer: {audio_duration_sec / 60:.1f} Min")
        progress.progress(0.40)

        # Schritt 3: 432 Hz Musik
        status.write("**[3/6] 432 Hz Hintergrundmusik generieren...**")
        music_path = str(tmp_dir / "music.wav")
        generate_432hz_ambient(
            output_path=music_path,
            duration_seconds=audio_duration_sec,
        )
        progress.progress(0.55)

        # Schritt 4: Audio mischen
        status.write("**[4/6] Audio mischen...**")
        mixed_path = str(tmp_dir / "mixed_audio.wav")
        mix_audio(
            voice_path=voice_path,
            music_path=music_path,
            output_path=mixed_path,
        )
        progress.progress(0.65)

        # Schritt 5: DALL-E Visual
        status.write("**[5/6] Hypnose-Bild generieren (DALL-E)...**")
        visual_path = str(tmp_dir / "visual_loop.mp4")
        generate_visual_loop(
            output_path=visual_path,
            loop_duration=10.0,
            width=1280,
            height=720,
            fps=24,
            color_scheme=farbschema,
            openai_api_key=openai_key,
        )
        progress.progress(0.80)

        # Schritt 6: Video assemblieren
        status.write("**[6/6] Finales Video assemblieren...**")
        final_path = str(tmp_dir / f"{safe_name}.mp4")
        assemble_video(
            audio_path=mixed_path,
            visual_loop_path=visual_path,
            output_path=final_path,
            resolution=(1280, 720),
        )
        progress.progress(1.0)

        elapsed = time.time() - start_time
        status.update(label=f"Fertig! ({elapsed / 60:.1f} Min)", state="complete")

        # Video in Session State speichern
        with open(final_path, "rb") as f:
            video_bytes = f.read()

        st.session_state["video_bytes"] = video_bytes
        st.session_state["video_name"] = f"{safe_name}.mp4"
        st.session_state["video_title"] = result.get("title", thema)
        st.session_state["video_description"] = result.get("description", "")
        st.session_state["video_tags"] = result.get("tags", [])

    except Exception as e:
        status.update(label="Fehler!", state="error")
        st.error(f"Fehler: {str(e)}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ── Video anzeigen ──

if "video_bytes" in st.session_state:
    st.divider()
    st.subheader(st.session_state.get("video_title", "Fertiges Video"))
    st.video(st.session_state["video_bytes"])

    st.download_button(
        label="Video herunterladen",
        data=st.session_state["video_bytes"],
        file_name=st.session_state["video_name"],
        mime="video/mp4",
        type="primary",
        use_container_width=True,
    )

    with st.expander("YouTube Metadata"):
        st.text_input("Titel", value=st.session_state.get("video_title", ""), disabled=True)
        st.text_area("Beschreibung", value=st.session_state.get("video_description", ""), disabled=True)
        tags = st.session_state.get("video_tags", [])
        if tags:
            st.write("**Tags:**", ", ".join(tags))

st.divider()
st.caption("Generierung dauert ca. 3-8 Minuten je nach Dauer.")
