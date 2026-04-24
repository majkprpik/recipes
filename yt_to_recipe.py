#!/usr/bin/env python3
"""
Convert a YouTube cooking video into a structured Markdown recipe.

Usage:
    python yt_to_recipe.py <youtube_url> [--out recipes/]

Requires:
    - ollama running locally (ollama serve)
    - model pulled: ollama pull llama3.1:8b
    - yt-dlp binary (brew install yt-dlp)
    - ffmpeg (brew install ffmpeg)  -- needed for Whisper fallback & OCR frames
    - pip install requests faster-whisper pyobjc-framework-Vision pyobjc-framework-Quartz
    (Apple Vision framework handles OCR — no external OCR install needed, macOS only)
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.1:8b"
WHISPER_MODEL = "small"  # tiny, base, small, medium, large-v3
OCR_FPS = 2  # frames per second to sample for OCR

PROMPT = """You are a recipe extractor. Below is a transcript of a cooking video.
Extract the recipe and return ONLY valid JSON with this exact shape:

{
  "title": "short recipe title",
  "servings": "e.g. 4 servings, or null if unknown",
  "prep_time": "e.g. 15 min, or null",
  "cook_time": "e.g. 30 min, or null",
  "ingredients": ["1 cup flour", "2 eggs", ...],
  "steps": ["Step 1 text", "Step 2 text", ...],
  "notes": "any tips or variations mentioned, or null"
}

Rules:
- Use exact quantities from the transcript when stated.
- If a quantity is vague ("a splash"), keep it as-is.
- Steps should be concise imperative sentences.
- Do NOT include ads, intros, outros, or sponsor mentions.
- Output JSON only, no markdown fences, no commentary.

TRANSCRIPT:
---
{transcript}
---
"""


def get_transcript(url: str) -> tuple[str, str]:
    """Return (video_title, transcript_text). Tries subs first, falls back to Whisper."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        # 1. Try YouTube subtitles (fast, free, no audio download needed)
        subprocess.run([
            "yt-dlp",
            "--skip-download",
            "--write-auto-sub",
            "--write-sub",
            "--sub-lang", "en",
            "--sub-format", "vtt",
            "--convert-subs", "vtt",
            "--print-to-file", "%(title)s", str(tmp / "title.txt"),
            "-o", str(tmp / "%(id)s.%(ext)s"),
            url,
        ], check=True)

        title = (tmp / "title.txt").read_text().strip()
        vtt_files = list(tmp.glob("*.vtt"))
        if vtt_files:
            return title, vtt_to_plain(vtt_files[0].read_text())

        # 2. Fallback: download audio and transcribe with Whisper locally
        print("No subtitles found, falling back to Whisper...", file=sys.stderr)
        subprocess.run([
            "yt-dlp",
            "-x", "--audio-format", "mp3",
            "-o", str(tmp / "audio.%(ext)s"),
            url,
        ], check=True)
        audio = next(tmp.glob("audio.*"))
        return title, whisper_transcribe(audio)


def extract_on_screen_text(url: str) -> list[str]:
    """Download video, sample frames, OCR with Apple Vision, dedupe. Returns ordered unique lines."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        print("Downloading video for OCR...", file=sys.stderr)
        subprocess.run([
            "yt-dlp",
            "-f", "mp4",
            "-o", str(tmp / "video.%(ext)s"),
            url,
        ], check=True)
        video = next(tmp.glob("video.*"))

        frames_dir = tmp / "frames"
        frames_dir.mkdir()
        print(f"Extracting frames at {OCR_FPS} fps...", file=sys.stderr)
        subprocess.run([
            "ffmpeg", "-i", str(video),
            "-vf", f"fps={OCR_FPS}",
            "-q:v", "2",
            str(frames_dir / "f_%04d.jpg"),
            "-loglevel", "error",
        ], check=True)

        frames = sorted(frames_dir.glob("*.jpg"))
        print(f"OCR on {len(frames)} frames (Apple Vision)...", file=sys.stderr)

        seen = set()
        ordered = []
        for f in frames:
            for line in apple_vision_ocr(f):
                line = line.strip()
                if len(line) < 2:
                    continue
                key = re.sub(r"\s+", " ", line.lower())
                if key in seen:
                    continue
                seen.add(key)
                ordered.append(line)
        return ordered


def apple_vision_ocr(image_path: Path) -> list[str]:
    """Use macOS Vision framework (VNRecognizeTextRequest) for OCR."""
    import Vision
    from Foundation import NSURL

    url = NSURL.fileURLWithPath_(str(image_path))
    handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(url, None)
    request = Vision.VNRecognizeTextRequest.alloc().init()
    request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    request.setUsesLanguageCorrection_(True)
    handler.performRequests_error_([request], None)

    results = request.results() or []
    lines = []
    for obs in results:
        candidate = obs.topCandidates_(1)
        if candidate and len(candidate) > 0:
            lines.append(str(candidate[0].string()))
    return lines


def whisper_transcribe(audio_path: Path) -> str:
    """Transcribe audio file using faster-whisper (local, Apple Silicon friendly)."""
    from faster_whisper import WhisperModel
    model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(str(audio_path), language="en")
    return " ".join(seg.text.strip() for seg in segments)


def vtt_to_plain(vtt: str) -> str:
    """Strip VTT timestamps, tags, and dedupe consecutive repeated lines."""
    lines = []
    for line in vtt.splitlines():
        line = line.strip()
        if not line or line.startswith(("WEBVTT", "Kind:", "Language:")):
            continue
        if "-->" in line:
            continue
        line = re.sub(r"<[^>]+>", "", line)
        if lines and lines[-1] == line:
            continue
        lines.append(line)
    return " ".join(lines)


def call_ollama(transcript: str) -> dict:
    prompt = PROMPT.replace("{transcript}", transcript[:16000])
    r = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.2},
        },
        timeout=600,
    )
    r.raise_for_status()
    return json.loads(r.json()["response"])


def to_markdown(recipe: dict, source_url: str) -> str:
    def line(label, val):
        return f"- **{label}:** {val}\n" if val else ""

    md = f"# {recipe.get('title', 'Untitled recipe')}\n\n"
    md += line("Servings", recipe.get("servings"))
    md += line("Prep", recipe.get("prep_time"))
    md += line("Cook", recipe.get("cook_time"))
    md += f"- **Source:** {source_url}\n\n"
    md += "## Ingredients\n\n"
    for ing in recipe.get("ingredients", []):
        md += f"- {ing}\n"
    md += "\n## Steps\n\n"
    for i, step in enumerate(recipe.get("steps", []), 1):
        md += f"{i}. {step}\n"
    if recipe.get("notes"):
        md += f"\n## Notes\n\n{recipe['notes']}\n"
    return md


def slugify(s: str) -> str:
    s = re.sub(r"[^\w\s-]", "", s).strip().lower()
    return re.sub(r"[-\s]+", "-", s)[:60] or "recipe"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--out", default="recipes", help="output directory")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Fetching transcript...", file=sys.stderr)
    video_title, transcript = get_transcript(args.url)

    ocr_lines = extract_on_screen_text(args.url)
    print("\n===== OCR LINES (on-screen text) =====", file=sys.stderr)
    for line in ocr_lines:
        print(f"  | {line}", file=sys.stderr)
    print(f"===== {len(ocr_lines)} unique lines =====\n", file=sys.stderr)

    print(f"Extracting recipe via {MODEL}...", file=sys.stderr)
    recipe = call_ollama(transcript)

    md = to_markdown(recipe, args.url)
    fname = slugify(recipe.get("title") or video_title) + ".md"
    path = out_dir / fname
    path.write_text(md)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
