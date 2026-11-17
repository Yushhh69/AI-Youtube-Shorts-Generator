"""Hugging Face Gradio app for free CPU-based Shorts generation."""
import os
import re
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse

# Set conservative cloud defaults before importing project modules.
os.environ["LOCAL_WHISPER_MODEL"] = "tiny"
os.environ["LOCAL_WHISPER_DEVICE"] = "cpu"
os.environ["LOCAL_OUTPUT_DIR"] = "/tmp/shorts-jobs"

import gradio as gr

MAX_SOURCE_MINUTES = 30
MAX_CLIPS = 5
OUTPUT_ROOT = Path("/tmp/shorts-jobs")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def valid_youtube_url(value):
    parsed = urlparse((value or "").strip())
    host = (parsed.hostname or "").lower()
    return parsed.scheme in ("http", "https") and (
        host == "youtu.be" or host == "youtube.com" or host.endswith(".youtube.com")
    )


def cleanup_old_jobs():
    now = time.time()
    for path in OUTPUT_ROOT.iterdir():
        try:
            if now - path.stat().st_mtime > 6 * 60 * 60:
                shutil.rmtree(path, ignore_errors=True)
        except OSError:
            pass


def generate(url, count, quality, aspect):
    if not valid_youtube_url(url):
        raise gr.Error("Enter a valid YouTube video URL.")
    count = int(count)
    if count < 1 or count > MAX_CLIPS:
        raise gr.Error("Choose between 1 and 5 clips.")

    cleanup_old_jobs()
    job_dir = OUTPUT_ROOT / uuid.uuid4().hex
    job_dir.mkdir(parents=True, exist_ok=True)

    try:
        from shorts_generator import pipeline
        # Existing modules capture LOCAL_OUTPUT_DIR at import time; point their
        # output and transcript cache paths at this isolated job directory.
        import shorts_generator.local.downloader as downloader
        import shorts_generator.local.transcriber as transcriber
        import shorts_generator.local.clipper as clipper
        downloader.LOCAL_OUTPUT_DIR = str(job_dir)
        transcriber.LOCAL_OUTPUT_DIR = str(job_dir)
        clipper.LOCAL_OUTPUT_DIR = str(job_dir)

        result = pipeline.generate_shorts(
            youtube_url=url.strip(),
            num_clips=count,
            aspect_ratio=aspect,
            download_format=str(quality),
            mode="local",
        )
        files = []
        for item in result.get("shorts", []):
            path = item.get("clip_url")
            if path and os.path.isfile(path):
                files.append(path)
        if not files:
            raise gr.Error("No clips were rendered. The video may be blocked, too long, or unsupported.")
        return files, "Finished. Generated {} clip(s).".format(len(files))
    except gr.Error:
        raise
    except Exception as exc:
        raise gr.Error("Processing failed: {}. Try a shorter public YouTube video.".format(str(exc)[:350]))


with gr.Blocks(title="AI YouTube Shorts Generator") as demo:
    gr.Markdown("# AI YouTube Shorts Generator")
    gr.Markdown(
        "Cloud processing uses yt-dlp, faster-whisper (tiny, CPU), local transcript heuristics and FFmpeg. "
        "Free hosting has limited memory, runtime and daily capacity. Start with a public video under 30 minutes."
    )
    url = gr.Textbox(label="YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
    with gr.Row():
        count = gr.Slider(1, MAX_CLIPS, value=3, step=1, label="Number of clips")
        quality = gr.Dropdown(["360", "480", "720"], value="480", label="Source quality")
        aspect = gr.Dropdown(["9:16", "1:1", "16:9"], value="9:16", label="Clip aspect ratio")
    run = gr.Button("Generate Shorts", variant="primary")
    output = gr.Files(label="Generated MP4 clips")
    status = gr.Textbox(label="Status", interactive=False)
    run.click(generate, inputs=[url, count, quality, aspect], outputs=[output, status])

demo.queue(max_size=2, default_concurrency_limit=1)
demo.launch()
