"""Hugging Face Gradio app for free CPU-based Shorts generation."""
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse

os.environ["LOCAL_WHISPER_MODEL"] = "tiny"
os.environ["LOCAL_WHISPER_DEVICE"] = "cpu"
os.environ["LOCAL_OUTPUT_DIR"] = "/tmp/shorts-jobs"

import gradio as gr
import spaces

@spaces.GPU
def zerogpu_compatibility_check():
    return True

MAX_SOURCE_MINUTES = 30
MAX_CLIPS = 5
MAX_UPLOAD_BYTES = 500 * 1024 * 1024
OUTPUT_ROOT = Path("/tmp/shorts-jobs")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def valid_youtube_url(value):
    parsed = urlparse((value or "").strip())
    host = (parsed.hostname or "").lower()
    return parsed.scheme in ("http", "https") and (
        host == "youtu.be" or host == "youtube.com" or host.endswith(".youtube.com")
    )


def media_duration(path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True, check=True, timeout=30
    )
    return float(result.stdout.strip())


def cleanup_old_jobs():
    now = time.time()
    for path in OUTPUT_ROOT.iterdir():
        try:
            if now - path.stat().st_mtime > 6 * 60 * 60:
                shutil.rmtree(path, ignore_errors=True)
        except OSError:
            pass


def generate(url, uploaded_file, count, quality, aspect):
    count = int(count)
    if count < 1 or count > MAX_CLIPS:
        raise gr.Error("Choose between 1 and 5 clips.")
    if uploaded_file:
        source = str(uploaded_file)
        if not os.path.isfile(source):
            raise gr.Error("Uploaded video file could not be read.")
        if os.path.getsize(source) > MAX_UPLOAD_BYTES:
            raise gr.Error("Upload must be 500 MB or smaller.")
    else:
        if not valid_youtube_url(url):
            raise gr.Error("Enter a valid YouTube URL or upload a video file.")
        try:
            import yt_dlp
            opts = {"quiet": True, "no_warnings": True, "skip_download": True}
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url.strip(), download=False)
            duration = float(info.get("duration") or 0)
            if duration and duration > MAX_SOURCE_MINUTES * 60:
                raise gr.Error("For free-tier testing, the source must be 30 minutes or shorter.")
        except gr.Error:
            raise
        except Exception as exc:
            raise gr.Error(
                "YouTube blocked this cloud request (often an anti-bot check). "
                "Upload the video file instead. Details: {}".format(str(exc)[:180])
            )

    cleanup_old_jobs()
    job_dir = OUTPUT_ROOT / uuid.uuid4().hex
    job_dir.mkdir(parents=True, exist_ok=True)
    try:
        if uploaded_file:
            ext = Path(source).suffix.lower() or ".mp4"
            local_source = job_dir / ("uploaded_source" + ext)
            shutil.copy2(source, local_source)
            try:
                duration = media_duration(str(local_source))
            except Exception:
                raise gr.Error("Could not read this video. Please upload a valid MP4/MOV/WebM file.")
            if duration > MAX_SOURCE_MINUTES * 60:
                raise gr.Error("For free-tier testing, the video must be 30 minutes or shorter.")
            source = str(local_source)

        from shorts_generator import pipeline
        import shorts_generator.local.downloader as downloader
        import shorts_generator.local.transcriber as transcriber
        import shorts_generator.local.clipper as clipper
        downloader.LOCAL_OUTPUT_DIR = str(job_dir)
        transcriber.LOCAL_OUTPUT_DIR = str(job_dir)
        clipper.LOCAL_OUTPUT_DIR = str(job_dir)

        result = pipeline.generate_shorts(
            youtube_url=source,
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
            raise gr.Error("No clips were rendered. Try another shorter video.")
        return files, "Finished. Generated {} clip(s).".format(len(files))
    except gr.Error:
        raise
    except Exception as exc:
        raise gr.Error("Processing failed: {}.".format(str(exc)[:300]))


with gr.Blocks(title="AI YouTube Shorts Generator") as demo:
    gr.Markdown("# AI YouTube Shorts Generator")
    gr.Markdown(
        "Cloud processing uses faster-whisper (tiny, CPU), local highlight ranking and FFmpeg. "
        "Free hosting has limited runtime and daily capacity. Videos must be 30 minutes or shorter."
    )
    url = gr.Textbox(label="YouTube URL (may be blocked by YouTube)", placeholder="https://www.youtube.com/watch?v=...")
    uploaded = gr.File(label="Or upload a video file (MP4/MOV/WebM, max 500 MB)", file_types=["video"], type="filepath")
    with gr.Row():
        count = gr.Slider(1, MAX_CLIPS, value=3, step=1, label="Number of clips")
        quality = gr.Dropdown(["360", "480", "720"], value="480", label="Download quality (URL only)")
        aspect = gr.Dropdown(["9:16", "1:1", "16:9"], value="9:16", label="Clip aspect ratio")
    run = gr.Button("Generate Shorts", variant="primary")
    output = gr.Files(label="Generated MP4 clips")
    status = gr.Textbox(label="Status", interactive=False)
    run.click(generate, inputs=[url, uploaded, count, quality, aspect], outputs=[output, status])

demo.queue(max_size=2, default_concurrency_limit=1)
demo.launch()
