"""Local clipping: ffmpeg subclip + OpenCV face-aware vertical crop.

Two stages per highlight:
  1. Cut the source video to [start, end] with ffmpeg (re-encoded, audio kept).
  2. Reframe the cut to the target aspect ratio. For 9:16 we slide a vertical
     window horizontally across the frame to keep faces centred (Haar
     cascade — same approach as the original repo, no external models).
"""
import os
import subprocess
from typing import Dict, List, Optional, Tuple

from ..config import LOCAL_OUTPUT_DIR


def _ratio(aspect_ratio: str) -> float:
    """Parse '9:16' → 9/16, '1:1' → 1.0."""
    try:
        w, h = aspect_ratio.split(":")
        return float(w) / float(h)
    except (ValueError, ZeroDivisionError):
        return 9.0 / 16.0


def _cut_subclip(source_path: str, start: float, end: float, out_path: str) -> str:
    """ffmpeg -ss start -to end → re-encoded mp4 with audio."""
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", source_path,
        "-ss", f"{start:.3f}",
        "-to", f"{end:.3f}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "aac", "-b:a", "128k",
        out_path,
    ]
    subprocess.run(cmd, check=True)
    return out_path


def _reframe_vertical(in_path: str, out_path: str, aspect_ratio: str) -> str:
    """Crop the cut clip to the target aspect ratio, tracking faces if possible."""
    try:
        import cv2  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "opencv-python is required for --mode local. Install it with:\n"
            "    pip install -r requirements-local.txt"
        ) from e

    target_ratio = _ratio(aspect_ratio)
    if not hasattr(cv2, "CascadeClassifier"):
        # Some minimal OpenCV wheels omit Haar cascade support. Fall back to
        # a centered FFmpeg crop so clip rendering still works reliably.
        ratio = target_ratio
        crop = (
            "crop=w='if(gt(iw/ih,{r}),ih*{r},iw)':"
            "h='if(gt(iw/ih,{r}),ih,iw/{r})':"
            "x='(iw-ow)/2':y='(ih-oh)/2'"
        ).format(r=ratio)
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error", "-i", in_path,
            "-vf", crop, "-c:v", "libx264", "-preset", "veryfast",
            "-crf", "22", "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart",
            out_path,
        ], check=True)
        return out_path

    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open {in_path}")

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Compute the largest crop that fits inside the frame at the target ratio.
    if target_ratio < src_w / src_h:
        crop_h = src_h
        crop_w = int(crop_h * target_ratio)
    else:
        crop_w = src_w
        crop_h = int(crop_w / target_ratio)
    crop_w = max(2, crop_w - (crop_w % 2))
    crop_h = max(2, crop_h - (crop_h % 2))

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    silent_path = out_path + ".silent.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(silent_path, fourcc, fps, (crop_w, crop_h))

    last_center: Optional[Tuple[int, int]] = None
    smoothing = 0.15  # how aggressively to chase a new face position
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        if len(faces) > 0:
            # Pick the largest face — usually the speaker.
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            cx = x + w // 2
            cy = y + h // 2
            if last_center is None:
                last_center = (cx, cy)
            else:
                lx, ly = last_center
                last_center = (
                    int(lx + (cx - lx) * smoothing),
                    int(ly + (cy - ly) * smoothing),
                )
        if last_center is None:
            last_center = (src_w // 2, src_h // 2)

        cx, cy = last_center
        x0 = max(0, min(src_w - crop_w, cx - crop_w // 2))
        y0 = max(0, min(src_h - crop_h, cy - crop_h // 2))
        cropped = frame[y0:y0 + crop_h, x0:x0 + crop_w]
        writer.write(cropped)

    cap.release()
    writer.release()

    # Mux audio from the cut clip back onto the silent reframed video.
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", silent_path,
        "-i", in_path,
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "128k",
        "-map", "0:v:0", "-map", "1:a:0?",
        "-shortest",
        out_path,
    ]
    subprocess.run(cmd, check=True)
    os.remove(silent_path)
    return out_path


def _srt_time(seconds: float) -> str:
    total_ms = max(0, int(round(seconds * 1000)))
    hours, rem = divmod(total_ms, 3600000)
    minutes, rem = divmod(rem, 60000)
    secs, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _burn_captions(video_path: str, segments: List[Dict], clip_start: float,
                   clip_end: float, out_path: str) -> str:
    """Burn Whisper transcript captions into the rendered clip."""
    entries = []
    for segment in segments:
        start = max(clip_start, float(segment.get("start", 0)))
        end = min(clip_end, float(segment.get("end", 0)))
        text = str(segment.get("text", "")).strip()
        if not text or end <= start:
            continue
        entries.append((start - clip_start, end - clip_start, text))
    if not entries:
        return video_path

    srt_path = out_path + ".srt"
    temp_path = out_path + ".captioned.mp4"
    blocks = []
    for index, (start, end, text) in enumerate(entries, 1):
        safe_text = text.replace("-->", "→").replace("\\", "")
        blocks.append(f"{index}\n{_srt_time(start)} --> {_srt_time(end)}\n{safe_text}")
    with open(srt_path, "w", encoding="utf-8") as handle:
        handle.write("\n\n".join(blocks) + "\n")

    # Use FFmpeg's subtitles filter for readable, centered, high-contrast captions.
    escaped = srt_path.replace("\\\\", "/").replace(":", "\\:").replace("'", "\\'")
    vf = "subtitles='{}':force_style='FontName=DejaVu Sans,FontSize=20,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=3,Shadow=1,Alignment=2,MarginV=48'".format(escaped)
    try:
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error", "-i", video_path,
            "-vf", vf, "-c:v", "libx264", "-preset", "veryfast",
            "-crf", "22", "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart", temp_path,
        ], check=True)
        os.replace(temp_path, out_path)
        return out_path
    finally:
        for path in (srt_path, temp_path):
            if os.path.exists(path):
                os.remove(path)


def crop_clip_local(
    source_path: str,
    start_time: float,
    end_time: float,
    aspect_ratio: str,
    out_path: str,
    transcript_segments: Optional[List[Dict]] = None,
) -> str:
    """Cut, reframe and caption one highlight, returning the local mp4 path."""
    cut_path = out_path + ".cut.mp4"
    try:
        _cut_subclip(source_path, start_time, end_time, cut_path)
        _reframe_vertical(cut_path, out_path, aspect_ratio)
        if transcript_segments:
            _burn_captions(out_path, transcript_segments, start_time, end_time, out_path)
    finally:
        if os.path.exists(cut_path):
            os.remove(cut_path)
    return out_path


def crop_highlights_local(
    source_path: str,
    highlights: List[Dict],
    aspect_ratio: str = "9:16",
    out_dir: Optional[str] = None,
    transcript_segments: Optional[List[Dict]] = None,
) -> List[Dict]:
    out_dir = out_dir or LOCAL_OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    results: List[Dict] = []
    for i, h in enumerate(highlights, 1):
        out_path = os.path.join(out_dir, f"short_{i:02d}.mp4")
        print(f"[clip/local] {i}/{len(highlights)}: {h.get('title', '(untitled)')}", flush=True)
        try:
            crop_clip_local(
                source_path,
                float(h["start_time"]),
                float(h["end_time"]),
                aspect_ratio,
                out_path,
                transcript_segments=transcript_segments,
            )
            results.append({**h, "clip_url": out_path})
        except Exception as e:
            print(f"[clip/local] {i} failed: {e}", flush=True)
            results.append({**h, "clip_url": None, "error": str(e)})
    return results
