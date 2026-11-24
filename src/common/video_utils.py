"""Video and audio utility functions for Scouter."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Dict, Any


def download_youtube_video(
    url: str,
    output_path: str | Path,
    resolution: str = "720",
    use_oauth2: bool = False,
    cookies_path: str | Path | None = None
) -> Path:
    """Downloads a YouTube video at the requested resolution using yt-dlp.

    Args:
        url: Direct YouTube video URL.
        output_path: Target file path (.mp4).
        resolution: Target vertical resolution (e.g. '720', '1080').
        use_oauth2: If True, uses the yt-dlp-youtube-oauth2 plugin (device code authentication).
        cookies_path: Path to optional cookies.txt file.

    Returns:
        Path to the downloaded video file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    format_str = f"bestvideo[height<={resolution}][ext=mp4]+bestaudio[ext=m4a]/best[height<={resolution}][ext=mp4]/best"

    cmd = [
        "yt-dlp",
        "-f", format_str,
        "--merge-output-format", "mp4",
        "-o", str(output_path)
    ]

    if use_oauth2:
        cmd.extend(["--username", "oauth2", "--password", ""])
    elif cookies_path and Path(cookies_path).exists():
        cmd.extend(["--cookies", str(cookies_path)])

    cmd.append(url)

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp download failed: {result.stderr}")

    return output_path


def extract_audio_from_video(video_path: str | Path, audio_path: str | Path, sample_rate: int = 16000) -> Path:
    """Extracts a mono WAV audio stream from a video file using ffmpeg.

    Args:
        video_path: Path to source video.
        audio_path: Path to target WAV file.
        sample_rate: Target audio sampling rate in Hz.

    Returns:
        Path to extracted audio file.
    """
    video_path = Path(video_path)
    audio_path = Path(audio_path)
    audio_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", "1",
        str(audio_path)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg audio extraction failed: {result.stderr}")

    return audio_path


def slice_video_clip(video_path: str | Path, output_clip_path: str | Path, start_sec: float, end_sec: float) -> Path:
    """Extracts a video sub-clip using fast stream copying in ffmpeg.

    Args:
        video_path: Path to source video.
        output_clip_path: Path to destination clip.
        start_sec: Clip start offset in seconds.
        end_sec: Clip end offset in seconds.

    Returns:
        Path to generated clip.
    """
    video_path = Path(video_path)
    output_clip_path = Path(output_clip_path)
    output_clip_path.parent.mkdir(parents=True, exist_ok=True)

    duration = max(0.1, end_sec - start_sec)

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_sec:.2f}",
        "-i", str(video_path),
        "-t", f"{duration:.2f}",
        "-c", "copy",
        str(output_clip_path)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg video slice failed: {result.stderr}")

    return output_clip_path
