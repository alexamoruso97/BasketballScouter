"""Cameraman Agent: Ingestion, video/audio extraction, and metadata extraction."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any
import cv2

from src.common.video_utils import download_youtube_video, extract_audio_from_video


class CameramanAgent:
    """Manages raw video download and preparation."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process(
        self,
        youtube_url: str,
        resolution: str = "720",
        use_oauth2: bool = False,
        cookies_path: str | Path | None = None
    ) -> Dict[str, Any]:
        """Downloads the video and extracts audio and metadata."""
        video_path = self.output_dir / "raw_video.mp4"
        audio_path = self.output_dir / "audio.wav"
        metadata_path = self.output_dir / "video_metadata.json"

        # 1. Download
        download_youtube_video(
            url=youtube_url,
            output_path=video_path,
            resolution=resolution,
            use_oauth2=use_oauth2,
            cookies_path=cookies_path
        )

        # 2. Extract Audio
        extract_audio_from_video(video_path, audio_path)

        # 3. Read Video Metadata
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / fps if fps > 0 else 0.0
        cap.release()

        metadata = {
            "source_url": youtube_url,
            "video_path": str(video_path),
            "audio_path": str(audio_path),
            "fps": fps,
            "width": width,
            "height": height,
            "total_frames": total_frames,
            "duration_sec": duration_sec
        }

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return metadata
