#!/usr/bin/env python3
"""End-to-End Execution Script for Scouter Pipeline in Google Colab.

Usage:
    python run_pipeline.py [--url MATCH_URL] [--res RESOLUTION] [--no-oauth]
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import traceback
from pathlib import Path
import cv2

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from src.agents.cameraman import CameramanAgent
from src.agents.scorekeeper import ScorekeeperAgent
from src.agents.archivist import ArchivistAgent


DEFAULT_MATCH_URL = "https://www.youtube.com/watch?v=37Lw3KKzlns"
DEFAULT_OUTPUT_DIR = Path("data/match_g1_milano_virtus")


def setup_logger(log_file: Path) -> logging.Logger:
    """Configures dual console and file logging."""
    logger = logging.getLogger("ScouterPipeline")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    # File handler
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(str(log_file), mode="w", encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def ensure_dependencies(logger: logging.Logger) -> None:
    """Auto-installs required packages if running in a fresh environment."""
    missing: list[str] = []
    for pkg in ["easyocr", "librosa", "yt_dlp"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append("yt-dlp" if pkg == "yt_dlp" else pkg)

    if missing:
        logger.info(f"Auto-installing missing packages: {', '.join(missing)}...")
        import subprocess
        cmd = [sys.executable, "-m", "pip", "install", "-q", "-U"] + missing
        subprocess.run(cmd, check=True)
        logger.info("Dependencies verified.")


def check_environment(logger: logging.Logger) -> None:
    """Logs GPU availability and key runtime properties."""
    logger.info("==================================================")
    logger.info("       SCOUTER MULTI-AGENT PIPELINE EXECUTION     ")
    logger.info("==================================================")
    logger.info(f"Python Version: {sys.version.split()[0]}")

    if HAS_TORCH and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        logger.info(f"PyTorch Version: {torch.__version__}")
        logger.info(f"CUDA Accelerator: Available | Device: {gpu_name} ({vram_gb:.1f} GB VRAM)")
    elif HAS_TORCH:
        logger.info(f"PyTorch Version: {torch.__version__}")
        logger.warning("CUDA Accelerator: NOT available. Running on CPU (processing will be slower).")
    else:
        logger.warning("PyTorch: Not installed in local environment.")


def format_time_sec(seconds: float) -> str:
    """Formats seconds into MM:SS format."""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


def generate_run_summary(
    output_dir: Path,
    match_url: str,
    video_path: Path,
    audio_path: Path,
    timeline_path: Path,
    clips: list[dict[str, Any]],
    elapsed_total_sec: float
) -> Path:
    """Compiles a consolidated markdown summary of the entire run."""
    import json
    from datetime import datetime

    summary_file = output_dir / "RUN_SUMMARY.md"

    # Read video properties
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0.0
    cap.release()

    # Read timeline events
    timeline_events: list[dict[str, Any]] = []
    if timeline_path.exists():
        with open(timeline_path, "r", encoding="utf-8") as f:
            timeline_events = json.load(f)

    score_events = [e for e in timeline_events if e.get("type") == "SCORE_CHANGE"]
    audio_events = [e for e in timeline_events if e.get("type") == "AUDIO_SPIKE"]

    final_score_home = score_events[-1].get("score_home", 0) if score_events else 0
    final_score_away = score_events[-1].get("score_away", 0) if score_events else 0

    top_audio = sorted(audio_events, key=lambda x: x.get("peak_rms", 0.0), reverse=True)[:10]

    gpu_desc = torch.cuda.get_device_name(0) if (HAS_TORCH and torch.cuda.is_available()) else "CPU / Local"

    lines = [
        "# Scouter Pipeline Execution Summary",
        "",
        f"- **Date & Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **Match URL:** {match_url}",
        f"- **Total Run Time:** {elapsed_total_sec:.1f} seconds",
        f"- **Output Directory:** `{output_dir}`",
        "",
        "## 1. Hardware & Ingestion (Cameraman)",
        f"- **GPU Accelerator:** {gpu_desc}",
        f"- **Video Resolution:** {width}x{height} ({fps} FPS)",
        f"- **Total Duration:** {duration_sec:.1f}s ({format_time_sec(duration_sec)}) | Frames: {total_frames}",
        f"- **Audio File:** `{audio_path.name}` (16 kHz mono PCM)",
        "",
        "## 2. Acoustic Analysis (Scorekeeper Audio Profiling)",
        f"- **Total Significant Audio Events:** {len(audio_events)}",
        "",
        "### Top Loudest Moments (Crowd Roars / Referee Whistles)",
        "| Rank | Start (MM:SS) | End (MM:SS) | Duration | Peak RMS |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]

    for i, ev in enumerate(top_audio, 1):
        st = ev.get("start_sec", 0.0)
        en = ev.get("end_sec", 0.0)
        dur = en - st
        lines.append(f"| #{i} | {format_time_sec(st)} | {format_time_sec(en)} | {dur:.2f}s | {ev.get('peak_rms', 0.0):.4f} |")

    lines.extend([
        "",
        "## 3. Scoreboard Tracking & OCR (Scorekeeper)",
        f"- **Total Scored Possessions Detected:** {len(score_events)}",
        f"- **Final Score Detected:** HOME {final_score_home} - {final_score_away} AWAY",
        "",
        "### Score Evolution Timeline",
        "| # | Timestamp | Match Time | Scoring Team | Delta | Score (HOME - AWAY) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ])

    for i, ev in enumerate(score_events, 1):
        ts = ev.get("timestamp_sec", 0.0)
        lines.append(
            f"| {i:02d} | {ts:.1f}s | {format_time_sec(ts)} | **{ev.get('team')}** | +{ev.get('points_delta')} | {ev.get('score_home')} - {ev.get('score_away')} |"
        )

    lines.extend([
        "",
        "## 4. Possession Clips Extracted (Archivist)",
        f"- **Total Clips Sliced:** {len(clips)}",
        "",
        "| Clip ID | File | Window (sec) | Points Delta | Score Event Time |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ])

    for c in clips:
        ev = c.get("score_event", {})
        ts = ev.get("timestamp_sec", 0.0)
        lines.append(
            f"| `{c.get('clip_id')}` | `{c.get('clip_file')}` | {c.get('start_sec', 0):.1f}s - {c.get('end_sec', 0):.1f}s | +{ev.get('points_delta', 2)} ({ev.get('team', '')}) | {format_time_sec(ts)} |"
        )

    lines.extend([
        "",
        "## 5. Artifacts Checklist",
        f"- [x] Full Video: `{video_path}`",
        f"- [x] Audio Track: `{audio_path}`",
        f"- [x] Scoreboard Snapshot: `{output_dir / 'scoreboard_preview.png'}`",
        f"- [x] Timeline Events JSON: `{timeline_path}`",
        f"- [x] Action Clips Directory: `{output_dir / 'clips'}`",
        f"- [x] Execution Log: `{output_dir / 'pipeline.log'}`",
        f"- [x] Run Summary: `{summary_file}`",
        ""
    ])

    content = "\n".join(lines)
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(content)

    return summary_file


def run(match_url: str, output_dir: Path, resolution: str = "720", use_oauth2: bool = True) -> int:
    """Executes the pipeline stages sequentially with structured error handling."""
    pipeline_start_time = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "pipeline.log"
    logger = setup_logger(log_path)

    ensure_dependencies(logger)
    check_environment(logger)

    # -------------------------------------------------------------
    # STAGE 1: Cameraman Ingestion
    # -------------------------------------------------------------
    logger.info("\n>>> STAGE 1: Cameraman Ingestion")
    video_path = output_dir / "raw_video.mp4"
    audio_path = output_dir / "audio.wav"

    if video_path.exists() and audio_path.exists():
        logger.info(f"Existing video and audio found at {output_dir}. Skipping download.")
    else:
        logger.info(f"Downloading stream from: {match_url}")
        logger.info(f"Target resolution: {resolution}p (Android client bypass active)")
        start_time = time.time()
        try:
            import subprocess

            format_str = f"best[height<={resolution}][ext=mp4]/bestvideo[height<={resolution}]+bestaudio/18/best"
            cmd = [
                "yt-dlp",
                "--extractor-args", "youtube:player_client=android,web",
                "-f", format_str,
                "--merge-output-format", "mp4",
                "-o", str(video_path),
                match_url
            ]

            proc = subprocess.run(cmd, check=False)
            if proc.returncode != 0:
                raise RuntimeError(f"yt-dlp exited with returncode {proc.returncode}")

            # Extract mono WAV audio for Scorekeeper
            logger.info("Extracting mono audio track via FFmpeg...")
            audio_cmd = [
                "ffmpeg", "-y",
                "-i", str(video_path),
                "-vn",
                "-acodec", "pcm_s16le",
                "-ar", "16000",
                "-ac", "1",
                str(audio_path)
            ]
            subprocess.run(audio_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            elapsed = time.time() - start_time
            logger.info(f"Ingestion successful in {elapsed:.1f}s")
        except Exception as e:
            logger.error(f"Stage 1 Ingestion Failed: {e}")
            logger.error(traceback.format_exc())
            return 1

    # -------------------------------------------------------------
    # STAGE 2: Frame Preview & Scoreboard Inspection
    # -------------------------------------------------------------
    logger.info("\n>>> STAGE 2: Scoreboard Inspection & Preview")
    preview_path = output_dir / "scoreboard_preview.png"
    try:
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        # Sample at minute 5:00 (300 sec)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(300 * fps))
        ret, frame = cap.read()
        cap.release()

        if ret:
            cv2.imwrite(str(preview_path), frame)
            logger.info(f"Saved game frame snapshot for visual inspection to: {preview_path}")
        else:
            logger.warning("Could not extract frame at 05:00 timestamp.")
    except Exception as e:
        logger.warning(f"Frame preview generation encountered an error: {e}")

    # -------------------------------------------------------------
    # STAGE 3: Scorekeeper Audio & Timeline Processing
    # -------------------------------------------------------------
    logger.info("\n>>> STAGE 3: Scorekeeper Timeline Generation")
    scorekeeper = ScorekeeperAgent(output_dir=output_dir)

    # Calibrated LBA overlay normalized bounding box (bottom-left: x, y, w, h)
    scoreboard_roi = (0.03, 0.83, 0.34, 0.13)

    try:
        logger.info("Detecting crowd cheers and whistle energy spikes from audio track...")
        audio_spikes = scorekeeper.detect_audio_spikes(audio_path)
        logger.info(f"Detected {len(audio_spikes)} significant audio energy events.")

        timeline_path = scorekeeper.generate_timeline(
            video_path=video_path,
            audio_path=audio_path,
            scoreboard_roi=scoreboard_roi
        )
        logger.info(f"Timeline events successfully exported to: {timeline_path}")
    except Exception as e:
        logger.error(f"Stage 3 Scorekeeper Failed: {e}")
        logger.error(traceback.format_exc())
        return 1

    # -------------------------------------------------------------
    # STAGE 4: Archivist Possession Slicing
    # -------------------------------------------------------------
    logger.info("\n>>> STAGE 4: Archivist Possession Slicing")
    archivist = ArchivistAgent(output_dir=output_dir)
    try:
        clips = archivist.extract_possessions(
            video_path=video_path,
            timeline_events_path=timeline_path,
            lookback_sec=14.0,
            follow_through_sec=2.0
        )
        logger.info(f"Archivist extracted {len(clips)} action clips into: {archivist.clips_dir}")
    except Exception as e:
        logger.error(f"Stage 4 Archivist Failed: {e}")
        logger.error(traceback.format_exc())
        return 1

    # -------------------------------------------------------------
    # STAGE 5: Consolidated Summary Generation
    # -------------------------------------------------------------
    elapsed_total = time.time() - pipeline_start_time
    logger.info("\n>>> STAGE 5: Generating RUN_SUMMARY.md")
    try:
        summary_path = generate_run_summary(
            output_dir=output_dir,
            match_url=match_url,
            video_path=video_path,
            audio_path=audio_path,
            timeline_path=timeline_path,
            clips=clips,
            elapsed_total_sec=elapsed_total
        )
        logger.info(f"Summary report generated at: {summary_path}")

        # Print formatted summary to stdout for immediate copy-pasting from Colab
        print("\n" + "=" * 65)
        print(summary_path.read_text(encoding="utf-8"))
        print("=" * 65 + "\n")
    except Exception as e:
        logger.warning(f"Could not generate run summary: {e}")

    logger.info("==================================================")
    logger.info("       SCOUTER PIPELINE COMPLETED SUCCESSFULLY!   ")
    logger.info(f"Results saved in: {output_dir}")
    logger.info(f"Summary report: {output_dir / 'RUN_SUMMARY.md'}")
    logger.info(f"Log file: {log_path}")
    logger.info("==================================================")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Scouter Multi-Agent Pipeline Runner")
    parser.add_argument("--url", default=DEFAULT_MATCH_URL, help="YouTube match URL")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory")
    parser.add_argument("--res", default="720", help="Video resolution (default: 720)")
    parser.add_argument("--no-oauth", action="store_true", help="Disable YouTube OAuth2 device code flow")

    args = parser.parse_args()

    sys.exit(run(
        match_url=args.url,
        output_dir=Path(args.output_dir),
        resolution=args.res,
        use_oauth2=not args.no_oauth
    ))


if __name__ == "__main__":
    main()
