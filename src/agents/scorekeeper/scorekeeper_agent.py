"""Scorekeeper Agent: Scoreboard OCR tracking and audio peak detection."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import cv2
import numpy as np


class ScorekeeperAgent:
    """Monitors game clock, scoreboard changes, and crowd/whistle audio energy spikes."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def detect_audio_spikes(
        self,
        audio_path: str | Path,
        threshold_factor: float = 3.2,
        merge_gap_sec: float = 1.5,
        min_duration_sec: float = 0.4
    ) -> List[Dict[str, Any]]:
        """Identifies sustained crowd cheers and whistle energy events with hysteresis merging."""
        import librosa

        y, sr = librosa.load(str(audio_path), sr=16000)
        frame_length = 2048
        hop_length = 512
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        times = librosa.times_like(rms, sr=sr, hop_length=hop_length)

        mean_rms = float(np.mean(rms))
        std_rms = float(np.std(rms))
        spike_threshold = mean_rms + (threshold_factor * std_rms)

        raw_events: List[Dict[str, Any]] = []
        is_above = False
        start_t = 0.0
        max_val = 0.0

        for t, val in zip(times, rms):
            val_f = float(val)
            t_f = float(t)
            if val_f >= spike_threshold and not is_above:
                is_above = True
                start_t = t_f
                max_val = val_f
            elif is_above:
                if val_f >= spike_threshold:
                    max_val = max(max_val, val_f)
                else:
                    is_above = False
                    raw_events.append({
                        "type": "AUDIO_SPIKE",
                        "start_sec": round(start_t, 2),
                        "end_sec": round(t_f, 2),
                        "peak_rms": round(max_val, 4)
                    })

        # Merge close events within merge_gap_sec to capture sustained cheers
        merged_events: List[Dict[str, Any]] = []
        for ev in raw_events:
            if not merged_events:
                merged_events.append(ev)
            else:
                if ev["start_sec"] - merged_events[-1]["end_sec"] <= merge_gap_sec:
                    merged_events[-1]["end_sec"] = ev["end_sec"]
                    merged_events[-1]["peak_rms"] = max(merged_events[-1]["peak_rms"], ev["peak_rms"])
                else:
                    merged_events.append(ev)

        # Filter out micro-spikes shorter than min_duration_sec
        filtered_events = [
            ev for ev in merged_events
            if (ev["end_sec"] - ev["start_sec"]) >= min_duration_sec
        ]

        return filtered_events

    def track_scoreboard(
        self,
        video_path: str | Path,
        scoreboard_roi: Tuple[float, float, float, float],  # normalized (x, y, w, h)
        sample_interval_sec: float = 1.0
    ) -> List[Dict[str, Any]]:
        """Samples frames at sample_interval_sec and extracts score changes using OCR."""
        try:
            import easyocr
        except ImportError:
            import subprocess
            import sys
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "easyocr"], check=True)
            import easyocr

        video_path = Path(video_path)
        if not video_path.exists():
            return []

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_step = max(1, int(fps * sample_interval_sec))

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        roi_x = int(scoreboard_roi[0] * width)
        roi_y = int(scoreboard_roi[1] * height)
        roi_w = int(scoreboard_roi[2] * width)
        roi_h = int(scoreboard_roi[3] * height)

        reader = easyocr.Reader(["en"], gpu=True, verbose=False)

        events: List[Dict[str, Any]] = []
        last_score_a: int | None = None
        last_score_b: int | None = None

        current_frame = 0
        while current_frame < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = cap.read()
            if not ret:
                break

            timestamp_sec = round(current_frame / fps, 2)
            crop = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
            if crop.size == 0:
                current_frame += frame_step
                continue

            # Crop the score digits column (rightmost ~22% of the scoreboard)
            crop_h, crop_w = crop.shape[:2]
            scores_col = crop[:, int(0.78 * crop_w):]

            # Upscale 3x for optimal CRAFT detector and CRNN recognition
            scores_up = cv2.resize(scores_col, (0, 0), fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(scores_up, cv2.COLOR_BGR2GRAY)

            res = reader.readtext(gray, allowlist="0123456789", detail=0)
            digits = [int(t) for t in res if t.isdigit()]

            if len(digits) >= 2:
                try:
                    score_a = digits[0]
                    score_b = digits[1]

                    # Basic sanity check: realistic basketball score deltas
                    if last_score_a is not None and last_score_b is not None:
                        delta_a = score_a - last_score_a
                        delta_b = score_b - last_score_b

                        if 1 <= delta_a <= 3 and delta_b == 0:
                            events.append({
                                "type": "SCORE_CHANGE",
                                "timestamp_sec": timestamp_sec,
                                "team": "HOME",
                                "points_delta": delta_a,
                                "score_home": score_a,
                                "score_away": score_b
                            })
                            last_score_a = score_a
                        elif 1 <= delta_b <= 3 and delta_a == 0:
                            events.append({
                                "type": "SCORE_CHANGE",
                                "timestamp_sec": timestamp_sec,
                                "team": "AWAY",
                                "points_delta": delta_b,
                                "score_home": score_a,
                                "score_away": score_b
                            })
                            last_score_b = score_b
                        elif score_a >= last_score_a and score_b >= last_score_b:
                            last_score_a = score_a
                            last_score_b = score_b
                    else:
                        last_score_a = score_a
                        last_score_b = score_b
                except ValueError:
                    pass

            current_frame += frame_step

        cap.release()
        return events

    def generate_timeline(
        self,
        video_path: str | Path,
        audio_path: str | Path,
        scoreboard_roi: Tuple[float, float, float, float]
    ) -> Path:
        """Combines audio and scoreboard tracking into timeline_events.json."""
        audio_events = self.detect_audio_spikes(audio_path)
        scoreboard_events = self.track_scoreboard(video_path, scoreboard_roi)

        timeline = sorted(audio_events + scoreboard_events, key=lambda e: e.get("start_sec", 0.0))
        output_file = self.output_dir / "timeline_events.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(timeline, f, indent=2)

        return output_file
