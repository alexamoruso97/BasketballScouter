"""Profiler Agent: Player detection, tracking, team clustering, and retroactive jersey Re-ID."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np


ROSTER_HOME = {
    0: "CORDINIER Isaia",
    1: "LUNDBERG Gabriel",
    3: "BELINELLI Marco",
    6: "PAJOLA Alessandro",
    13: "DOBRIC Ognjen",
    14: "MASCOLO Bruno",
    21: "SHENGELIA Tornike",
    23: "HACKETT Daniel",
    33: "POLONARA Achille",
    41: "ZIZIC Ante",
    42: "DUNSTON Bryant",
    55: "ABASS Awudu"
}

ROSTER_AWAY = {
    3: "BORTOLANI Giordano",
    7: "TONUT Stefano",
    9: "MELLI Nicolo",
    13: "NAPIER Shabazz",
    17: "RICCI Giampaolo",
    21: "FLACCADORI Diego",
    22: "HALL Devon",
    30: "CARUSO Guglielmo",
    31: "SHIELDS Shavon",
    33: "MIROTIC Nikola",
    42: "HINES Kyle",
    77: "VOIGTMANN Johannes"
}


class ProfilerAgent:
    """Detects, tracks, and profiles basketball players in possession action clips."""

    def __init__(self, output_dir: str | Path, model_name: str = "yolo11n.pt"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.profiles_dir = self.output_dir / "profiles"
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self._model = None
        self._ocr_reader = None

    def _get_model(self):
        """Lazy loads YOLO tracking model."""
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self.model_name)
        return self._model

    def _get_ocr_reader(self):
        """Lazy loads EasyOCR reader with GPU acceleration when available."""
        if self._ocr_reader is None:
            import easyocr
            import torch
            use_gpu = torch.cuda.is_available()
            self._ocr_reader = easyocr.Reader(["en"], gpu=use_gpu, verbose=False)
        return self._ocr_reader

    def profile_clip(
        self,
        clip_path: str | Path,
        clip_id: str,
        sample_step: int = 2,
        min_box_height: int = 60
    ) -> Dict[str, Any]:
        """Profiles all players in a possession clip and resolves jersey numbers retroactively."""
        clip_path = Path(clip_path)
        if not clip_path.exists():
            raise FileNotFoundError(f"Clip not found: {clip_path}")

        model = self._get_model()
        reader = self._get_ocr_reader()

        cap = cv2.VideoCapture(str(clip_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / fps if fps > 0 else 0.0

        # Run tracking stream across clip frames
        tracks_data: Dict[int, Dict[str, Any]] = {}

        results = model.track(
            source=str(clip_path),
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0],  # Person class
            stream=True,
            verbose=False
        )

        frame_idx = 0
        for r in results:
            frame_idx += 1
            if frame_idx % sample_step != 0:
                continue

            frame_img = r.orig_img
            if frame_img is None or r.boxes is None:
                continue

            timestamp_sec = round(frame_idx / fps, 2)

            for box in r.boxes:
                if box.id is None:
                    continue

                track_id = int(box.id[0])
                conf = float(box.conf[0])
                xyxy = [int(c) for c in box.xyxy[0]]
                bx1, by1, bx2, by2 = xyxy
                bw = bx2 - bx1
                bh = by2 - by1

                # Discard spectators/bench outside active play height threshold
                if bh < min_box_height or conf < 0.40:
                    continue

                if track_id not in tracks_data:
                    tracks_data[track_id] = {
                        "track_id": track_id,
                        "appearances": 0,
                        "boxes": [],
                        "torso_brightness": [],
                        "ocr_numbers": [],
                        "team": "UNKNOWN",
                        "jersey_number": None,
                        "player_name": "UNKNOWN"
                    }

                tracks_data[track_id]["appearances"] += 1
                tracks_data[track_id]["boxes"].append({
                    "frame": frame_idx,
                    "timestamp_sec": timestamp_sec,
                    "xyxy": xyxy,
                    "conf": round(conf, 2)
                })

                # Crop upper torso for jersey color and number recognition
                pad_y1 = by1 + int(bh * 0.15)
                pad_y2 = by1 + int(bh * 0.55)
                pad_x1 = bx1 + int(bw * 0.15)
                pad_x2 = bx2 - int(bw * 0.15)

                if pad_y2 > pad_y1 and pad_x2 > pad_x1:
                    torso = frame_img[pad_y1:pad_y2, pad_x1:pad_x2]
                    if torso.size > 0:
                        gray_torso = cv2.cvtColor(torso, cv2.COLOR_BGR2GRAY)
                        brightness = float(np.mean(gray_torso))
                        tracks_data[track_id]["torso_brightness"].append(brightness)

                        # Sample OCR on reasonably large torsos
                        if bh >= 80 and frame_idx % (sample_step * 2) == 0:
                            ocr_crop = cv2.resize(gray_torso, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
                            ocr_res = reader.readtext(ocr_crop, allowlist="0123456789", detail=0)
                            for token in ocr_res:
                                if token.isdigit() and len(token) <= 2:
                                    tracks_data[track_id]["ocr_numbers"].append(int(token))

        cap.release()

        # Step 2: Classify Team based on median brightness clustering
        # Virtus Bologna = Dark/Black (lower brightness), Olimpia Milano = Light/White (higher brightness)
        active_tracks = [t for t in tracks_data.values() if t["appearances"] >= 5]
        if active_tracks:
            all_brightness = [
                float(np.mean(t["torso_brightness"])) for t in active_tracks if t["torso_brightness"]
            ]
            if all_brightness:
                split_threshold = float(np.median(all_brightness))
                for t in active_tracks:
                    mean_b = float(np.mean(t["torso_brightness"])) if t["torso_brightness"] else split_threshold
                    t["team"] = "AWAY" if mean_b >= split_threshold else "HOME"

        # Step 3: Retroactive Identity Resolution via Roster Consensus
        for t in active_tracks:
            team = t["team"]
            valid_roster = ROSTER_HOME if team == "HOME" else ROSTER_AWAY

            if t["ocr_numbers"]:
                # Select most frequent valid jersey number detected across all frames
                candidates = [num for num in t["ocr_numbers"] if num in valid_roster]
                if candidates:
                    most_common_num = Counter(candidates).most_common(1)[0][0]
                    t["jersey_number"] = most_common_num
                    t["player_name"] = valid_roster.get(most_common_num, f"Player #{most_common_num}")

        profile_summary = {
            "clip_id": clip_id,
            "clip_file": clip_path.name,
            "clip_path": str(clip_path),
            "duration_sec": round(duration_sec, 2),
            "total_players_tracked": len(active_tracks),
            "home_players": [
                {"track_id": t["track_id"], "jersey": t["jersey_number"], "name": t["player_name"]}
                for t in active_tracks if t["team"] == "HOME"
            ],
            "away_players": [
                {"track_id": t["track_id"], "jersey": t["jersey_number"], "name": t["player_name"]}
                for t in active_tracks if t["team"] == "AWAY"
            ],
            "tracks": active_tracks
        }

        # Save profile manifest for this possession clip
        profile_file = self.profiles_dir / f"{clip_id}_profile.json"
        with open(profile_file, "w", encoding="utf-8") as f:
            json.dump(profile_summary, f, indent=2)

        return profile_summary

    def profile_all_clips(
        self,
        clips_dir: str | Path,
        manifest_path: str | Path,
        max_clips: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Batches player profiling across all extracted possession clips."""
        manifest_path = Path(manifest_path)
        if not manifest_path.exists():
            return []

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        if max_clips:
            manifest = manifest[:max_clips]

        summaries = []
        for item in manifest:
            clip_file = item.get("clip_file")
            clip_id = item.get("clip_id")
            clip_path = Path(clips_dir) / clip_file

            if clip_path.exists():
                summary = self.profile_clip(clip_path=clip_path, clip_id=clip_id)
                summaries.append(summary)

        # Write overall profiling registry
        registry_file = self.output_dir / "all_clips_profiles.json"
        with open(registry_file, "w", encoding="utf-8") as f:
            json.dump(summaries, f, indent=2)

        return summaries
