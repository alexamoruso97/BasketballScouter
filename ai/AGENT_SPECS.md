# Agent Specifications and Contracts - Scouter

Each agent is self-contained and exchanges structured data via standardized files (JSON, clip directories, numpy feature arrays).

---

### Agent 1: Cameraman (`src/agents/cameraman`)
* **Role:** Ingestion and pre-processing of video and audio streams.
* **Input:** YouTube video URL or local video path.
* **Responsibilities:**
  - Selective resolution download via `yt-dlp`.
  - Extraction of isolated mono audio stream (.wav) for energy and spike detection.
  - Video resolution, FPS, and frame count extraction.
* **Output:**
  - `data/raw_video.mp4`
  - `data/audio.wav`
  - `data/video_metadata.json` (`fps`, `width`, `height`, `duration_sec`, `total_frames`)

---

### Agent 2: Scorekeeper (`src/agents/scorekeeper`)
* **Role:** Game clock tracking, scoreboard OCR, and audio energy analysis (Context Analyzer).
* **Input:** `raw_video.mp4`, `audio.wav`, normalized ROI coordinates of the scoreboard overlay `(x, y, w, h)`.
* **Responsibilities:**
  - Subsampling scoreboard ROI frames (e.g. 1 frame/sec instead of 30 fps) for compute efficiency.
  - OCR extraction of Home/Away scores and game clock.
  - Audio energy profiling: detection of decibel peaks (scoring crowd bursts, horn, whistle).
  - Identification of active play (clock running) vs. passive play (clock stopped / timeout).
* **Output:**
  - `data/timeline_events.json`: ordered list of events with timestamps, score delta, and event type (`SCORE_CHANGE`, `AUDIO_SPIKE`, `TIME_STOP`).

---

### Agent 3: Archivist (`src/agents/archivist`)
* **Role:** Selective possession segmentation and highlight extraction (Possession Extractor).
* **Input:** `raw_video.mp4`, `timeline_events.json`.
* **Responsibilities:**
  - Computes the prior action window for each score change: `[t_score - lookback_sec, t_score + follow_through_sec]` (typically lookback = 12-15 seconds).
  - Executes lossless stream-copy video slicing via FFmpeg without re-encoding.
  - Flags clutch situations (final 2 minutes of fourth quarter, lead changes, buzzer beaters).
* **Output:**
  - Directory: `data/clips/possession_XXX.mp4`
  - `data/clips/manifest.json`: mapping of clip IDs, start/end timestamps, and expected score deltas.

---

### Agent 4: Profiler (`src/agents/profiler`)
* **Role:** Player detection, multi-object tracking, and retroactive Re-ID resolution.
* **Input:** Frames extracted from `raw_video.mp4` or possession clips.
* **Responsibilities:**
  - Player detection using YOLOv8 / YOLOv11.
  - Local multi-object tracking (ByteTrack) to preserve IDs across consecutive frames.
  - Team color clustering (automatic Home vs. Away separation).
  - Jersey number OCR on torso/back crops when unobstructed.
  - Deep Re-ID feature embedding extraction.
  - **Retroactive Resolution:** Once a jersey number is confirmed for a provisional track ID, retroactively updates all past and future appearances sharing the same feature vector.
* **Output:**
  - `data/player_identities.json`: mapping of `player_id -> jersey_number, team, temporal_bounding_boxes`.
  - Player on-court presence log (effective playing minutes).

---

### Agent 5: Head Scout (`src/agents/head_scout`)
* **Role:** Tactical action recognition, statistical attribution, and boxscore aggregation.
* **Input:** `possession_XXX.mp4`, `player_identities.json`, `timeline_events.json`.
* **Responsibilities:**
  - Ball trajectory analysis and hand-to-ball spatial proximity.
  - Shooter identification (player who released the ball immediately preceding the confirmed score event).
  - Assist detection (final passer preceding a made field goal within temporal limit).
  - Rebound tracking on missed field goal sequences.
  - Final boxscore computation (Points, 2PT, 3PT, FT, Rebounds, Assists, Turnovers, Fouls).
* **Output:**
  - `data/final_boxscore.json`
  - `data/report_summary.md`
  - Dashboard-ready metadata for web presentation (Streamlit / React).
