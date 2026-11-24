# Scouter - Automated Basketball Scouting & Analytics

Scouter is a multi-agent computer vision and analytics framework designed for automated scouting and statistics extraction from basketball match recordings (e.g. YouTube).
The system uses a non-linear, retroactive temporal architecture to handle occlusions and focus heavy compute only on relevant possession clips.

## 📁 Repository Structure

```text
Scouter/
├── docs/                      # Documentation assets, presentations (PDF, DOCX, PPT), court diagrams, and images
├── ai/                        # System architectures, agent specifications, prompts, and context markdown files
│   ├── ARCHITECTURE.md        # Technical architecture overview of the multi-agent pipeline
│   └── AGENT_SPECS.md         # Input/output contracts and responsibilities for each agent
├── notebooks/                 # Jupyter & Google Colab notebooks for prototyping and cloud execution
├── src/                       # Modular Python source code
│   ├── common/                # Shared utilities (video acquisition, audio extraction, clip slicing)
│   └── agents/
│       ├── cameraman/         # Agent 1: Video ingestion, resolution checks, and audio extraction
│       ├── scorekeeper/       # Agent 2: Scoreboard OCR tracking and audio energy peak detection
│       ├── archivist/         # Agent 3: Possession window slicing and key moment isolation
│       ├── profiler/          # Agent 4: Player detection, tracking, jersey OCR, and retroactive Re-ID
│       └── head_scout/        # Agent 5: Action attribution (points, assists, rebounds) and boxscore generation
├── run_pipeline.py            # Automated end-to-end execution script (Colab / CLI)
├── requirements.txt           # Python dependencies
└── .gitignore                 # Exclusion rules for heavy media, checkpoints, and cache
```

## 🚀 Google Colab Execution (Single Cell)
Computer Vision models (YOLO, OCR, Re-ID) require GPU acceleration.
Mount Google Drive and run the automated pipeline in one command:

```python
%cd /content/drive/MyDrive/Scouter
!pip install -q -U yt-dlp yt-dlp-youtube-oauth2 opencv-python ffmpeg-python ultralytics easyocr librosa pydub matplotlib
!python run_pipeline.py
```
