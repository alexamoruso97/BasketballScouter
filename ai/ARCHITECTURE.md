# System Architecture - Scouter Multi-Agent Framework

## 1. Vision and Objective
The Scouter framework employs an asynchronous, non-linear multi-agent approach based on **Retroactive Temporal Reasoning**:
Rather than executing dense action recognition across every individual frame sequentially, the system:
1. Detects macro events (score changes, referee whistles, crowd cheers).
2. Segments relevant temporal windows (offensive possessions).
3. Builds an appearance embedding map (Re-ID) of all active players across the match to resolve temporary occlusions and jersey visibility lapses retroactively.
4. Delegates to the Head Scout agent the analytical attribution of key actions (field goals, assists, rebounds, fouls).

```
                      +-----------------------------+
                      |        YouTube Video        |
                      +--------------+--------------+
                                     |
                                     v
                        [ Agent 1: Cameraman ]
                                     |
                    +----------------+----------------+
                    |                                 |
                    v (Audio & Scoreboard ROI)        v (Frames & Tracking)
        [ Agent 2: Scorekeeper ]              [ Agent 4: Profiler ]
                    |                                 |
             Timeline Events                  Re-ID & Player Maps
                    |                                 |
                    v                                 |
         [ Agent 3: Archivist ]                       |
                    |                                 |
         Possession Clips (10-15s)                    |
                    |                                 |
                    +----------------+----------------+
                                     |
                                     v
                        [ Agent 5: Head Scout ]
                                     |
                                     v
                          Boxscore & Web Report
```

## 2. Data Flow
- **Source:** YouTube video stream (720p/1080p at 30fps/60fps).
- **Audio Track:** Mono .wav file sampled at 16kHz for energy analysis (RMS / dB).
- **Event Metadata:** Shared JSON file (`timeline_events.json`) storing timestamps of score changes and active/passive play boundaries.
- **Re-ID Vectors:** Embedding database associated with provisional cluster IDs, finalized once jersey number OCR is confirmed.
- **Final Output:** Structured statistical boxscore (FIBA/NBA format) linked with clip timestamps of all critical moments.
