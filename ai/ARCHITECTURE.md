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

## 3. Multi-Match & Multi-Angle Generalization Strategy

The system is fundamentally designed to generalize across arbitrary video sources (amateur gym cameras, broadcast TV, single-cam tactical wide views) without hardcoded assumptions:

### A. Adaptive Score & Possession Detection (Dual Mode)
1. **Broadcast HUD Mode (TV / Live streams):**
   - Automatically detects or accepts configuration for arbitrary scoreboard layouts (top-left, bottom-left, banner at bottom, corner bugs).
   - In future iterations, an object detection model (`yolo_scoreboard.pt`) locates the scoreboard widget anywhere in the frame without manual calibration.
2. **Visual Ball-in-Hoop Mode (Amateur footage without on-screen graphics):**
   - When no digital scoreboard overlay exists (e.g. gym tripod recording):
     - Detects the two baskets/rims.
     - Tracks ball trajectory entering the rim cylinder from above.
     - Detects net deformation/vibration to confirm a made basket independently of any scoreboard.

### B. Camera Angle Invariance (Court Homography & 2D Projection)
- Different matches use different camera elevations (high tactical sideline, low angle, corner view).
- Court line keypoints (mid-court line, key area, 3-point arcs, baseline) are registered to compute a homography matrix H.
- Projected player coordinates are transformed into a canonical top-down 2D court model (28m x 15m FIBA metric space), rendering spatial calculations (2PT vs. 3PT, paint presence, spacing) invariant to camera zoom, tilt, and pan.

### C. Multi-Angle Player Profiling (Re-ID Embeddings)
- Players constantly turn, face away, or get occluded by teammates and defenders.
- Deep metric learning embeddings (OSNet / FastReID) extract multi-view visual signatures (torso aspect ratio, uniform color histogram, footwear, skin tone, accessories) that recognize a player consistently from front, side, or back views even when the jersey number is temporarily out of view.
