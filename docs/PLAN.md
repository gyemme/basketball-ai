# Youth Basketball Video Analytics Pipeline — Plan

**Goal:** Drop in an XbotGo game recording and get back per-player stats, a shot chart (where each shot came from, make or miss), and coaching insights that help each player improve.

**Constraints**
- **Source:** XbotGo recordings, either Chameleon (phone on an AI gimbal) or Falcon (all-in-one 4K camera)
- **Camera behavior:** auto-tracking. The camera **pans (and may zoom) to follow the play**, and can burn a **ScoreSync scoreboard overlay** into the video.
- Players identified by jersey number, mapped to a roster (no face recognition)
- Runs locally on a Mac with Apple Silicon
- Minimal cost, using open-source models and free tools

> **Status (v1 built):** `hoopstats` CLI in this repo. See README.md for usage. Differences from this plan:
> - Default detector is YOLO11 (Ultralytics, AGPL-3.0). RF-DETR (Apache-2.0) is available with `--detector rfdetr`.
> - Court registration uses clicked keyframes + SIFT/optical flow. A court-keypoint model is a phase-2 upgrade.
> - The rim is projected from the recovered camera pose, so no rim detector is needed.
> - ScoreSync digits are read with templates harvested from a real XbotGo Falcon recording.

---

## 1. What you get per game

The **main deliverable is a single interactive HTML file** (`report.html`) holding all the details, with an interactive full-court shot chart at its center (section 1.1). Everything else is supporting data.

| Output | Description |
|---|---|
| **`report.html`** | **Self-contained interactive report:** full-court shot chart with player selection, player stats, zone breakdowns, heatmaps, insights, team summary. One file that opens in any browser, works offline, and can be emailed or shared. |
| `player_stats.csv` | One row per player (see section 6) |
| `shots.csv` | Every shot: time, shooter, court x/y, distance, zone, make/miss, assisted, video timestamp |
| `events.csv` | Possessions, passes, rebounds, possession losses |
| `coach_summary.json` | Compact summary to hand to Claude (or a local LLM) for written coaching plans |
| `annotated.mp4` | Optional overlay video (boxes, names, ball, court registration, MAKE/MISS banners) for checking accuracy |
| `clips/` | Optional short clips per shot, cut from the original XbotGo file |

### 1.1 The interactive HTML report

**Layout**

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  Hawks 42 – Lions 38 · Oct 18, 2026 · Lincoln MS        [Team ▾] [Period ▾]  │
├───────────────────┬──────────────────────────────────────────────────────────┤
│ PLAYERS           │               FULL-COURT SHOT CHART                      │
│ ☑ All             │  ┌────────────────────────┬────────────────────────┐     │
│ ☐ #3  Ava         │  │  ○   ●            ●    │     ×      ●     ○     │     │
│ ☑ #12 Maya  ◀──── │  │     ×  (●)            │        ●       ×       │     │
│ ☐ #21 Priya       │  │  ●       ○     ×      │   ○          ●         │     │
│ ☐ #5  Zoe         │  └────────────────────────┴────────────────────────┘     │
│ ...               │   ● make   × miss   size/color = player   (hover = info) │
│                   │   [Makes ☑] [Misses ☑] [Zone ▾] [2PT/3PT/FT ▾]           │
├───────────────────┴──────────────────────────────────────────────────────────┤
│ SELECTED: #12 Maya — 6/13 FG (46%) · 2/5 3PT · 14 pts · 3 reb · 2 ast         │
│ Zone table · Area FG% (left corner … right corner) · Heatmap · Insights      │
├──────────────────────────────────────────────────────────────────────────────┤
│ Full box score (sortable) · Team comparison · Unidentified-tracks gallery    │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Full-court shot chart**
- Drawn as a scaled SVG of the **entire court** (both halves, lanes, 3-point arcs, center circle), using your league's court size from the config.
- Every shot is plotted at the **actual court spot it was taken from**. Shots are not folded onto one half, so you can see which basket each team attacked in each half.
- **Make** = filled circle, **miss** = ×, color-coded by player (or by team when "All" is selected).
- **Player names on the court:** each selected player's shots are labeled with their jersey number, and a legend lists number + name.

**Player selection**
- The sidebar lists every rostered player (number + name), grouped by team.
- **Click a name** to show only that player's shots, or select several to compare. "All" shows everyone.
- Selecting a player also updates the stat panel below the court: line, zone and area FG%, heatmap, and insights.
- **Click a player's shot dot** on the court to select that player.

**Shot details on hover or click**
- Player, period and game clock (from the scoreboard overlay when available), video timestamp
- Distance (ft), zone, 2PT/3PT/FT, make/miss
- Catch-and-shoot or off-dribble, assisted by whom
- **"Watch" link** that opens the shot clip, or jumps to that moment in the original XbotGo video if it sits next to the report

**Filters:** team, period, makes/misses, zone, shot type (2PT/3PT/FT).

**How it's built**
- One `.html` file with inline SVG, CSS and plain JavaScript. All game data is embedded as JSON, and nothing loads from the internet, so it opens offline and is easy to share.
- Charts in the stat panel (zone bars, movement heatmap) render client-side from the same embedded data.
- Readable on phones too, so parents and players can open it.
- **Season view (phase 4):** the same page format with a game picker, so you can see a player's shots across all games.

---

## 2. Architecture

```
 XbotGo file(s) (.mp4 / .mov, H.264 or HEVC)
    │
    ▼
 [0] Ingest ──► probe, fix rotation / variable frame rate, join split files,
    │           find & mask the scoreboard overlay (optionally read score + clock)
    ▼
 [1] Detect ──► players, ball, rim, jersey-number boxes          (RF-DETR, fine-tuned)
    │
    ▼
 [2] Track ───► stable track IDs across frames                   (ByteTrack via supervision)
    │
    ▼
 [3] Register ► per-frame court homography for a MOVING camera
    │           (court-keypoint model + frame-to-frame camera-motion estimate, fused & smoothed)
    ▼
 [4] Identify ► team (jersey color / embeddings) + number (OCR) → roster name
    │
    ▼
 [5] Map ─────► each player's foot point → court coordinates (ft) using that frame's homography
    │
    ▼
 [6] Events ──► possession · shot attempt · make/miss · shooter · assist · rebound · turnover
    │
    ▼
 [7] Stats ───► box score, zone FG%, movement, spacing, defensive distance
    │
    ▼
 [8] Insights ► rule-based findings + optional LLM-written coaching notes
    │
    ▼
 [9] Report ──► interactive report.html (full-court shot chart + player selector), CSVs, annotated video, clips
```

Stages 0–4 are the expensive part and are cached to disk. Stages 5–9 can be re-run in seconds after you fix a jersey ID or change a setting.

---

## 3. XbotGo-specific handling

### 3.1 Ingest (stage 0)
XbotGo files come out of the XbotGo app or cloud, the phone's camera roll (Chameleon), or the Falcon's storage. The ingest step makes all of them uniform.

| Issue | Handling |
|---|---|
| **Container and codec:** `.mp4` or `.mov`, H.264 or **HEVC (H.265)**, which iPhones record by default | `ffprobe` inspects every file. HEVC is decoded with the Mac's hardware decoder, and is transcoded only if OpenCV can't read it. |
| **Variable frame rate**, common on phone recordings | Detected by comparing nominal and average frame rate. If variable, convert to a constant 30 fps with `ffmpeg` (hardware `h264_videotoolbox`, fast). A frame→original-timestamp map is kept so clips and report links point to the right moment. |
| **Rotation metadata,** for portrait-mounted phones or flipped mounts | Applied during normalization. Portrait footage is flagged, since landscape is strongly recommended. |
| **4K (Falcon) vs 1080p** | Analyze at native resolution for ball and number detail, or downscale for speed. This is configurable. |
| **Game split across files** (quarters, restarts, storage limits) | Pass a folder or list. Files are ordered by creation time and joined on one game timeline, with per-file offsets kept. |
| **ScoreSync scoreboard overlay** burned into the frame | Auto-detected as the one region that stays pixel-static while the camera pans. It is masked out so it isn't detected as players or court. **Optional bonus:** OCR the overlay to read score and clock, which splits the game into periods and lets us check our points against the real score. |
| **Highlight clips** exported by the XbotGo app | Not used for stats. Use the **full game recording**, because highlights are missing possessions and misses. |

### 3.2 Moving-camera court registration (stage 3)
A one-time click calibration only works for a fixed camera. XbotGo pans and zooms, so the image-to-court mapping changes every frame.

1. **Court keypoint model.** Detect court landmarks in each frame: corners, lane corners, free-throw line ends, half-court line, 3-point line ends. Any 4+ visible keypoints give that frame's homography.
2. **Camera-motion estimate.** Between consecutive frames, track background features (floor lines, walls), ignoring players, the ball and the scoreboard overlay. This gives the frame-to-frame motion, which carries the court mapping through frames where few keypoints are visible, such as when zoomed on one end.
3. **Fusion and smoothing.** Combine both, reject outliers (RANSAC), and smooth over time. Each frame gets a **registration confidence score**.
4. **Quality gate.** Frames with low confidence are excluded from location-based stats (shot spots, distance covered). They still count for events like makes and misses. The report shows the share of the game that was well registered.
5. **Manual fallback.** If a gym's floor confuses the model (busy multi-sport lines), you click landmarks on a few keyframes per game, and camera-motion tracking carries them forward.

### 3.3 Other effects of a tracking camera
- **Rims move in the image.** The rim position comes from the detector's `rim` class each frame, not from clicks. Shot logic is measured in rim-widths, so it still works when the camera zooms.
- **Not all 10 players are always in frame.** The camera follows the ball, so off-ball players at the far end can drop out. Movement and spacing stats are computed only while a player is visible and well registered, and the report shows each player's *tracked minutes*.
- **Motion blur during fast pans** lowers detection and number-read quality. Multi-frame voting and gap interpolation absorb most of it.
- **Avoid XbotGo's "Follow Me" (single jersey) mode for game stats.** It frames one player and loses the rest of the court. Use the standard full-court basketball game mode.

---

## 4. Recommended open-source models

All are free. Licenses are permissive (Apache 2.0 or MIT) unless noted.

| Task | Primary pick | Alternative | Notes |
|---|---|---|---|
| **Player / ball / rim / number detection** | **RF-DETR (Small or Nano)**, starting from Roboflow's open basketball model (classes include player, ball, rim, number, referee, jump-shot, layup, ball-in-basket) | YOLO11 (Ultralytics) | RF-DETR is Apache 2.0. Ultralytics is AGPL-3.0, which is fine for personal or club use but restrictive if you ever sell it. It was trained on NBA footage, so fine-tune on your XbotGo footage (section 7). |
| **Court keypoints** (needed because the camera moves) | Keypoint model trained on **Roboflow's basketball court keypoint dataset**, e.g. **RTMPose-style head via MMPose** (Apache 2.0) | **YOLO11-pose** (AGPL-3.0, easiest to train) | Fine-tune with frames from your gyms. Youth courts and multi-sport floors vary. |
| **Camera motion** | **OpenCV** feature tracking (ORB/SIFT + RANSAC homography, or sparse optical flow) with players and overlay masked out | — | No model needed. Fast and free. |
| **Tracking** | **ByteTrack** (via `supervision`) with camera-motion compensation | BoT-SORT (built-in camera-motion compensation); **SAM2** for hard clips | Compensating for pans keeps track IDs stable while the camera moves. |
| **Team assignment** | **Jersey color clustering** (k-means on torso pixels) | **SigLIP** image embeddings + UMAP + k-means | Start with color. Switch to SigLIP if the colors are similar or gym lighting shifts. |
| **Jersey number reading** | **SmolVLM2** prompted for digits | **PaddleOCR**, or a small classifier trained on Roboflow's jersey-number OCR dataset | Vote across many frames per track, and accept only numbers on that team's roster. |
| **Scoreboard overlay reading (optional)** | **PaddleOCR** on the masked overlay region | SmolVLM2 | Reads score and clock to check our points and split the game into periods. |
| **Shot make/miss** | Ball-trajectory rules around the detected rim + the `ball-in-basket` class | — | Handles rim bounces, the net hiding the ball, and air balls passing behind the rim. |
| **Pose (phase 5, optional)** | **RTMPose** (MMPose) | MediaPipe Pose, YOLO11-pose | Shooting form: release point, elbow angle, jumper vs. layup. |
| **Coaching notes** | **Claude** (upload `coach_summary.json`) | Local LLM via **Ollama** (Qwen or Llama 8B-class) for $0 | Optional writing layer. The stats don't depend on it. |

**Supporting tools (all free):** `ffmpeg` / `ffprobe`, `supervision`, OpenCV, PyTorch with the Apple MPS backend, pandas, matplotlib.

---

## 4b. Data sources: video only vs. video + GameChanger

**Rule: every stat must be producible from the video alone.** GameChanger is an optional second input that improves accuracy and removes manual checking. It is never required.

### Scenario A — video only (default)
| Stat | Comes from | Confidence |
|---|---|---|
| Made baskets + point value, by team | ScoreSync overlay score changes (OCR), confirmed by ball-through-rim | High |
| Shot attempts and misses | Ball trajectory near the rim | Medium, improves with fine-tuning |
| Shooter | Last ball holder before release + jersey OCR | Medium; manual fixes via `overrides.yaml` |
| Shot location | Shooter's feet at release → court homography | Medium-high when registration is good |
| Rebounds, assists, steals, turnovers | Possession changes | Estimates |
| Movement, spacing, touch time, heatmaps | Tracking + homography | Video only, no other source has these |

### Scenario B — video + GameChanger timeline
GameChanger's play-by-play is scorer-entered, so it is the most accurate record of *who* did *what*. It has no *where* and no movement. The two sources complement each other:

| GameChanger provides (trusted) | Video adds |
|---|---|
| Shooter, make/miss, 2/3/FT, rebounds, assists, steals, turnovers, fouls | Shot location (x, y), shot type (catch vs dribble), touch time, movement, spacing, defensive distance, clips |

**How the fusion works**
1. **Time alignment.** XbotGo burns the wall clock into the bottom-left of the frame (e.g. `11:31:37`). GameChanger events also carry timestamps, so OCR of that clock maps each GameChanger event to a video time. Score changes (overlay vs GameChanger running score) fine-tune the offset and catch drift.
2. **Event matching.** For each GameChanger shot event, search the video ±8 s for a ball release whose shooter matches the jersey number. Use the video for location and type, and GameChanger for player and result.
3. **Conflicts.** GameChanger wins on who and result. The video wins on where. Unmatched events are listed in the report's data-quality section.
4. **Free accuracy checks.** `evaluate.py` compares video-only output against GameChanger automatically, so no hand-charting is needed.
5. **Free training labels.** Matched events become labeled examples (shooter crops, jersey numbers, make/miss clips) for fine-tuning in phase 2.

**Getting the data out.** GameChanger has no public API.
- Box scores and season stats can be exported from the app or web.
- The play-by-play timeline is visible in the app and on web.gc.com but not exportable as a file.
- The web app uses private endpoints tied to your login. Those can change without notice, and scraping them may break GameChanger's terms.

So the pipeline reads GameChanger data through a small **import adapter** (`hoopstats/sources/gamechanger.py`) that accepts a file:
- a JSON or CSV of play-by-play, however it was obtained
- or just the box score

When only the box score is available, it is still used as the source of truth for totals, and as a per-player check on the video counts.

**Pipeline behavior:** `--gc path/to/game.json` turns on fusion mode. Without it, scenario A runs unchanged.

---

## 5. Event detection

| Event | Logic |
|---|---|
| **Possession** | The ball sits inside a player's box for at least 0.2 s. Short flickers (passes flying past) are ignored. |
| **Shot attempt** | A free ball rises above the detected rim and comes down near it. Frames where a player holds the ball are excluded, which stops perspective fakes. |
| **Make / miss** | Where the ball crosses rim height relative to rim width, plus a `ball-in-basket` detection. A bounce on the rim that drops through counts as one make. |
| **Shooter and location** | The last player to hold the ball before release. Their foot point at release, through that frame's homography, gives the shot spot (only if registration confidence is high). |
| **Shot type** | Catch-and-shoot (held < 1 s) vs. off-dribble. Layup or jumper from detector classes, later from pose. |
| **Assist (est.)** | A made shot where the shooter received a teammate's pass within ~1.5 s and shot within ~3 s. |
| **Rebound** | The first possession after a miss. Offensive or defensive by team. |
| **Possession lost / won** | Possession switches teams with no shot in between (steals and turnovers). |
| **Free throws** | Shooter at the FT line, nobody within ~6 ft. Counted separately. Cross-checked against score changes on the overlay when available. |

---

## 6. Stats and "player intelligence"

**Box score:** FGA, FGM, FG%, 2PT and 3PT splits, estimated points, rebounds (O/D), estimated assists, touches, possession losses and wins.

**Shooting profile:**
- FG% by zone: at rim (< 6 ft), paint, mid-range, three
- FG% by area: left corner, left wing, middle, right wing, right corner
- Shot distribution
- Average shot distance
- Catch-and-shoot vs. off-dribble FG%
- Left vs. right tendency

**Movement and effort** (during tracked minutes):
- Distance per minute
- Average and top speed
- Time in the paint

**Spacing and defense:**
- Offense: distance to the nearest teammate (crowding) and to the nearest defender (how open they get)
- Defense: distance to the nearest opponent (pressure)

**Ball handling:** average touch time, possession-loss rate per touch.

**Auto-generated insights (each backed by numbers):**
- "48% at the rim but 22% from mid-range, and mid-range is 45% of attempts → attack the basket more."
- "78% of shots from the right side → develop left-side finishes."
- "Averages 4.1 s per touch → quicker decisions."
- "Often within 7 ft of a teammate on offense → spacing drills."
- "Covers 30% less distance per minute than the team → conditioning and off-ball movement."
- Team level: hot and cold zones, assisted share of made shots, offensive rebound rate.

Estimated stats are labeled as estimates. Use them for trends over a season, not as an official box score. When the scoreboard overlay is readable, the report shows **our points vs. the real score** as a built-in accuracy check.

---

## 7. Fine-tuning on your XbotGo footage

This is the biggest accuracy lever, and it's free.

1. **Start** from the open basketball detection model and the court keypoint dataset on Roboflow Universe.
2. **Label frames** from 3–5 of your XbotGo games:
   - Detection: ~300–500 frames with player, ball, rim, number and referee. Include fast pans (motion blur), zoomed views and pile-ups.
   - Court keypoints: ~150–300 frames across your usual gyms and camera angles.
   - Tools: **CVAT** or **Label Studio** (open source, local), or Roboflow's free tier. Pre-label with the base models and only correct their output.
3. **Train** on a free **Google Colab** GPU, about 1–3 hours per model, or slowly on the Mac.
4. **Optional:** fine-tune the number reader on 1–2k jersey crops from your own games.

---

## 8. XbotGo filming guidelines

These matter as much as the models.
- Use **basketball game mode with full-court tracking**, not Follow Me or half-court mode, for game stats.
- **Landscape orientation**, highest resolution (**4K on Falcon**, or 4K/1080p on the phone), **30 fps**.
- Mount **high** (top of the bleachers or a tall tripod) near **mid-court**. Higher means less occlusion and more visible court lines for registration.
- If XbotGo lets you, cap the auto-zoom or keep it wide. Wider framing means more players and court keypoints in view.
- **ScoreSync overlay:** fine to leave on, since the pipeline masks it and can even read it. Keep it in a corner, not over the court.
- Record the **whole game continuously** where possible. If it splits into files, keep them together in one folder.
- Keep a roster with jersey colors for every game, including the opponent's if you want their stats.

---

## 9. Accuracy validation

Hand-chart **one full game** (shots, makes, shooter, rebounds) and compare.

| Metric | Initial target | After fine-tune |
|---|---|---|
| Shot attempts detected | ≥ 85% | ≥ 95% |
| Make/miss correct (of detected) | ≥ 90% | ≥ 97% |
| Shooter correctly identified | ≥ 70% | ≥ 90% (with manual ID fixes) |
| Frames with good court registration | ≥ 80% | ≥ 95% |
| Shot location error | ≤ 4 ft | ≤ 2 ft |
| Points vs. scoreboard overlay | within 15% | within 5% |

`evaluate.py` compares `shots.csv` to your hand-charted CSV and prints these numbers.

---

## 10. Project layout and usage

```
hoopstats/
  run.py                # CLI: ingest | detect | analyze | annotate | clips | all
  config.yaml           # teams, colors, rosters, court size, thresholds, overlay settings
  overrides.yaml        # manual track → player fixes
  calibrate.py          # optional manual keyframe calibration (fallback)
  hoopstats/
    ingest.py           # XbotGo probing, VFR/rotation fixes, file joining, overlay mask
    detect.py  track.py  register.py  identify.py  court.py
    events.py  stats.py  insights.py  scoreboard.py  annotate.py
    report/
      build.py          # embeds game data JSON into the HTML template
      template.html     # single-file interactive report (SVG court, player selector, filters)
  evaluate.py           # accuracy check vs. hand-charted game
  tests/                # synthetic-game tests (incl. simulated camera pans)
```

```bash
# single XbotGo file, or a folder of split files
python run.py all --video ~/Movies/XbotGo/2026-10-18_game.mp4 \
                  --config config.yaml --out runs/game1
open runs/game1/report.html
```

**Config highlights:** court length (youth courts vary, 74–94 ft), 3-point distance (or none), team colors, roster numbers → names, overlay on/off/auto.

---

## 11. Build phases

| Phase | Scope | Result |
|---|---|---|
| **1. Core pipeline** | XbotGo ingest, detection (base RF-DETR), ByteTrack with camera-motion compensation, per-frame court registration (base keypoint model + motion tracking), color teams, number OCR, shots and make/miss, **interactive report.html with full-court shot chart, player selector and shot hover details**, overrides | First usable report from a real XbotGo game |
| **2. Accuracy** | Label your frames, fine-tune detector, court keypoints and number reader; `evaluate.py`; tune thresholds | Numbers you can trust |
| **3. Intelligence** | Assists, rebounds, possession losses, spacing and defensive distance, insights engine, scoreboard OCR cross-check, `coach_summary.json`, per-shot clips | Coaching-grade insights |
| **4. Season view** | Combine games per player: trends and progress over time | Development tracking |
| **5. Form analysis (optional)** | RTMPose shooting mechanics | Shot-form feedback |

---

## 12. Cost

| Item | Cost |
|---|---|
| Models (RF-DETR, court keypoints, ByteTrack, SmolVLM2 / PaddleOCR, SigLIP, RTMPose) | $0 (open source) |
| Inference | $0 on your Mac. Expect roughly 1–2.5× real time for a full game. Court registration adds cost compared with a fixed camera. |
| Labeling | $0 with CVAT / Label Studio, or Roboflow's free tier |
| Training | $0 on free Colab GPU |
| Coaching write-ups | $0 with a local LLM, or use Claude |

---

## 13. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Court registration drifts during fast pans or zooms | Fuse keypoints + camera motion, re-anchor whenever 4+ keypoints are visible, confidence gate, manual keyframe fallback |
| Busy multi-sport floor lines confuse the keypoint model | Fine-tune on your gyms. Per-gym keypoint templates. |
| Players out of frame (camera follows the ball) | Report tracked minutes. Compute per-minute rates only over visible time. |
| Ball small or blurry, especially in pans | High-res analysis, fine-tune on XbotGo footage, gap interpolation |
| Jersey numbers unreadable | Multi-frame voting, roster constraint, thumbnail gallery with manual overrides |
| Track IDs break during pans | Camera-motion-compensated tracking; merge by team + number |
| Scoreboard overlay detected as objects | Auto-masked as the static region in a panning video |
| HEVC or variable frame rate files misread | `ffprobe` checks, normalize with ffmpeg, keep timestamp map |

---

## 14. Privacy (youth players)

- No face recognition. Identity comes only from jersey number + roster.
- Everything runs locally, and footage downloaded from XbotGo stays on your Mac.
- Get parents' consent before sharing reports or clips.

---

## 15. Decisions and inputs needed before the build

1. **A sample XbotGo file**, even a 1–2 minute clip. This confirms the codec, resolution, frame rate, overlay position and how much the camera zooms. Before we start, you can also run `ffprobe yourfile.mp4` and paste the output.
2. **Which XbotGo:** Chameleon (with which phone) or Falcon? And do you use ScoreSync?
3. **Court size and 3-point line** for your league (or "varies by gym").
4. **Opponent stats too,** or only your team?
5. **License preference:** Apache-only stack (RF-DETR + MMPose keypoints, recommended) vs. allowing Ultralytics YOLO (AGPL-3.0).
6. **Labeling:** willing to label ~300 detection frames + ~150 court-keypoint frames for phase 2?
