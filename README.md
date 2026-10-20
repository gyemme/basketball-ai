# hoopstats

Basketball stats, shot charts and coaching insights from XbotGo game video, using open-source models on your Mac.

```
hoopstats analyze "videos/18:00:21.MP4" --open
```

This writes `runs/18-00-21/report.html`, an interactive report you can open in any browser:
- a full-court shot chart, filterable by player, half, makes or misses, and zone
- a stat breakdown for each player, with a heatmap and coaching insights
- a box score
- a ▶ Watch link on every shot that jumps to that moment in the video

It also writes CSV files and `coach_summary.json` (described below).

## Setup (once)

```bash
cd ~/dev/basketball-ai
bash setup.sh
source .venv/bin/activate
```

The first `analyze` downloads the YOLO11 weights (~40 MB). Everything after that runs locally.

## Analysing a game

```bash
hoopstats score   "videos/18:00:21.MP4"
hoopstats analyze "videos/18:00:21.MP4" --open
```

On the first run of a game, `analyze`:
1. Creates `runs/<video>/game.yaml`. Put team names (left team = left side of the XbotGo score overlay), jersey colours and rosters there. It works without them, but names and rosters make the report much better.
2. Offers to **calibrate the court** (about 3 minutes, once per gym and camera spot; see below). Without calibration, makes and points still work, but shot locations are only approximate.
3. Detects and tracks players and the ball. This is the slow step: roughly 0.5–1× the length of the video on an Apple Silicon Mac at 10 analysed frames per second. Results are cached in `runs/<video>/cache/`.
4. Builds events, stats, insights and the report. This takes seconds.

Useful options:

| Option | What it does |
|---|---|
| `--start 5:00 --end 12:00` | analyse only part of the video (good for a first test) |
| `--fps 5` | faster, slightly less accurate |
| `--model yolo11l.pt` | bigger detector: better at finding the ball, slower |
| `--detector rfdetr` | use RF-DETR (Apache-2.0) instead of YOLO11 (AGPL-3.0). Needs `pip install -e '.[rfdetr]'` |
| `--force` | ignore the cache and re-detect |

You can pass several files or a folder if XbotGo split the game into multiple files.

## Calibrating the court

```bash
hoopstats calibrate "videos/18:00:21.MP4"
```

`analyze` offers this automatically the first time. Calibration happens in your **web browser**:
1. It samples the video, which takes about 1–2 minutes for a full 4K game, and groups the frames into the camera views XbotGo used.
2. It opens a page with one frame per view, plus a court diagram.
3. For each view, click a landmark on the diagram (or follow the highlighted one), then click where it is on the frame. A magnifier follows your mouse.
4. After 4 or more points, magenta court lines show the fit. When they sit on the painted lines, the view is good.
5. Click **Save calibration**. The terminal continues with the analysis.

Set the court size in `runs/<video>/game.yaml` first. Youth gyms vary: 84 ft is high school, 74 ft is common for youth, 94 ft is college/NBA. Landmarks around a basket (lane corners, free-throw line ends, 3-point line) are the easiest to place and don't depend on court length.

## Fixing players it couldn't identify

The report's *Data quality* section shows thumbnails of players whose jersey number couldn't be read. Add them to `runs/<video>/overrides.yaml`:

```yaml
track_57: A10      # A = left team on the overlay, B = right team
track_88: B4
track_131: ignore  # a coach, a sub at the scorer's table, ...
```

Then `hoopstats report runs/<video>`. This reuses the cache and takes seconds.

## How it works

| Step | Method |
|---|---|
| Video | ffmpeg (hardware decoding on Mac). Handles 4K, HEVC, odd file names like `18:00:21.MP4`, and split files |
| Score | XbotGo ScoreSync overlay read with digit templates. Score changes = made baskets with their point value. Scorer corrections and recording pauses are handled |
| Players / ball | YOLO11 (or RF-DETR) person + ball detection, own ByteTrack-style tracker with camera-motion compensation |
| Court | Calibrated keyframes + SIFT matching + optical flow → court position of every player in every frame, even while XbotGo pans and zooms |
| Rim | Camera pose recovered from the court mapping → the 10 ft rim is projected into the image (no rim detector needed) |
| Teams | Jersey colour clustering, referees filtered by striped shirts. Which colour is the left/right overlay team is learned from who was shooting before each score change |
| Numbers | Apple Vision OCR on jersey crops, voted over many frames and limited to the roster if you give one |
| Events | ball possession → shot attempts (release → ball near rim) → make/miss (scoreboard, else ball through the rim) → rebounds, assists, turnovers, steals |
| Stats / insights | zone and area FG%, catch vs dribble, movement, spacing, touch time + rule-based coaching insights |

## Outputs

`runs/<video>/` contains:
- `report.html`
- `player_stats.csv`
- `shots.csv`
- `events.csv`
- `coach_summary.json`: hand this to Claude for written development plans
- `game.json`
- `annotated.mp4`: optional, from `hoopstats annotate runs/<video> --start 10:00 --end 12:00`. Use it to check what the pipeline sees

## Accuracy

These are **estimates** to use for development trends, not an official box score.
- **Points per team** come from the scoreboard overlay and are reliable.
- **Shooter, location, assists and rebounds** depend on how well the ball and the jersey numbers are seen.
- **The report's Data quality section** shows how much of the game was registered on the court and identified.
- **The biggest improvement** is fine-tuning the detector on a few hundred labelled frames from your own games. See `docs/PLAN.md`, section 7.

## Tests

```bash
python -m pytest tests -q
hoopstats sample
```
