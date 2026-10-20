"""End-to-end test of everything after detection, on a synthetic game.

    python -m pytest tests/ -q      (or: python tests/test_pipeline.py)
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hoopstats import analyze, config  # noqa: E402
from hoopstats.court import Court  # noqa: E402
from hoopstats.scoreboard import clean_timeline, increments  # noqa: E402
from synthetic import generate  # noqa: E402


def test_court_zones():
    c = Court()
    assert c.classify(80, 25)["zone"] == "rim"
    assert c.classify(84 - 5.25 - 21, 25)["zone"] == "three"
    assert c.classify(84 - 5.25 - 21, 25)["value"] == 3
    assert c.classify(84 - 5.25 - 12, 25)["zone"] == "paint"
    assert c.classify(66, 15)["zone"] == "mid"
    # offense-left when attacking the right basket is the far sideline (high y)
    assert c.classify(80, 45)["area"] == "left_corner"
    assert c.classify(4, 45)["area"] == "right_corner"


def test_scoreboard_cleaning():
    raw = ([(t, 0, 0, 1) for t in range(5)] + [(t, 2, 0, 1) for t in range(5, 9)] + [(t, 2, 3, 1) for t in range(9, 12)]
           + [(t, 2, 0, 1) for t in range(12, 20)] + [(t, 12, 8, 1) for t in range(20, 30)])
    tl = clean_timeline(raw)
    b, g = increments(tl)
    assert [(x["side"], x["points"]) for x in b] == [("left", 2)]   # the +3 was taken back
    assert g == [{"t": 20, "left": 10, "right": 8}]


def run_synthetic(pan=True):
    d = Path(tempfile.mkdtemp())
    truth = generate(d, seconds=300, pan=pan)
    cfg = config.load(None)
    cfg["teams"][0]["name"], cfg["teams"][1]["name"] = "Blue", "Navy"
    game = analyze.run(d, cfg, log=lambda *a: None)
    return d, truth, game


def test_end_to_end():
    d, truth, game = run_synthetic()
    st = game["stats"]["teams"]
    # points from the scoreboard-matched makes
    assert st["A"]["pts"] == truth["score"]["A"], (st["A"]["pts"], truth["score"])
    assert st["B"]["pts"] == truth["score"]["B"]
    n_true = len(truth["shots"])
    n_found = len([s for s in game["shots"] if not s["free_throw"]])
    assert abs(n_found - n_true) <= max(2, 0.15 * n_true), (n_found, n_true)
    # shooters identified by number
    named = [s for s in game["shots"] if not s["player"].endswith("?")]
    assert len(named) >= 0.7 * n_found
    # makes attributed to the right player
    tp = {(round(s["t"]), f"{s['team']}{s['player'][1]}") for s in truth["shots"] if s["made"]}
    assert (d / "report.html").exists() and (d / "shots.csv").exists()
    reb = [e for e in game["events"] if e["type"] == "rebound"]
    assert len(reb) >= 0.6 * sum(1 for s in truth["shots"] if not s["made"])


if __name__ == "__main__":
    test_court_zones()
    test_scoreboard_cleaning()
    d, truth, game = run_synthetic()
    st = game["stats"]["teams"]
    print("truth score", truth["score"], "detected", st["A"]["pts"], st["B"]["pts"])
    print("true shots", len(truth["shots"]), "found", len(game["shots"]))
    for s in truth["shots"][:8]:
        print("  T", round(s["t"], 1), s["team"], s["player"], "made" if s["made"] else "miss", s["value"], [round(v) for v in s["xy"]])
    for s in game["shots"][:8]:
        print("  F", s["t"], s["team"], s["player"], "made" if s["made"] else "miss", s["value"], s["x"], s["y"], s["source"])
    print("report:", d / "report.html")
