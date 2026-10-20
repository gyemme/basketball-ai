"""Build the single-file interactive report.html from a computed game dict."""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

TEMPLATE = Path(__file__).with_name("template.html")


def _clean(o):
    """Make the game dict JSON-safe (numpy scalars, dates, tuples, NaN)."""
    if isinstance(o, (_dt.date, _dt.datetime)):
        return o.isoformat()
    try:
        import numpy as np
        if isinstance(o, np.generic):
            o = o.item()
    except ImportError:
        pass
    if isinstance(o, float) and o != o:
        return None
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    return o


def build(game: dict, out_path: str | Path) -> Path:
    game = dict(game)
    game.setdefault("meta", {})
    game["meta"].setdefault("generated", _dt.datetime.now().strftime("%Y-%m-%d %H:%M"))
    # positions are only needed for stats; drop them to keep the HTML small
    slim = {k: v for k, v in game.items() if k not in ("positions", "events", "possessions")}
    data = json.dumps(_clean(slim), separators=(",", ":"), allow_nan=False)
    data = data.replace("</", "<\\/")  # never close the <script> early
    html = TEMPLATE.read_text(encoding="utf-8").replace("__GAME_JSON__", data)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out
