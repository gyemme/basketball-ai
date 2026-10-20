"""Court geometry: dimensions, hoop positions, shot zones and areas.

Court coordinate system (feet), used everywhere in the pipeline:
    x: 0 .. length  along the sideline, from the LEFT baseline (as seen in the report)
    y: 0 .. width   across the court, from the BOTTOM sideline (near the report viewer)
Hoops sit at (hoop_offset, width/2) and (length - hoop_offset, width/2).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict


@dataclass
class Court:
    length: float = 84.0          # high-school / many youth gyms; NBA/college = 94
    width: float = 50.0
    hoop_offset: float = 5.25     # baseline -> center of hoop
    three_pt: float | None = 19.75  # arc radius from hoop center; None = no 3-pt line
    corner_three: float | None = None  # straight corner-line distance from hoop; None = same as arc
    lane_width: float = 12.0
    ft_line: float = 19.0         # baseline -> free-throw line
    rim_radius_ft: float = 6.0    # "at the rim" zone radius

    @classmethod
    def from_config(cls, cfg: dict | None) -> "Court":
        cfg = cfg or {}
        known = {k: v for k, v in cfg.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def to_dict(self) -> dict:
        return asdict(self)

    # ------------------------------------------------------------------ hoops
    @property
    def hoops(self) -> list[tuple[float, float]]:
        return [(self.hoop_offset, self.width / 2), (self.length - self.hoop_offset, self.width / 2)]

    def nearest_hoop(self, x: float, y: float) -> int:
        """0 = left hoop, 1 = right hoop."""
        return 0 if x < self.length / 2 else 1

    def contains(self, x: float, y: float, margin: float = 0.0) -> bool:
        return -margin <= x <= self.length + margin and -margin <= y <= self.width + margin

    # ------------------------------------------------------------- geometry
    def _local(self, x: float, y: float, hoop: int) -> tuple[float, float]:
        """(depth, lateral) relative to a hoop, from the offense's view facing the basket.

        depth   = distance out from the hoop toward mid-court (ft, negative = behind the hoop)
        lateral = offense-LEFT is positive
        """
        hx, hy = self.hoops[hoop]
        if hoop == 1:      # attacking the right basket: facing +x, left is +y
            return hx - x, y - hy
        return x - hx, hy - y  # attacking the left basket: facing -x, left is -y

    def is_three(self, x: float, y: float, hoop: int) -> bool:
        if self.three_pt is None:
            return False
        depth, lat = self._local(x, y, hoop)
        corner = self.corner_three if self.corner_three is not None else self.three_pt
        dist = math.hypot(depth, lat)
        # Corner straight segments run from the baseline until they meet the arc.
        if corner < self.three_pt:
            break_depth = math.sqrt(max(self.three_pt ** 2 - corner ** 2, 0.0))
            if depth <= break_depth:
                return abs(lat) >= corner
        return dist >= self.three_pt

    def in_paint(self, x: float, y: float, hoop: int) -> bool:
        depth, lat = self._local(x, y, hoop)
        baseline_depth = -self.hoop_offset
        return baseline_depth <= depth <= self.ft_line - self.hoop_offset and abs(lat) <= self.lane_width / 2

    def classify(self, x: float, y: float, hoop: int | None = None, free_throw: bool = False) -> dict:
        """Describe a shot taken from court point (x, y) at `hoop` (default: nearest)."""
        if hoop is None:
            hoop = self.nearest_hoop(x, y)
        depth, lat = self._local(x, y, hoop)
        dist = math.hypot(depth, lat)
        if free_throw:
            return {"hoop": hoop, "dist_ft": round(dist, 1), "zone": "ft", "area": "middle", "value": 1}

        if dist <= self.rim_radius_ft:
            zone = "rim"
        elif self.is_three(x, y, hoop):
            zone = "three"
        elif self.in_paint(x, y, hoop):
            zone = "paint"
        else:
            zone = "mid"

        angle = math.degrees(math.atan2(lat, max(depth, 0.01)))  # 0 = straight on, +90 = offense left
        if abs(angle) >= 68 and depth <= 9:
            area = "left_corner" if angle > 0 else "right_corner"
        elif abs(angle) >= 25:
            area = "left_wing" if angle > 0 else "right_wing"
        else:
            area = "middle"
        return {
            "hoop": hoop,
            "dist_ft": round(dist, 1),
            "zone": zone,
            "area": area,
            "value": 3 if zone == "three" else 2,
        }


ZONES = ["rim", "paint", "mid", "three"]
ZONE_LABELS = {"rim": "At rim (<6 ft)", "paint": "Paint", "mid": "Mid-range", "three": "Three", "ft": "Free throw"}
AREAS = ["left_corner", "left_wing", "middle", "right_wing", "right_corner"]
AREA_LABELS = {
    "left_corner": "Left corner", "left_wing": "Left wing", "middle": "Middle",
    "right_wing": "Right wing", "right_corner": "Right corner",
}
