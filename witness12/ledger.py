"""Persistent spacecraft state. Prediction lives here, not on the chips.

The chips classify spike packets. This ledger is the world model: it
predicts the next bus state, stores the residual, and assimilates a mode
only after a specialist fires.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from witness12.catalog import observation
from witness12.codec import CHANNELS, SIGMA, residual


@dataclass
class Ledger:
    mode: str = "quiet"
    t_s: float = 0.0
    history: list[dict] = field(default_factory=list)

    def predict(self) -> np.ndarray:
        if self.mode in ("eclipse_entry", "eclipse_hold"):
            return observation("eclipse_hold")
        if self.mode == "commanded_heater":
            return observation("commanded_heater")
        if self.mode == "uncommanded_heater":
            return observation("uncommanded_heater")
        return observation("quiet")

    def step(self, kind: str, readings: list[dict], dt_s: float = 30.0) -> dict:
        pred = self.predict()
        obs = observation(kind)
        z = residual(obs, pred)
        fired = [r for r in readings if r["fired"]]
        self.t_s += dt_s
        row = {
            "t_s": self.t_s,
            "kind": kind,
            "mode_before": self.mode,
            "obs": obs.copy(),
            "pred": pred.copy(),
            "z": z.copy(),
            "fired": [r["name"] for r in fired],
            "readings": readings,
        }
        if len(fired) == 1:
            self.mode = fired[0]["name"]
        elif kind == "eclipse_hold" and self.mode == "eclipse_entry":
            self.mode = "eclipse_hold"
        row["mode_after"] = self.mode
        self.history.append(row)
        return row


def hottest(z: np.ndarray, k: int = 4) -> list[tuple[str, float]]:
    order = np.argsort(-np.abs(z))
    out = []
    for i in order[:k]:
        if abs(z[i]) < 1.0:
            break
        out.append((CHANNELS[int(i)], float(z[i])))
    return out


def channel_value(vec: np.ndarray, name: str) -> float:
    return float(vec[CHANNELS.index(name)])
