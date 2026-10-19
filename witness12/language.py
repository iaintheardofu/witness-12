"""Claim-typed speech. Every number in a sentence is copied from a claim field.

There is no language model in this path. A sentence that needs a number the
ledger does not hold cannot be built.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from witness12.codec import CHANNELS
from witness12.ledger import hottest

_NUM = re.compile(r"-?\d+(?:\.\d+)?")


@dataclass(frozen=True)
class Claim:
    kind: str
    text: str
    numbers: tuple[str, ...]


def _fmt(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def _join(parts: list[str]) -> str:
    return ", ".join(parts)


def claims_for(step: dict, learned_windows: int | None) -> list[Claim]:
    z: np.ndarray = step["z"]
    obs: np.ndarray = step["obs"]
    pred: np.ndarray = step["pred"]
    fired = [r for r in step["readings"] if r["fired"]]
    hot = hottest(z)
    if not fired and not hot:
        return []

    claims: list[Claim] = []
    t = _fmt(step["t_s"], 0)
    obs_parts = []
    obs_nums = [t]
    for name, _ in hot:
        i = CHANNELS.index(name)
        shown = _fmt(float(obs[i]))
        obs_parts.append(f"{name}={shown}")
        obs_nums.append(shown)
    # Command and eclipse are context even when their residual is small.
    for name in ("heater_command", "eclipse"):
        i = CHANNELS.index(name)
        shown = _fmt(float(obs[i]))
        piece = f"{name}={shown}"
        if piece not in obs_parts:
            obs_parts.append(piece)
            obs_nums.append(shown)
    claims.append(Claim("OBSERVED", f"[OBSERVED] t={t}s " + _join(obs_parts), tuple(obs_nums)))

    pred_parts = []
    pred_nums = []
    for name, _ in hot:
        i = CHANNELS.index(name)
        shown = _fmt(float(pred[i]))
        pred_parts.append(f"{name}={shown}")
        pred_nums.append(shown)
    claims.append(
        Claim(
            "PREDICTED",
            "[PREDICTED] coasting ledger expected " + _join(pred_parts),
            tuple(pred_nums),
        )
    )

    res_parts = []
    res_nums = []
    for name, value in hot:
        shown = _fmt(value)
        res_parts.append(f"{name} z={shown}")
        res_nums.append(shown)
    claims.append(Claim("RESIDUAL", "[RESIDUAL] " + _join(res_parts), tuple(res_nums)))

    if fired:
        for reading in fired:
            got = _fmt(float(reading["scores"][0]), 0)
            need = _fmt(float(reading["scores"][1]), 0)
            chip = str(int(reading["chip"]))
            claims.append(
                Claim(
                    "CHIP",
                    f"[CHIP] chip {chip} {reading['name']} matched {got} of {need} "
                    f"private spikes ({reading['source']})",
                    (chip, got, need),
                )
            )
    else:
        claims.append(
            Claim(
                "CHIP",
                "[CHIP] no specialist fired. The residual is not one of the learned classes.",
                (),
            )
        )

    if learned_windows is not None and any(r["name"] == "uncommanded_heater" and r["fired"] for r in fired):
        n = str(learned_windows)
        claims.append(
            Claim(
                "LEARNED",
                f"[LEARNED] chip 11 assigned the label uncommanded_heater to {n} spike windows. "
                "The label names those windows. It is not a diagnosed cause.",
                (n, "11"),
            )
        )
        claims.append(
            Claim(
                "HYPOTHESIS",
                "[HYPOTHESIS] Those windows share a battery temperature rise and a bus voltage sag "
                "while heater_command stayed low. Confirm if the command bit stays low and the "
                "temperature residual stays high. Drop the label if a later command bit is high.",
                (),
            )
        )
    elif not fired:
        claims.append(
            Claim(
                "HYPOTHESIS",
                "[HYPOTHESIS] No stored class matches this residual. Withhold a fault name until "
                "a reserved chip has learned the windows.",
                (),
            )
        )
    else:
        name = fired[0]["name"].replace("_", " ")
        claims.append(
            Claim(
                "HYPOTHESIS",
                f"[HYPOTHESIS] The firing specialist is {name}. This restates the class label. "
                "It does not identify a root cause.",
                (),
            )
        )

    claims.append(
        Claim(
            "ACTION",
            "[ACTION] Recommend the operator review the heater and power lines. "
            "This report does not send a command.",
            (),
        )
    )
    return claims


def render(claims: list[Claim]) -> str:
    return "\n".join(c.text for c in claims)


def ungrounded_numbers(claims: list[Claim]) -> list[str]:
    """Numbers printed in the text that were not copied from the claim fields."""
    bad = []
    for claim in claims:
        allowed = set(claim.numbers)
        for found in _NUM.findall(claim.text):
            if found not in allowed:
                bad.append(found)
    return bad
