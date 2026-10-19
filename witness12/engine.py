"""One timeline. Speech is produced only on steps that break the prediction."""

from __future__ import annotations

from dataclasses import dataclass, field

from witness12.catalog import HostCrew, observation
from witness12.codec import encode
from witness12.language import Claim, claims_for, render, ungrounded_numbers
from witness12.ledger import Ledger

# kind, or a learn marker. Learn is not an observation.
TIMELINE: tuple[tuple[str, str], ...] = (
    ("obs", "quiet"),
    ("obs", "quiet"),
    ("obs", "eclipse_entry"),
    ("obs", "eclipse_hold"),
    ("obs", "eclipse_exit"),
    ("obs", "quiet"),
    ("obs", "uncommanded_heater"),
    ("learn", "uncommanded_heater"),
    ("obs", "uncommanded_heater"),
    ("obs", "uncommanded_heater"),
)


@dataclass
class Run:
    lines: list[str] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)
    spoken_steps: list[str] = field(default_factory=list)
    silent_steps: list[str] = field(default_factory=list)
    learned_windows: int = 0
    n_obs: int = 0

    @property
    def transcript(self) -> str:
        return "\n".join(self.lines)


def run_host() -> Run:
    crew = HostCrew()
    ledger = Ledger()
    run = Run()
    learned: int | None = None
    for kind, name in TIMELINE:
        if kind == "learn":
            run.learned_windows = crew.learn_reserve()
            learned = run.learned_windows
            run.lines.append(
                f"[LEARN] host prototype for chip 11 stored {run.learned_windows} windows "
                "of the new label. No sentence was issued for the fit itself."
            )
            continue
        pred = ledger.predict()
        packet = encode(observation(name), pred)
        readings = crew.score(packet)
        step = ledger.step(name, readings)
        step_claims = claims_for(step, learned if name == "uncommanded_heater" else None)
        run.n_obs += 1
        if step_claims:
            # The LEARNED claim belongs only to the first spoken step after fit.
            if learned is not None and not any(c.kind == "LEARNED" for c in run.claims):
                pass
            elif any(c.kind == "LEARNED" for c in step_claims) and any(
                c.kind == "LEARNED" for c in run.claims
            ):
                step_claims = [c for c in step_claims if c.kind != "LEARNED"]
            text = render(step_claims)
            run.lines.append(text)
            run.claims.extend(step_claims)
            run.spoken_steps.append(name)
        else:
            run.lines.append(f"[SILENT] t={step['t_s']:.0f}s {name} residual inside the ledger")
            run.silent_steps.append(name)
        _ = packet
    return run


def assert_grounded(run: Run) -> None:
    bad = ungrounded_numbers(run.claims)
    if bad:
        raise AssertionError(f"ungrounded numbers: {bad}")
