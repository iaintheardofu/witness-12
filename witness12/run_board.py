"""Map one specialist per AKD1000 and few-shot the reserved class on chip 11.

Fit happens after hw_only map, which is the on-chip learning path. This
script never calls the power meter. A chip that does not separate is
recorded as a miss. The run does not invent a pass.
"""

from __future__ import annotations

import json
import signal
import time
from pathlib import Path

import numpy as np

from witness12.catalog import RIVALS, SPECIALISTS
from witness12.sparse import N_SPARSE, sparse_batch, sparse_packet

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "board.json"
NEURONS = 16
NUM_WEIGHTS = 20
BATCH = 8
CHIP_BUDGET_S = 60


class ChipTimeout(Exception):
    pass


def _alarm(_signum, _frame):
    raise ChipTimeout()


def _model(n_classes: int):
    import akida

    model = akida.Model()
    model.add(akida.InputData(input_shape=(1, 1, N_SPARSE), input_bits=1, name="spikes"))
    model.add(
        akida.FullyConnected(
            NEURONS * n_classes,
            name="sense",
            weights_bits=1,
            activation=False,
            act_bits=1,
        )
    )
    model.compile(
        akida.AkidaUnsupervised(
            num_weights=NUM_WEIGHTS,
            num_classes=n_classes,
            initial_plasticity=1.0,
            min_plasticity=1.0,
            plasticity_decay=0.0,
            learning_competition=0.0,
        )
    )
    return model


def _sums(model, kind: str, n_classes: int) -> list[int]:
    out = model.forward(sparse_batch(kind, BATCH), batch_size=BATCH)
    pot = np.asarray(out).reshape(out.shape[0], -1).astype(np.int64)
    width = pot.shape[1] // n_classes
    # Eight identical windows. The mean is the window.
    return [int(pot[:, i * width : (i + 1) * width].sum(axis=1).mean()) for i in range(n_classes)]


def _fit(model, kind: str, label: int) -> None:
    y = np.full(BATCH, label, dtype=np.int32)
    model.fit(sparse_batch(kind, BATCH), y, batch_size=BATCH)


def _separated(own: list[int], rival: list[int], quiet: list[int]) -> bool:
    return own[1] > own[0] and rival[1] < rival[0] and quiet[1] <= quiet[0]


def _empty_devices() -> list[str]:
    nodes = sorted(Path("/dev").glob("akida*"))
    return [str(p) for p in nodes]


def run() -> dict:
    report: dict = {
        "stimulus": "synthetic cubesat bus, not a downlink",
        "power_meter": "not called",
        "batch": BATCH,
        "num_weights": NUM_WEIGHTS,
        "devices_nodes": _empty_devices(),
        "chips": [],
        "reserve": None,
        "control": None,
        "errors": [],
    }
    if len(report["devices_nodes"]) < 12:
        report["errors"].append(f"found {len(report['devices_nodes'])} /dev/akida nodes")
        return report
    import akida

    t0 = time.perf_counter()
    devs = list(akida.devices())
    report["enumerate_s"] = round(time.perf_counter() - t0, 3)
    report["device_desc"] = [str(d.desc) for d in devs]
    if len(devs) < 12:
        report["errors"].append(f"akida.devices() returned {len(devs)}")
        return report

    signal.signal(signal.SIGALRM, _alarm)
    mapped = []
    for row, dev in zip(SPECIALISTS, devs[:12]):
        chip = {
            "chip": row["chip"],
            "name": row["name"],
            "device": str(dev.desc),
            "map": None,
            "scores": {},
            "separated": False,
            "error": None,
        }
        n_classes = 3 if row["reserve"] else 2
        model = None
        try:
            signal.alarm(CHIP_BUDGET_S)
            model = _model(n_classes)
            model.map(dev, hw_only=True)
            chip["map"] = "hw_only"
            if row["reserve"]:
                _fit(model, "commanded_heater", 0)
                _fit(model, "eclipse_entry", 1)
                chip["scores"] = {
                    "novel_before": _sums(model, "uncommanded_heater", 3),
                    "commanded_before": _sums(model, "commanded_heater", 3),
                    "quiet_before": _sums(model, "quiet", 3),
                }
            else:
                _fit(model, RIVALS[row["name"]], 0)
                _fit(model, row["name"], 1)
                own = _sums(model, row["name"], 2)
                rival = _sums(model, RIVALS[row["name"]], 2)
                quiet = _sums(model, "quiet", 2)
                chip["scores"] = {"own": own, "rival": rival, "quiet": quiet}
                chip["separated"] = _separated(own, rival, quiet)
            signal.alarm(0)
        except ChipTimeout:
            signal.alarm(0)
            chip["error"] = f"exceeded {CHIP_BUDGET_S}s"
        except Exception as exc:  # noqa: BLE001
            signal.alarm(0)
            chip["error"] = str(exc)
            model = None
        if chip["error"]:
            model = None
        report["chips"].append(chip)
        mapped.append(model)
        print(
            f"chip {row['chip']:2d} {row['name']:22} {chip['map'] or 'unmapped'} "
            f"sep={chip['separated']} {chip['error'] or ''}",
            flush=True,
        )

    # Control: chips 0-10 on one packet, then the reserved class is fit, then again.
    reserve = mapped[11]
    if reserve is not None and report["chips"][11]["error"] is None:
        before = []
        after = []
        try:
            signal.alarm(CHIP_BUDGET_S)
            for member in mapped[:11]:
                if member is None:
                    before.append(None)
                else:
                    before.append(_sums(member, "eclipse_entry", 2))
            _fit(reserve, "uncommanded_heater", 2)
            novel = _sums(reserve, "uncommanded_heater", 3)
            commanded = _sums(reserve, "commanded_heater", 3)
            quiet = _sums(reserve, "quiet", 3)
            for member in mapped[:11]:
                if member is None:
                    after.append(None)
                else:
                    after.append(_sums(member, "eclipse_entry", 2))
            signal.alarm(0)
            report["chips"][11]["scores"]["novel_after"] = novel
            report["chips"][11]["scores"]["commanded_after"] = commanded
            report["chips"][11]["scores"]["quiet_after"] = quiet
            learned = int(np.argmax(novel)) == 2 and novel[2] > max(novel[0], novel[1])
            commanded_held = int(np.argmax(commanded)) == 0 and commanded[0] > max(commanded[1], commanded[2])
            quiet_held = max(quiet) == 0 or int(np.argmax(quiet)) != 2 or quiet[2] <= quiet[0]
            report["reserve"] = {
                "learned_windows": BATCH,
                "novel_after_argmax_is_new_class": learned,
                "commanded_stays_original_class": commanded_held,
                "quiet_does_not_become_the_new_class": bool(quiet_held),
                "novel_after": novel,
                "commanded_after": commanded,
                "quiet_after": quiet,
            }
            same = []
            for b, a in zip(before, after):
                same.append(b is not None and a is not None and b == a)
            report["control"] = {
                "packet": "eclipse_entry",
                "unchanged": same,
                "all_unchanged": all(same),
                "before": before,
                "after": after,
            }
        except ChipTimeout:
            signal.alarm(0)
            report["errors"].append("reserve fit exceeded budget")
        except Exception as exc:  # noqa: BLE001
            signal.alarm(0)
            report["errors"].append(str(exc))
    report["spike_counts"] = {row["name"]: int(sparse_packet(row["name"]).sum()) for row in SPECIALISTS}
    report["spike_counts"]["quiet"] = int(sparse_packet("quiet").sum())
    return report


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    report = run()
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {OUT}")
    reserve = report.get("reserve")
    if reserve:
        print(
            "reserve learned="
            f"{reserve['novel_after_argmax_is_new_class']} "
            f"commanded_held={reserve['commanded_stays_original_class']}"
        )


if __name__ == "__main__":
    main()
