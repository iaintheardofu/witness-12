"""AKD1000 crew. Software forward is allowed. Hardware forward is opt-in.

No power-sensor calls. Batches stay at 8. hw_only is required for a board claim.
"""

from __future__ import annotations

import numpy as np

from witness12.catalog import N_EXAMPLES, SPECIALISTS, training_pairs
from witness12.codec import N_BITS, as_akida_batch

NEURONS_PER_CLASS = 16
NUM_WEIGHTS = 32


def _model(n_classes: int):
    import akida

    model = akida.Model()
    model.add(akida.InputData(input_shape=(1, 1, N_BITS), input_bits=1, name="spikes"))
    model.add(
        akida.FullyConnected(
            NEURONS_PER_CLASS * n_classes,
            name="sense",
            weights_bits=1,
            activation=False,
            act_bits=1,
        )
    )
    model.compile(akida.AkidaUnsupervised(num_weights=NUM_WEIGHTS, num_classes=n_classes))
    return model


def class_sums(out: np.ndarray, n_classes: int) -> np.ndarray:
    pot = np.asarray(out).reshape(out.shape[0], -1).astype(np.int64)
    width = pot.shape[1] // n_classes
    return np.stack([pot[:, i * width : (i + 1) * width].sum(axis=1) for i in range(n_classes)], axis=1)


def fit_pairs(model, pairs: list[tuple[np.ndarray, int]]) -> int:
    calls = 0
    for bits, label in pairs:
        if len(bits) != N_EXAMPLES:
            raise ValueError(f"fit batch must be {N_EXAMPLES}, got {len(bits)}")
        y = np.full(len(bits), label, dtype=np.int32)
        model.fit(as_akida_batch(bits), y, batch_size=N_EXAMPLES)
        calls += 1
    return calls


def build_crew(include_reserve_class: bool) -> list[dict]:
    crew = []
    for row in SPECIALISTS:
        n_classes = 3 if row["reserve"] else 2
        model = _model(n_classes)
        opened = bool(include_reserve_class and row["reserve"])
        fit_pairs(model, training_pairs(row["name"], opened))
        crew.append({"model": model, "row": row, "reserve_fit": opened, "n_classes": n_classes})
    return crew


def score_packet(crew: list[dict], packet: np.ndarray) -> list[dict]:
    batch = as_akida_batch(packet)
    readings = []
    for member in crew:
        row = member["row"]
        sums = class_sums(member["model"].forward(batch, batch_size=1), member["n_classes"])[0]
        target = 2 if row["reserve"] else 1
        if row["reserve"] and not member["reserve_fit"]:
            fired = False
        else:
            others = np.delete(sums, target)
            fired = int(np.argmax(sums)) == target and int(sums[target]) > int(others.max())
        readings.append(
            {
                "chip": row["chip"],
                "name": row["name"],
                "reserve": row["reserve"],
                "fired": bool(fired),
                "scores": [int(s) for s in sums],
                "source": "akida-software",
            }
        )
    return readings


def fit_reserve(crew: list[dict]) -> int:
    member = crew[11]
    pairs = training_pairs("uncommanded_heater", True)
    fit_pairs(member["model"], [pairs[-1]])
    member["reserve_fit"] = True
    return N_EXAMPLES
