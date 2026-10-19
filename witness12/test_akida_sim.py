"""Optional dense-packet probe. Not the board path and not part of test_host.

`run_board.py` sends the sparse packet in `sparse.py`. This module builds the
dense 512-bit packet from `codec.py` and asks MetaTF whether each specialist
separates. On the development host that probe did not separate, so the default
entry point does not treat it as a pass/fail test of the repository.
Set WITNESS_DENSE_PROBE=1 to run the strict checks. They do not open /dev/akida.
"""

from __future__ import annotations

import numpy as np

from witness12.akida_backend import build_crew, fit_reserve, score_packet
from witness12.catalog import SPECIALISTS, spike_pair


def _fired(models, kind):
    return [r["name"] for r in score_packet(models, spike_pair(kind)) if r["fired"]]


def test_each_specialist_separates_in_software():
    models = build_crew(include_reserve_class=False)
    # The reserve class is withheld, so its onset must not be claimed.
    assert _fired(models, "uncommanded_heater") == []
    for row in SPECIALISTS:
        if row["reserve"]:
            continue
        fired = _fired(models, row["name"])
        assert fired == [row["name"]], (row["name"], fired, score_packet(models, spike_pair(row["name"]))[row["chip"]])
    fit_reserve(models)
    assert _fired(models, "uncommanded_heater") == ["uncommanded_heater"]
    assert _fired(models, "commanded_heater") == ["commanded_heater"]
    quiet = score_packet(models, spike_pair("quiet"))
    assert [r["name"] for r in quiet if r["fired"]] == []


if __name__ == "__main__":
    import os

    if os.environ.get("WITNESS_DENSE_PROBE") == "1":
        test_each_specialist_separates_in_software()
        print("akida software separation passed")
    else:
        print("dense-packet probe skipped; board path is witness12.sparse / run_board.py")
