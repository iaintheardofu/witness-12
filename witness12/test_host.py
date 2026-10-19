"""Host-side checks. These do not open /dev/akida and are not a board result."""

from __future__ import annotations

import numpy as np

from witness12.catalog import SPECIALISTS, HostCrew, observation, predicted_before, spike_pair
from witness12.codec import N_BITS, encode, residual
from witness12.engine import run_host
from witness12.language import ungrounded_numbers


def test_packet_is_binary_512():
    packet = spike_pair("eclipse_entry")
    assert packet.shape == (N_BITS,)
    assert set(np.unique(packet)).issubset({0, 1})


def test_quiet_residual_is_zero():
    z = residual(observation("quiet"), predicted_before("quiet"))
    assert np.all(np.abs(z) < 1e-9)


def test_timeline_speech_and_silence():
    run = run_host()
    assert run.spoken_steps == [
        "eclipse_entry",
        "eclipse_exit",
        "uncommanded_heater",
        "uncommanded_heater",
    ]
    assert run.silent_steps == ["quiet", "quiet", "eclipse_hold", "quiet", "uncommanded_heater"]
    assert ungrounded_numbers(run.claims) == []


def test_label_is_withheld_until_the_reserved_class_is_stored():
    run = run_host()
    before, after = run.transcript.split("[LEARN]", 1)
    assert "uncommanded_heater" not in before
    assert "assigned the label uncommanded_heater" in after
    assert "not a diagnosed cause" in after
    assert "does not send a command" in after


def test_each_onset_fires_only_its_specialist():
    crew = HostCrew()
    crew.learn_reserve()
    for row in SPECIALISTS:
        fired = [r["name"] for r in crew.score(spike_pair(row["name"])) if r["fired"]]
        assert fired == [row["name"]], (row["name"], fired)


def test_commanded_heater_does_not_claim_the_uncommanded_window():
    crew = HostCrew()
    packet = encode(observation("uncommanded_heater"), predicted_before("uncommanded_heater"))
    readings = {r["name"]: r for r in crew.score(packet)}
    assert readings["commanded_heater"]["fired"] is False
    assert readings["uncommanded_heater"]["fired"] is False
    crew.learn_reserve()
    readings = {r["name"]: r for r in crew.score(packet)}
    assert readings["uncommanded_heater"]["fired"] is True
    assert readings["commanded_heater"]["fired"] is False


def test_learning_does_not_rewrite_other_specialists():
    crew = HostCrew()
    before = crew.masks["eclipse_entry"].copy()
    packet = spike_pair("eclipse_entry")
    crew.learn_reserve()
    assert np.array_equal(before, crew.masks["eclipse_entry"])
    fired = [r["name"] for r in crew.score(packet) if r["fired"]]
    assert fired == ["eclipse_entry"]


if __name__ == "__main__":
    test_packet_is_binary_512()
    test_quiet_residual_is_zero()
    test_timeline_speech_and_silence()
    test_label_is_withheld_until_the_reserved_class_is_stored()
    test_each_onset_fires_only_its_specialist()
    test_commanded_heater_does_not_claim_the_uncommanded_window()
    test_learning_does_not_rewrite_other_specialists()
    print("host tests passed")
