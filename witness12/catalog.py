"""Twelve specialists and the synthetic CubeSat situations they watch.

The stimulus is scripted. Nothing here was downlinked from orbit.
"""

from __future__ import annotations

import numpy as np

from witness12.codec import BASE, CHANNELS, INDEX, N_BITS, encode

# chip, name, channels whose residual thermometer must be hot to fire
SPECIALISTS = (
    {"chip": 0, "name": "eclipse_entry", "channels": (6, 8), "reserve": False},
    {"chip": 1, "name": "eclipse_exit", "channels": (6, 8), "reserve": False},
    {"chip": 2, "name": "wheel_desat", "channels": (3, 11), "reserve": False},
    {"chip": 3, "name": "commanded_heater", "channels": (5, 1, 0), "reserve": False},
    {"chip": 4, "name": "radio_pass", "channels": (4, 13), "reserve": False},
    {"chip": 5, "name": "payload_on", "channels": (2, 12), "reserve": False},
    {"chip": 6, "name": "safe_hold", "channels": (2, 3), "reserve": False},
    {"chip": 7, "name": "gyro_tumble", "channels": (7,), "reserve": False},
    {"chip": 8, "name": "array_overcurrent", "channels": (8, 14), "reserve": False},
    {"chip": 9, "name": "coolant_swing", "channels": (9, 10), "reserve": False},
    {"chip": 10, "name": "vibration_burst", "channels": (10,), "reserve": False},
    {"chip": 11, "name": "uncommanded_heater", "channels": (1, 0, 5), "reserve": True},
)

BY_NAME = {row["name"]: row for row in SPECIALISTS}
N_EXAMPLES = 8


def _vec(**over: float) -> np.ndarray:
    out = BASE.copy()
    for name, value in over.items():
        out[INDEX[name]] = value
    return out


def observation(kind: str) -> np.ndarray:
    if kind == "quiet":
        return _vec()
    if kind in ("eclipse_entry", "eclipse_hold"):
        return _vec(eclipse=1.0, solar_current_a=0.05, bus_voltage_v=7.40)
    if kind == "eclipse_exit":
        # One-step transient as the array comes out of eclipse. Not the fault.
        return _vec(eclipse=0.0, solar_current_a=2.05, bus_voltage_v=8.05, array_temp_c=42.0)
    if kind == "wheel_desat":
        return _vec(wheel_rpm=2800.0, wheel_current_a=0.70, gyro_rate_dps=0.30)
    if kind == "commanded_heater":
        return _vec(
            heater_command=1.0,
            battery_temp_c=34.0,
            bus_voltage_v=7.30,
            bus_current_a=0.90,
        )
    if kind == "radio_pass":
        return _vec(radio_rssi=0.90, radio_command=1.0, payload_current_a=0.45)
    if kind == "payload_on":
        return _vec(payload_current_a=1.30, payload_temp_c=27.0, bus_voltage_v=7.45)
    if kind == "safe_hold":
        return _vec(
            payload_current_a=0.02,
            wheel_rpm=0.0,
            wheel_current_a=0.01,
            radio_rssi=0.02,
        )
    if kind == "gyro_tumble":
        return _vec(gyro_rate_dps=3.20, wheel_rpm=900.0)
    if kind == "array_overcurrent":
        return _vec(solar_current_a=2.60, array_temp_c=62.0, bus_voltage_v=8.40)
    if kind == "coolant_swing":
        return _vec(tank_pressure_bar=3.10, vibration_g=0.55, payload_temp_c=8.0)
    if kind == "vibration_burst":
        return _vec(vibration_g=1.40)
    if kind == "uncommanded_heater":
        # Same thermal and power shape as a heater load, command line stays low.
        return _vec(
            heater_command=0.0,
            battery_temp_c=36.0,
            bus_voltage_v=7.10,
            bus_current_a=0.95,
        )
    raise KeyError(kind)


def predicted_before(kind: str) -> np.ndarray:
    """State the coasting model would expect just before this onset."""
    if kind == "eclipse_exit":
        return observation("eclipse_hold")
    return observation("quiet")


def spike_pair(kind: str) -> np.ndarray:
    return encode(observation(kind), predicted_before(kind))


# Each specialist must beat quiet and this near miss. The mask is the bits
# that are on for the specialist and off for both of those.
RIVALS = {
    "eclipse_entry": "eclipse_exit",
    "eclipse_exit": "eclipse_entry",
    "wheel_desat": "gyro_tumble",
    "commanded_heater": "uncommanded_heater",
    "radio_pass": "payload_on",
    "payload_on": "radio_pass",
    "safe_hold": "wheel_desat",
    "gyro_tumble": "wheel_desat",
    "array_overcurrent": "eclipse_exit",
    "coolant_swing": "vibration_burst",
    "vibration_burst": "coolant_swing",
    "uncommanded_heater": "commanded_heater",
}


def _stack(kind: str) -> np.ndarray:
    row = spike_pair(kind)
    return np.repeat(row.reshape(1, -1), N_EXAMPLES, axis=0)


def training_pairs(name: str, include_reserve_class: bool) -> list[tuple[np.ndarray, int]]:
    """Labeled spike batches of exactly 8, the batch this host's cards accept.

    The reserve chip is compiled for three classes. Class 2 is withheld until
    the learn step when include_reserve_class is false.
    """
    row = BY_NAME[name]
    if row["reserve"]:
        pairs = [
            (_stack("quiet"), 0),
            (_stack("commanded_heater"), 1),
        ]
        if include_reserve_class:
            pairs.append((_stack("uncommanded_heater"), 2))
        return pairs
    rival = "uncommanded_heater" if name == "commanded_heater" else "commanded_heater"
    if name == "eclipse_entry":
        rival = "eclipse_exit"
    elif name == "eclipse_exit":
        rival = "eclipse_entry"
    neg = np.concatenate([_stack("quiet")[:4], _stack(rival)[:4]], axis=0)
    return [(_stack(name), 1), (neg, 0)]


def prototype(bits: np.ndarray) -> np.ndarray:
    mean = np.asarray(bits, dtype=np.float64).mean(axis=0)
    return (mean >= 0.5).astype(np.uint8)


def _channel_keep(channels: tuple[int, ...]) -> np.ndarray:
    keep = np.zeros(N_BITS, dtype=np.uint8)
    for c in channels:
        keep[c * 32 : c * 32 + 20] = 1
    return keep


def private_mask(name: str) -> np.ndarray:
    """Spikes that belong to this situation and not to quiet or its rival."""
    row = BY_NAME[name]
    pos = spike_pair(name)
    mask = pos & (1 - spike_pair("quiet")) & (1 - spike_pair(RIVALS[name]))
    return mask & _channel_keep(row["channels"])


class HostCrew:
    """Exact spike-mask decoder. This is the ledger's readout, not a chip result."""

    def __init__(self) -> None:
        self.masks = {row["name"]: private_mask(row["name"]) for row in SPECIALISTS}
        self.reserve_open = False

    def learn_reserve(self) -> int:
        self.reserve_open = True
        return N_EXAMPLES

    def score(self, packet: np.ndarray) -> list[dict]:
        from witness12.codec import residual_pop

        packet = np.asarray(packet, dtype=np.uint8)
        covered = []
        for row in SPECIALISTS:
            mask = self.masks[row["name"]]
            need = int(mask.sum())
            got = int((packet & mask).sum())
            open_class = (not row["reserve"]) or self.reserve_open
            covered.append((need, got, open_class, row))
        winner = None
        best_key = None
        tied = False
        for need, got, open_class, row in covered:
            if not open_class or need == 0 or got != need:
                continue
            hot = residual_pop(packet, row["channels"])
            key = (hot, need)
            if best_key is None or key > best_key:
                winner = row["name"]
                best_key = key
                tied = False
            elif key == best_key:
                tied = True
        if tied:
            winner = None
        readings = []
        for need, got, open_class, row in covered:
            readings.append(
                {
                    "chip": row["chip"],
                    "name": row["name"],
                    "reserve": row["reserve"],
                    "fired": row["name"] == winner,
                    "scores": [got, need],
                    "residual_spikes": residual_pop(packet, row["channels"]),
                    "source": "host",
                }
            )
        return readings


def packet_shape_ok(packet: np.ndarray) -> bool:
    packet = np.asarray(packet)
    return packet.shape == (N_BITS,) and set(np.unique(packet)).issubset({0, 1})
