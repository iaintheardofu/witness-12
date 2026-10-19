"""Spike packet for one ledger step.

512 bits. Sixteen channels, thirty-two bits each. The packet is the only
input an AKD1000 specialist is allowed to see.

Per channel:
  0        sample valid
  1        residual sign (1 if observed > predicted)
  2:10     eight-bit thermometer of |z|
  10:18    eight-bit thermometer of the raw value inside its span
  18       value is at or above the span midpoint
  19       value is below the span midpoint
  20:32    unused, left at 0

A zero command is a spike, not silence. Bit 19 carries it. Prototype
learning on Akida locks onto spikes that are present.
"""

from __future__ import annotations

import numpy as np

N_CH = 16
BITS_PER = 32
N_BITS = N_CH * BITS_PER

CHANNELS = (
    "bus_voltage_v",
    "battery_temp_c",
    "payload_current_a",
    "wheel_rpm",
    "radio_rssi",
    "heater_command",
    "eclipse",
    "gyro_rate_dps",
    "solar_current_a",
    "tank_pressure_bar",
    "vibration_g",
    "wheel_current_a",
    "payload_temp_c",
    "radio_command",
    "array_temp_c",
    "bus_current_a",
)

# Sunlit, idle, heater off. These are script values for the synthetic bus,
# not a recording from a spacecraft.
BASE = np.array(
    [
        7.80, 22.0, 0.20, 200.0,
        0.10, 0.0, 0.0, 0.02,
        1.40, 2.00, 0.05, 0.10,
        18.0, 0.0, 30.0, 0.40,
    ],
    dtype=np.float64,
)

SIGMA = np.array(
    [
        0.08, 0.50, 0.05, 30.0,
        0.05, 0.05, 0.05, 0.05,
        0.10, 0.08, 0.03, 0.04,
        0.60, 0.05, 1.50, 0.06,
    ],
    dtype=np.float64,
)

# Spans only scale the value thermometer. They are not sensor calibrations.
LO = np.array(
    [
        6.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
    ],
    dtype=np.float64,
)
HI = np.array(
    [
        9.0, 50.0, 2.0, 4000.0,
        1.0, 1.0, 1.0, 5.0,
        3.0, 4.0, 2.0, 1.5,
        40.0, 1.0, 80.0, 2.0,
    ],
    dtype=np.float64,
)

INDEX = {name: i for i, name in enumerate(CHANNELS)}


def residual(obs: np.ndarray, pred: np.ndarray) -> np.ndarray:
    obs = np.asarray(obs, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    return (obs - pred) / SIGMA


def encode(obs: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Return a (512,) uint8 vector of 0 and 1."""
    obs = np.asarray(obs, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    z = residual(obs, pred)
    bits = np.zeros(N_BITS, dtype=np.uint8)
    mid = (LO + HI) / 2.0
    span = np.maximum(HI - LO, 1e-6)
    for c in range(N_CH):
        b = c * BITS_PER
        bits[b] = 1
        if z[c] > 0:
            bits[b + 1] = 1
        mag = int(min(8, max(0, abs(z[c]))))
        if mag:
            bits[b + 2 : b + 2 + mag] = 1
        vn = float(np.clip((obs[c] - LO[c]) / span[c], 0.0, 0.999))
        vb = int(vn * 8)
        if vb:
            bits[b + 10 : b + 10 + vb] = 1
        if obs[c] >= mid[c]:
            bits[b + 18] = 1
        else:
            bits[b + 19] = 1
    return bits


def as_akida_batch(bits: np.ndarray) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.uint8)
    if bits.ndim == 1:
        bits = bits.reshape(1, -1)
    if bits.shape[1] != N_BITS:
        raise ValueError(f"expected {N_BITS} bits, got {bits.shape[1]}")
    if not np.isin(bits, (0, 1)).all():
        raise ValueError("spike packet contains values other than 0 and 1")
    return bits.reshape(bits.shape[0], 1, 1, N_BITS)


def residual_pop(bits: np.ndarray, channels: tuple[int, ...]) -> int:
    """How many residual-thermometer spikes are set on these channels."""
    bits = np.asarray(bits)
    total = 0
    for c in channels:
        b = c * BITS_PER
        total += int(bits[b + 2 : b + 10].sum())
    return total
