"""Sparse residual packet for the AKD1000 crew.

Only two kinds of spikes are emitted. A channel that moved contributes its
magnitude bin and four copies of a "this channel moved" bit. A binary
command that is high contributes sixteen copies of that flag. A quiet bus
is an all-zero packet.

This is the packet the chips see. The host ledger in codec.py is a separate,
denser code used for the claim transcript.
"""

from __future__ import annotations

import numpy as np

from witness12.catalog import observation, predicted_before
from witness12.codec import INDEX, residual

N_SPARSE = 256
_FLAG_AT = {"heater_command": 192, "eclipse": 208, "radio_command": 224}
_FLAG_COPIES = 16


def sparse_packet(kind: str) -> np.ndarray:
    obs = observation(kind)
    z = residual(obs, predicted_before(kind))
    bits = np.zeros(N_SPARSE, dtype=np.uint8)
    for c in range(16):
        mag = int(min(8, max(0, abs(float(z[c])))))
        if mag:
            bits[c * 8 + (mag - 1)] = 1
            bits[128 + c * 4 : 128 + c * 4 + 4] = 1
    for name, start in _FLAG_AT.items():
        if float(obs[INDEX[name]]) > 0.5:
            bits[start : start + _FLAG_COPIES] = 1
    return bits


def sparse_batch(kind: str, n: int = 8) -> np.ndarray:
    row = sparse_packet(kind)
    return np.repeat(row.reshape(1, 1, 1, N_SPARSE), n, axis=0)
