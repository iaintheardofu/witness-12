# Witness-12 package

Operator notes for the Python package. The repository README and
`TECHNICAL_OMNIBUS.md` sit in the parent directory.

Twelve physical AKD1000 specialists, one spacecraft ledger, and a language
report that can print only numbers the ledger already holds.

The bus in this directory is synthetic. It is not a downlink and it is not a
flight result. The chips, when `run_board.py` is used, are the AKD1000s on
this host.

## What is actually new

On-chip few-shot learning on one AKD1000 is a BrainChip feature. A two-board
federated keyword study on AKD1000s has been published (Motta and Nanni,
2026). Spiking world models and spiking language models exist without this
board.

The measurement this directory is built to make is narrower:

Twelve resident models, one per AKD1000, each a different sensory specialist
of one bus ledger. A reserved chip is mapped first and then fit with eight
windows of a lookalike fault (heater load, command line still low). The
report of that fault uses four claim types. Every number in those sentences
is copied from the ledger. After the ledger accepts the new mode, the same
bus produces no sentence.

## Two paths

`python -m witness12` runs the host decoder and prints the transcript. It
does not open `/dev/akida`.

`python -m witness12.run_board` maps one model per chip with `hw_only`, fits
after the map, and writes `results/board.json`. It does not call the on-chip
power meter. A chip that does not separate its class is stored as a miss.

`python -m witness12.test_host` checks silence, the withheld fault name, and
that every printed number was copied from a claim.

## Measured on this host

`results/board.json` is one `hw_only` run. The power meter was not called.
`akida.devices()` returned all 12 (`PCIe/NSoC_v2/0` through `/11`) in 20.7 seconds.

Five binary specialists separated their own class from their rival, with quiet
at zero: eclipse exit, radio pass, payload on, safe hold, vibration.
Six did not. Their scores are in the JSON. A miss stays a miss.

The reserved chip, `PCIe/NSoC_v2/11`, was fit with eight novel windows after
it was mapped. On those windows the new-class potential rose from 24 to 36
and became the argmax. The same fit raised the new-class potential on the
commanded-heater windows from 51 to 61, so that lookalike also landed in the
new class. Quiet stayed at zero. The other eleven chips returned the same
control potentials after the fit as before it.

The host transcript from `python -m witness12` is a separate decoder. It is
not the chip table.

## What this will not say

No watt figure. No 12-times throughput. No statement that a missed chip
passed. The RTX Pro 2000 enclosure can appear on USB while the GPU stays off
the PCI bus. That GPU is not part of the result.
