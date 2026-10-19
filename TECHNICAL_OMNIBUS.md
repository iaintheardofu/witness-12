# Witness-12 Technical Omnibus

**System:** twelve resident AKD1000 specialists, one physical ledger, claim-typed speech
**Stimulus:** synthetic CubeSat bus. Not a downlink. Not a flight result.
**Recorded hardware run:** one `hw_only` pass, Akida SDK 2.19.3, batch 8, power meter not called
**Artifact:** `witness12/results/board.json`
**Date of the recorded run:** 2026-10-19

This document is the engineering record of the repository. Where a number describes a chip, it is copied from `board.json`. Where a behavior describes the host decoder, it is the behavior checked by `witness12/test_host.py`. The two are different programs and they do not get mixed in the claims below.

---

## 1. Purpose

Witness-12 is a small physical world model with a language interface that is mechanically narrow.

The ledger holds a 16-channel bus state. Each step it predicts the next sample from the last mode it accepted. The difference between that prediction and the new sample is the residual. Spikes are produced from the residual and from a few binary command flags. Twelve models, one per AKD1000, are specialists for twelve situations. The host prints a sentence only when a specialist fires or a residual is large, and every number in that sentence is copied from the sample, the prediction, the residual, or the specialist score.

The reserved specialist is there for a fault that looks like a fault the crew already knows. A commanded heater and an uncommanded heater share a temperature rise, a voltage sag, and a current rise. They differ in the heater command bit. The research question on the silicon is whether a last-layer 1-bit prototype, fit after the model is mapped, can take eight windows of the second situation without also claiming the first.

The recorded run answers a narrower question cleanly, and answers that research question in the negative. Section 12 has the numbers.

## 2. What this is, stated as measurements

These statements match the recorded run.

1. Twelve AKD1000 devices were enumerated in one process: `PCIe/NSoC_v2/0` through `PCIe/NSoC_v2/11`. Enumeration took 20.732 seconds.
2. Twelve different models were mapped with `hw_only`. The run recorded no map error and no forward error.
3. A quiet packet, all zeros in the sparse code, produced class-potential sums of zero on every chip that was scored for quiet.
4. After the reserved chip was fit with eight windows of the new class, eleven other chips were forwarded again on the same `eclipse_entry` packet. All eleven integer score pairs were identical to the pairs recorded before that fit.
5. On the reserved chip, the new-class sum on the novel packet moved from 24 to 36. The sums of the two previously fit classes on that same packet stayed at 29 and 27. The new class became the argmax.
6. On the reserved chip, the new-class sum on the commanded-heater packet moved from 51 to 61. The previously fit class sums stayed at 55 and 52. The new class became the argmax of the lookalike as well.
7. Five binary specialists satisfied the separation rule in Section 10. Six did not.

These statements are outside the recorded run, and this repository does not make them.

- No energy, watt, or joule figure. The on-chip INA path was not called. On this host that call has failed initialization and stalled later forwards, so the run leaves it alone.
- No claim that twelve chips ran their forwards at the same time. Earlier measurements on this same host showed the runtime servicing device DMA one chip at a time. This repository does not repeat that study and does not report a speedup.
- No claim that the host transcript's "chip 11 matched 2 of 2 private spikes" was computed on an AKD1000. That line is the exact-mask decoder in `catalog.py`. Its source field is `host`.

## 3. Prior art this design was written against

The repository is a measurement harness around a specific crew. It is not a claim to have invented on-chip learning, spiking language models, or world models.

- BrainChip documents on-chip few-shot learning for AKD1000 as a last-layer operation: `FullyConnected`, binary weights, binary inputs. Edge Impulse exposes the same feature as an Akida learning block. The recorded run uses that feature. It does not originate it.
- Motta and Nanni, 2026, report federated few-shot class learning on two physical AKD1000 nodes (Raspberry Pi 5 boards), exchanging neuron-level weight maps for keyword classes. That is two devices and a keyword task. Witness-12 is twelve resident models on one host, a bus ledger, and a different packet.
- Spiking language models (SpikeGPT and later elastic-spike and spiking-LM lines of work) generate text with spiking or binary activations, generally in software or on FPGA-class assumptions. Witness-12 does not generate tokens. Its sentences are filled from a ledger.
- Spiking world models for reinforcement learning, and event-camera plus language-model systems, cover pieces of "predict the world" and "talk about events." They are not this hardware crew, and this repository does not benchmark against them.

The measurement that is actually new in this tree is the twelve-model residency plus the before/after control on the other eleven chips, together with an honest miss on lookalike discrimination. A literature search that would support a patent claim or a priority claim has not been done here.

## 4. Machine the run was recorded on

| Item | Recorded fact |
| --- | --- |
| Board | BIOSTAR TB250-BTC PRO |
| CPU at the time of the surrounding hardware work | Intel Core i7-7700 |
| OS | Ubuntu 22.04, kernel 5.15.0-198-generic |
| Cards | 12× AKD1000, PCI IDs `1e7c:bca1`, each link 5.0 GT/s x1 |
| Runtime identity | `PCIe/NSoC_v2/0` … `/11`, version string `BC.00.000.002`, IP v1, 78 neural processors, 300 MHz |
| SDK | `akida` 2.19.3 |
| Nodes | `/dev/akida0` … `/dev/akida11` |
| Discrete GPU | A Razer Core X V2 enumerated on USB (`1532:0f51`). The GPU was not on the PCI bus. `nvidia-smi` was not installed. The run does not use it. |

The Python environment that executed `run_board.py` was the host's existing Akida environment. NumPy was 2.1.3 under Python 3.10.12. The host decoder needs NumPy only.

Operator constraints that the run obeyed: no driver reload, no firmware write, no power-meter call, batch size 8, one process.

## 5. Architecture

```
synthetic bus sample
        |
        v
CPU ledger  ---- predicts the next 16-vector from the accepted mode
        |
        v
residual z = (observed - predicted) / sigma
        |
        +---- dense 512-bit packet --> host exact-mask decoder --> claim text
        |
        +---- sparse 256-bit packet --> 12 mapped AKD1000 models
                                              |
                                              v
                                    class potential sums
                                    (board.json)
```

Prediction, mode memory, and sentence assembly stay on the CPU. The AKD1000s receive a spike packet and return potentials. They do not integrate the bus ODE, and they do not choose the words.

There are two packets because they serve two jobs.

The dense packet is a full description of one step: validity, residual sign, residual magnitude, value thermometer, and an explicit high/low bit so that a command value of zero is a spike. The host decoder reads it. A 1-bit prototype learner does badly on it. During development, MetaTF software forwards on this dense code produced class sums that were close and often on the wrong side of the decision, including on quiet, because most bits are on in both classes (the baseline value thermometers). `test_akida_sim.py` is that probe. It is off unless `WITNESS_DENSE_PROBE=1`. The board was not run on the dense packet.

The sparse packet is what `run_board.py` ships. A channel contributes spikes only when its residual is non-zero. A binary flag contributes a population only when it is high. Quiet is the zero vector. Section 7 specifies the bits.

## 6. Ledger

Implementation: `witness12/ledger.py`, `witness12/codec.py`, `witness12/catalog.py`.

### 6.1 Channels

Index order is fixed. Sigma scales the residual. The spans (`LO`, `HI`) exist only to place the dense value thermometer. They are not a calibration of a flown sensor.

| Index | Name | Baseline | Sigma |
| ---: | --- | ---: | ---: |
| 0 | bus_voltage_v | 7.80 | 0.08 |
| 1 | battery_temp_c | 22.0 | 0.50 |
| 2 | payload_current_a | 0.20 | 0.05 |
| 3 | wheel_rpm | 200 | 30 |
| 4 | radio_rssi | 0.10 | 0.05 |
| 5 | heater_command | 0 | 0.05 |
| 6 | eclipse | 0 | 0.05 |
| 7 | gyro_rate_dps | 0.02 | 0.05 |
| 8 | solar_current_a | 1.40 | 0.10 |
| 9 | tank_pressure_bar | 2.00 | 0.08 |
| 10 | vibration_g | 0.05 | 0.03 |
| 11 | wheel_current_a | 0.10 | 0.04 |
| 12 | payload_temp_c | 18.0 | 0.60 |
| 13 | radio_command | 0 | 0.05 |
| 14 | array_temp_c | 30.0 | 1.50 |
| 15 | bus_current_a | 0.40 | 0.06 |

Baseline is the sunlit idle bus: heater off, eclipse off, payload quiet, wheels at a low rate.

### 6.2 Prediction

The predictor is a mode table, not a learned dynamical system.

| Accepted mode | Next prediction |
| --- | --- |
| `eclipse_entry` or `eclipse_hold` | eclipse vector: eclipse 1, solar current 0.05 A, bus 7.40 V, other channels at baseline |
| `commanded_heater` | commanded-heater vector |
| `uncommanded_heater` | uncommanded-heater vector |
| anything else, including the initial state and `eclipse_exit` | baseline |

Residual:

```
z = (observed - predicted) / sigma
```

If exactly one specialist fires, the ledger sets the mode to that specialist's name. An `eclipse_hold` sample after `eclipse_entry` moves the mode to `eclipse_hold` even though no specialist fires, so the next prediction stays on the eclipse vector. That is the only silent assimilation.

The step size in the timeline is 30 seconds. It is a label on the transcript clock. The synthetic samples do not integrate continuous time.

### 6.3 Situations

Each situation is a full 16-vector. Only the overrides are listed. Everything else stays at baseline. `predicted_before` is the prediction the onset is scored against. For every situation except eclipse exit, that prediction is baseline. Eclipse exit is scored against the eclipse-hold vector, because the bus is leaving eclipse.

| Name | Overrides | Channels the specialist watches |
| --- | --- | --- |
| eclipse_entry | eclipse 1, solar 0.05 A, bus 7.40 V | eclipse, solar |
| eclipse_exit | eclipse 0, solar 2.05 A, bus 8.05 V, array temp 42 C | eclipse, solar |
| wheel_desat | wheel 2800 rpm, wheel current 0.70 A, gyro 0.30 dps | wheel rpm, wheel current |
| commanded_heater | heater command 1, battery 34 C, bus 7.30 V, bus current 0.90 A | heater, battery temp, bus voltage |
| radio_pass | RSSI 0.90, radio command 1, payload current 0.45 A | RSSI, radio command |
| payload_on | payload current 1.30 A, payload temp 27 C, bus 7.45 V | payload current, payload temp |
| safe_hold | payload current 0.02 A, wheel 0 rpm, wheel current 0.01 A, RSSI 0.02 | payload current, wheel rpm |
| gyro_tumble | gyro 3.20 dps, wheel 900 rpm | gyro |
| array_overcurrent | solar 2.60 A, array temp 62 C, bus 8.40 V | solar, array temp |
| coolant_swing | tank 3.10 bar, vibration 0.55 g, payload temp 8 C | tank, vibration |
| vibration_burst | vibration 1.40 g | vibration |
| uncommanded_heater | heater command 0, battery 36 C, bus 7.10 V, bus current 0.95 A | battery temp, bus voltage, heater |

Eclipse exit is a one-step transient, brighter and warmer than idle, dimmer and cooler than the array-overcurrent fault. The uncommanded heater is the commanded heater's thermal and current shape with the command left at zero and the temperature and voltage pushed a step further.

### 6.4 Timeline the host decoder runs

`engine.py` walks this list. "Spoken" means the claim renderer returned sentences. "Silent" means the residual thermometers were cold and no specialist fired.

| t (s) | Sample | Decoder result |
| ---: | --- | --- |
| 30 | quiet | silent |
| 60 | quiet | silent |
| 90 | eclipse_entry | chip 0, eclipse entry, 9 of 9 private spikes |
| 120 | eclipse_hold | silent. Prediction now matches the eclipse vector. |
| 150 | eclipse_exit | chip 1, eclipse exit, 4 of 4 private spikes |
| 180 | quiet | silent. Mode `eclipse_exit` predicts baseline. |
| 210 | uncommanded_heater | spoken as an unexplained residual. The name is withheld. |
| — | learn | host stores 8 windows for chip 11. No claim sentence. |
| 240 | uncommanded_heater | chip 11 fires. The label is introduced as a name for those windows. |
| 270 | uncommanded_heater | silent. The mode now predicts this vector, so the residual is gone. |

`test_host.py` locks this pattern: four spoken observation steps, five silent observation steps, the string `uncommanded_heater` absent from the transcript before the learn marker, and the grounding check in Section 8.

## 7. Spike packets

### 7.1 Dense packet, 512 bits, host only

`codec.encode(obs, pred)`. Sixteen channels, 32 bits each.

| Bits | Meaning |
| --- | --- |
| 0 | sample valid (always 1 once a sample exists) |
| 1 | 1 if the residual is positive |
| 2:10 | thermometer of `floor(abs(z))`, capped at 8 |
| 10:18 | thermometer of the raw value across `[LO, HI)`, eight steps |
| 18 | 1 if the raw value is at or above the span midpoint |
| 19 | 1 if the raw value is below the midpoint |
| 20:32 | left at 0 |

Bit 19 is why a heater command of zero is visible. A prototype that only learns active spikes can see "command is low" as a spike instead of as a hole.

Akida's input view of this packet, if it were used, would be shape `(n, 1, 1, 512)`, `uint8` values in `{0, 1}`, `input_bits=1`. The board path does not feed this tensor.

### 7.2 Sparse packet, 256 bits, chips

`sparse.sparse_packet(kind)`. Built from the same observation and the same `predicted_before` vector the catalog uses for an onset. The live ledger prediction is what the host timeline uses. The board run scores onset packets, not the full timeline. Those are the packets whose spike counts are in Section 12.4.

| Region | Contents |
| --- | --- |
| `c*8 + (mag-1)` | one bit, the magnitude bin of channel `c`, with `mag = min(8, floor(abs(z)))`. No bit if `mag` is 0. This is one-hot, not a thermometer, so a larger residual is not a superset of a smaller one. |
| `128 + c*4` for 4 bits | four copies of "channel c moved", set together with the magnitude bit |
| 192:208 | sixteen 1s if `heater_command > 0.5` |
| 208:224 | sixteen 1s if `eclipse > 0.5` |
| 224:240 | sixteen 1s if `radio_command > 0.5` |
| 240:256 | unused, left at 0 |

A flag that is low adds nothing. Quiet, whose residual is zero and whose flags are low, is the zero vector. The recorded spike counts:

| Packet | Popcount |
| --- | ---: |
| eclipse_entry | 31 |
| eclipse_exit | 20 |
| wheel_desat | 15 |
| commanded_heater | 36 |
| radio_pass | 31 |
| payload_on | 15 |
| safe_hold | 20 |
| gyro_tumble | 10 |
| array_overcurrent | 15 |
| coolant_swing | 15 |
| vibration_burst | 5 |
| uncommanded_heater | 15 |
| quiet | 0 |

The commanded heater has 36 spikes and the uncommanded heater has 15. The difference is largely the 16-spike heater-command population, plus magnitude bins that do not match. The uncommanded packet is not a subset in the one-hot magnitude code, but it shares the "these channels moved" populations on the thermal and electrical channels. That shared support is what the reserved class locked onto. Section 13 discusses the consequence.

## 8. Host decoder

`HostCrew` in `catalog.py`. This is an exact mask, computed on the CPU, over the dense packet. It is the thing that produces the transcript. It is not a model of the AKD1000 potentials.

For each specialist the private mask is:

```
mask = onset_packet AND NOT quiet_packet AND NOT rival_onset_packet
mask = mask restricted to bits 0:20 of the specialist's own channels
```

Rivals:

| Specialist | Rival used to cut the mask |
| --- | --- |
| eclipse_entry | eclipse_exit |
| eclipse_exit | eclipse_entry |
| wheel_desat | gyro_tumble |
| commanded_heater | uncommanded_heater |
| radio_pass | payload_on |
| payload_on | radio_pass |
| safe_hold | wheel_desat |
| gyro_tumble | wheel_desat |
| array_overcurrent | eclipse_exit |
| coolant_swing | vibration_burst |
| vibration_burst | coolant_swing |
| uncommanded_heater | commanded_heater |

A specialist matches when every 1 in its mask is 1 in the packet. Among full matches, the winner is the specialist with the hottest residual thermometer on its own channels, then the larger mask. A tie on that key fires nobody. The reserved mask is ignored until `learn_reserve()` has been called, which is how the name stays out of the transcript at t=210 s.

The score reported in a host `CHIP` line is `(got, need)`: how many private spikes matched, and how many the mask contains. Example from the passing host run: eclipse entry reports 9 of 9, eclipse exit reports 4 of 4, the reserved class reports 2 of 2.

`test_each_onset_fires_only_its_specialist` builds every onset with `spike_pair`, opens the reserved class, and requires that the only firing name is that onset's own specialist.

## 9. Sentences

`language.py`. A `Claim` is a kind, a text, and a tuple of the number strings that text is allowed to contain. The renderer does not pull numbers from anywhere else. `ungrounded_numbers` finds every integer or decimal in the text and reports any that are missing from the tuple. `test_host.py` requires that list to be empty for a full timeline.

The kinds that can appear:

| Kind | Contents |
| --- | --- |
| OBSERVED | timestamp and the channels whose absolute residual is at least 1, plus heater command and eclipse even when their residual is small |
| PREDICTED | the predicted value of each hot channel |
| RESIDUAL | `z` of each hot channel, two decimal places |
| CHIP | either "no specialist fired" or one line per firing specialist with chip index and coverage |
| LEARNED | only on the first spoken step after the reserved class is stored, and only if that class fired. The sentence says the label names the windows and is not a diagnosed cause. |
| HYPOTHESIS | if nothing fired: withhold a fault name. If the reserved class fired: the windows share a temperature rise and a voltage sag while the command stayed low, and the label should be dropped if a later command bit is high. Otherwise: the specialist name restates a class label and does not identify a root cause. |
| ACTION | recommend that an operator review the heater and power lines. The report does not send a command. |

Formats are fixed: values and residuals use two decimal places, the timestamp and the coverage integers use none. The grounding check is string equality against those formatted tokens, so a rounded value and the token in the sentence are the same string by construction. The test is still there so a later edit that pastes a literal number into a sentence fails.

There is no language-model call in this path. A paraphrase layer was considered for a GPU that is not on the bus. It is not implemented. If one is added later, it has to run the same grounding check and fail closed when a number appears that the claims do not list.

## 10. What was mapped onto each AKD1000

Implementation: `witness12/run_board.py`.

### 10.1 Model

Akida 1.0 model, built with the Python API, no TensorFlow graph.

```
InputData  input_shape=(1, 1, 256)  input_bits=1
FullyConnected  units=16 * n_classes
                weights_bits=1
                activation=False
                act_bits=1
AkidaUnsupervised
                num_weights=20
                num_classes=n_classes
                initial_plasticity=1.0
                min_plasticity=1.0
                plasticity_decay=0.0
                learning_competition=0.0
```

`activation=False` keeps the forward output in the potential domain so the class sums are comparable integers rather than a 1-bit spike. Neuron groups are contiguous: class `k` owns neurons `[16k, 16k+16)`. The class sum is the sum of those sixteen potentials, averaged across the eight identical windows in the batch. With identical windows the average is the window.

Binary specialists use `n_classes=2` (32 neurons). The reserved chip uses `n_classes=3` (48 neurons).

`num_weights=20` is the number of connections each neuron may learn. `plasticity_decay=0` and `min_plasticity=1` keep the learning rate from collapsing across the eight identical samples. That choice was made so the eight-window fit would still be plastic on the last sample. It is a configuration of this run, not a vendor default.

### 10.2 Labels

Binary chip, for specialist `name` with rival `RIVALS[name]`:

| Class | Fit packet |
| --- | --- |
| 0 | eight copies of the rival onset |
| 1 | eight copies of this specialist's onset |

Reserved chip, `PCIe/NSoC_v2/11`:

| Class | When it is fit | Packet |
| --- | --- | --- |
| 0 | before any novel window | eight copies of commanded_heater |
| 1 | before any novel window | eight copies of eclipse_entry |
| 2 | after the before-scores | eight copies of uncommanded_heater |

Class 1 on the reserved chip is a second already-known situation, so the new class is not the only pattern the chip has ever seen besides the lookalike. Eclipse entry was used because its sparse support (eclipse flag population plus solar and bus residuals) is far from the heater packets.

Fit is called only after `model.map(device, hw_only=True)`. A `hw_only` failure is recorded and that chip is not scored. The run does not fall back to a host forward and call it a chip result. In the recorded run every map returned `hw_only`.

Each chip has a 60-second `SIGALRM` budget around map, fit, and its own forwards. The recorded run did not hit the alarm. Enumeration of `akida.devices()` sits outside that per-chip budget and took 20.732 seconds.

### 10.3 Decision rule for a binary chip

A binary chip is marked `separated: true` only when all three hold:

```
own[1]    > own[0]
rival[1]  < rival[0]
quiet[1] <= quiet[0]
```

`own` is the class-sum pair on the specialist's own packet, `rival` on the rival packet, `quiet` on the zero packet. The rule is strict. A one-count inversion on the rival is a miss. Commanded heater missed on that basis (rival class sums 23 and 24).

### 10.4 Reserved-chip questions

After the class-2 fit, the report records three booleans:

| Field | Rule |
| --- | --- |
| `novel_after_argmax_is_new_class` | argmax of the novel sums is class 2, and class 2 is strictly above the other two |
| `commanded_stays_original_class` | argmax of the commanded-heater sums is class 0, and class 0 is strictly above the other two |
| `quiet_does_not_become_the_new_class` | the zero packet does not come out with class 2 as a strict winner over class 0 |

The control packet for the other eleven chips is `eclipse_entry`. Their class sums are stored before the class-2 fit and after it. `all_unchanged` is true only when every one of those eleven pairs matches.

### 10.5 APIs the run does not call

`enable_power`, floor programming, and the power meter are absent from `run_board.py`. `model.statistics` is not read. The driver is not reloaded. No FlatBuffer from another project is overwritten. `board.json` is the only result file this script writes, and it writes it under `witness12/results/`.

## 11. Protocol of the recorded run

1. Confirm twelve `/dev/akida*` nodes exist.
2. `akida.devices()` in the same process that will map them. The runtime's exclusive lock means a second process would see no devices.
3. For chips 0–10: build the 2-class model, `map(..., hw_only=True)`, fit rival as class 0, fit own event as class 1, forward own, rival, and quiet at batch 8.
4. For chip 11: build the 3-class model, map `hw_only`, fit commanded heater as class 0, fit eclipse entry as class 1, forward novel, commanded heater, and quiet.
5. Forward `eclipse_entry` on chips 0–10 and store the sums.
6. Fit eight uncommanded-heater windows as class 2 on chip 11.
7. Forward novel, commanded heater, and quiet on chip 11.
8. Forward `eclipse_entry` on chips 0–10 again.
9. Write `results/board.json`.

No sample was shuffled. The eight windows in a fit are copies of one packet. The study is "can these prototypes latch a repeated sparse pattern, and does that write disturb the other devices?" It is not a generalization study across noisy telemetry.

## 12. Recorded scores

Class sums are integers. For a binary chip the pair is `[class0, class1]`. For the reserved chip the triple is `[class0, class1, class2]`.

### 12.1 Binary specialists

| Chip | Device | Situation | Own [c0, c1] | Rival [c0, c1] | Quiet | Separated | Why a miss failed the rule |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 0 | NSoC_v2/0 | eclipse_entry | 50, 42 | 51, 36 | 0, 0 | no | own class prefers class 0 (50 > 42) |
| 1 | NSoC_v2/1 | eclipse_exit | 34, 41 | 53, 42 | 0, 0 | yes | |
| 2 | NSoC_v2/2 | wheel_desat | 25, 31 | 18, 22 | 0, 0 | no | rival also prefers class 1 (22 > 18) |
| 3 | NSoC_v2/3 | commanded_heater | 48, 58 | 23, 24 | 0, 0 | no | rival prefers class 1 by one count (24 > 23) |
| 4 | NSoC_v2/4 | radio_pass | 37, 55 | 34, 21 | 0, 0 | yes | |
| 5 | NSoC_v2/5 | payload_on | 20, 26 | 53, 49 | 0, 0 | yes | |
| 6 | NSoC_v2/6 | safe_hold | 34, 41 | 40, 22 | 0, 0 | yes | |
| 7 | NSoC_v2/7 | gyro_tumble | 21, 19 | 37, 24 | 0, 0 | no | own class prefers class 0 (21 > 19) |
| 8 | NSoC_v2/8 | array_overcurrent | 30, 38 | 42, 47 | 0, 0 | no | rival prefers class 1 (47 > 42) |
| 9 | NSoC_v2/9 | coolant_swing | 22, 35 | 10, 12 | 0, 0 | no | rival prefers class 1 (12 > 10) |
| 10 | NSoC_v2/10 | vibration_burst | 8, 15 | 32, 25 | 0, 0 | yes | |

Five passes: chips 1, 4, 5, 6, 10. Six misses: chips 0, 2, 3, 7, 8, 9. Quiet is zero everywhere in this table. Several misses are small inversions on an event the chip was not supposed to claim. The rule does not grade on a curve.

### 12.2 Reserved chip, NSoC_v2/11

| Packet | Before class-2 fit [c0, c1, c2] | After eight novel windows |
| --- | --- | --- |
| uncommanded_heater | 29, 27, 24 | 29, 27, 36 |
| commanded_heater | 55, 52, 51 | 55, 52, 61 |
| quiet | 0, 0, 0 | 0, 0, 0 |

Reading the columns:

- Class 0 and class 1 did not move, on either heater packet. The fit wrote into the class-2 neuron group.
- Class 2 on the novel packet rose by 12, from 24 to 36, and passed both older classes. `novel_after_argmax_is_new_class` is true.
- Class 2 on the commanded-heater packet rose by 10, from 51 to 61, and passed class 0 (55) and class 1 (52). `commanded_stays_original_class` is false.
- Quiet stayed at zero. `quiet_does_not_become_the_new_class` is true.

### 12.3 Control, eclipse_entry packet, chips 0–10

Before the class-2 fit, and again after it:

| Chip | [class0, class1] both times |
| ---: | --- |
| 0 | 50, 42 |
| 1 | 53, 42 |
| 2 | 28, 43 |
| 3 | 39, 35 |
| 4 | 37, 31 |
| 5 | 42, 46 |
| 6 | 38, 41 |
| 7 | 41, 32 |
| 8 | 52, 46 |
| 9 | 44, 38 |
| 10 | 39, 41 |

`control.all_unchanged` is true. These pairs are not the "own event" scores from Section 12.1. They are a second forward of one shared packet, used only as a stability control. Chip 0's own-event score happens to be the same pair because chip 0's own event is eclipse entry. The other rows differ from Section 12.1 on purpose.

### 12.4 Errors

`errors` in `board.json` is an empty list. No chip took the timeout branch. No exception string was stored.

## 13. What the reserved-chip numbers mean

The fit did what on-chip last-layer learning is supposed to do locally. Eight presentations of a sparse pattern raised the potential of the neuron group reserved for the new class, and they did not change the potentials of the other eleven devices on the control packet. Old class sums on the reserved chip itself also stayed put. The new group was additive.

The new group also answers to the commanded heater. That packet contains the channel-moved populations and the thermal and electrical magnitude bins that the novel packet contains, and it adds the 16-spike heater-command population. Neurons that learned the novel support still see a large active subset when the command flag is high. Their summed potential (61) exceeded the class that was actually fit on the commanded heater (55).

A 1-bit excitatory prototype can require a spike to be present. It cannot, in this configuration, require a spike to be absent. "Heater command is low" was encoded as the absence of the 192:208 population. Absence is invisible to a neuron that learned the spikes the novel packet does have. The dense host code tries to fix that by giving command-low its own spike (dense bit 19) and then cutting a private mask against the rival. That is why the host decoder separates the two heaters with a 2-spike mask, and why that result is labeled `source: host`.

Closing the gap on the chip would mean a sparse code in which the novel class has a positive spike population the commanded class does not have, large enough that dropping those spikes pulls class 2 below class 0 on the commanded packet, and small enough that quiet (which also has the command low) does not gain them. A conjunction spike — "temperature residual is hot and the command bit was observed low" — would do that, and it would also move the detection into the host encoder. This run does not add that bit, because the point of the run was to see what the unmodified sparse code does.

The binary misses have the same shape as this gap, at smaller scale. Vibration separates (own 8, 15 against a rival of 32, 25) with only five spikes in its packet. Gyro tumble, with ten spikes, prefers its negative class on its own packet (21, 19). Margin of a few counts is the whole decision. Treating a one-count rival inversion as a pass would have moved commanded heater into the pass column. The JSON keeps the strict rule so the table cannot be talked into a better result after the fact.

## 14. Host transcript versus chip table

A reader who runs `python -m witness12` sees chip 11 fire on the uncommanded heater and stay silent on the following step. That is the mask decoder plus the mode table. It demonstrates the sentence discipline and the "speak once, then predict the new mode" loop. It does not demonstrate that `PCIe/NSoC_v2/11` rejected the commanded heater.

A reader who opens `board.json` sees the opposite discrimination result on the silicon, and sees the eleven-chip control succeed. Both documents are in the repository so that the transcript cannot be quoted as the hardware result.

The action line is the same in every spoken host step, and it has no numbers in it. It recommends a human review. There is no actuator path, no radio command, and no "safe hold" command emitted as a packet to a vehicle.

## 15. File map

| Path | Role |
| --- | --- |
| `witness12/codec.py` | Channel table, sigma, dense 512-bit encoder |
| `witness12/sparse.py` | 256-bit chip packet |
| `witness12/catalog.py` | Situation vectors, rivals, private masks, `HostCrew` |
| `witness12/ledger.py` | Mode table, residual, assimilation |
| `witness12/language.py` | Claim types and the grounding check |
| `witness12/engine.py` | Timeline driver for the host decoder |
| `witness12/akida_backend.py` | MetaTF builder and software forward used by the dense probe |
| `witness12/run_board.py` | `hw_only` map, on-chip fit, JSON report |
| `witness12/test_host.py` | Silence, withheld name, grounding, one-specialist-per-onset |
| `witness12/test_akida_sim.py` | Optional dense-packet probe. Off by default. |
| `witness12/draw_architecture.py` | Regenerates `architecture.png` with drawn text |
| `witness12/architecture.png` | The figure in the README |
| `witness12/results/board.json` | The recorded run |
| `witness12/__main__.py` | Prints the host transcript |

`draw_architecture.py` uses DejaVu Sans from the usual Debian/Ubuntu font path and writes the PNG in place. The words in the figure are drawn by that script. They are not produced by an image model.

## 16. How to reproduce

Host decoder and tests, from the repository root, with NumPy importable:

```bash
python -m witness12
python -m witness12.test_host
```

Redraw the figure, which needs Pillow and DejaVu Sans:

```bash
python -m witness12.draw_architecture
```

Board run, only on a machine with the devices and the SDK, from the same environment that can `import akida`:

```bash
python -m witness12.run_board
```

That command overwrites `witness12/results/board.json`. Copy the recorded file aside first if the original integers should be kept. Expect `akida.devices()` to take on the order of ten to twenty seconds on this host. Expect the runtime to lock every `/dev/akida*` node it opens, and to stop at the first missing node. Do not start a second Akida process while it runs. Do not add a power-meter call to "complete" the measurement. A stalled forward after a failed INA init is a known failure mode on this host, and it is not a power number.

The dense probe, if someone wants to see the packet the board refused:

```bash
WITNESS_DENSE_PROBE=1 python -m witness12.test_akida_sim
```

That path builds software models. It does not map a device. On the development host it did not meet its own separation asserts. Leave it off for a cleanliness check of the repository. `test_host.py` is the check that is supposed to pass.

## 17. Limits

The stimulus is a hand-written vector per situation, repeated eight times when a chip is fit. There is no noise model, no sensor timestamp jitter, and no second example of the same fault with different magnitudes. A pass means the prototype latched that one pattern and rejected the one rival it was shown. It does not mean the specialist would survive a different eclipse depth or a different heater size.

The predictor is a table of the same vectors. Once a mode is accepted, the next prediction is the canonical vector of that mode, so a repeated sample has a zero residual and the host goes silent. A real bus would need a filter and a covariance. Those are not in this repository.

Eleven binary models and one three-class model are small relative to the 78 neural processors on an AKD1000. The run does not try to fill the chip, and it does not report throughput.

Class sums in the low tens, with decisions that flip on one or two counts, are a weak margin. The vibration pass (15 against 8) and the gyro miss (19 against 21) sit in the same range. Another fit, or a different `num_weights`, could move a borderline row. The published artifact is this configuration and this run, not an average over seeds. Akida's unsupervised rule can break ties inside the chip in ways this repository does not sweep.

The host that was measured had twelve cards on a desktop PCIe topology, each at x1. Results do not transfer to a single M.2 or Raspberry Pi HAT by multiplication, and they do not transfer to AKD1500 or AKD2000. The learning API used here is the AKD1000 `FullyConnected` plus `AkidaUnsupervised` path. Later silicon documents a different feature mix, and this code does not target it.

Sentence grounding stops a claim from containing a number the ledger did not supply. It does not make the hypothesis true. The hypothesis text is a template. On the host path the template is allowed to name `uncommanded_heater` only after the mask for that class is opened, and the sentence says the label is not a cause. On the chip path the same physical pattern also drives the commanded-heater score into the new class, so a report that trusted the chip's argmax alone would misname the commanded case. The ledger would still be wrong if it treated that argmax as a cause. The code does not do that.

## 18. A follow-on that would change the result

Three measurements would move the reserved-chip conclusion, and each one is a different system.

1. A sparse code with a positive spike for the conjunction of a hot thermal residual and a low heater command, fit under the same `hw_only` and eight-window rule, with the commanded packet and the quiet packet both re-scored. The conjunction bit has to be counted in the popcount table the way Section 7.2 counts today's bits.
2. The same twelve models forwarded on packets whose magnitude bins are shifted by one, to see which of the five passes survive a neighbor of the training pattern.
3. A second run on another day, same binary, same JSON schema, stored next to this file rather than over it, to see which borderline rows move.

Until those exist, the result to cite is Section 12: twelve resident models, eleven unchanged by a fit on the twelfth, quiet at zero, the new class acquired on its own windows, and the commanded heater acquired along with it.
