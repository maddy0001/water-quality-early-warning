# Segment 08 – The early warning on an Arduino Uno, every part optional

**Status:** done and verified in the cloud (reference, PC build, and the real Uno firmware on a simulated
Uno); the board numbers come from **your Uno** (section 4) · **Builds on:** S4 (fault checks, triage),
S5 (false-alarm budget, adaptive threshold), S6 (deviation explainer) · **Feeds:** S9 (dashboard and live demo)
**Proves:** contribution **C9** – the whole decision chain (detector, adaptive threshold, fault triage,
explanation) runs on a ₹350 8-bit Arduino Uno and gives **bit-identical** answers to the laptop;
experiment **E11**

This replaces the earlier ESP32 version (there was no ESP32). Version 1 and S0–S7 are unchanged.

## 1. Goal

Paper 1 needed a laptop: a 310,602-weight BiLSTM, an RF + XGBoost + SVM ensemble, Python. A utility
station has a small controller. This segment puts the S4–S6 decision chain on the smallest common one,
an Arduino Uno (ATmega328P: **2 KB RAM**, 32 KB flash, 8-bit CPU, no hardware floating point), and asks:

1. Does it fit, and how fast is it?
2. How good is it, scored exactly like the S5 pipeline on GECCO-Triage?
3. Does the chip compute *exactly* the same as the laptop – every alarm, every sensor alert, every bit?

And, from your requirement: **nothing may break when a part is missing** – no board, no sensor, no
data files, no compiler, an unplugged cable in the middle of a run.

## 2. What happens

```
 pre-test data ─► limits (S4) + 625-weight network (Keras, once) ─► edge_spec_<station>.json ─► edge_model.h (flash)
                                                                          │
 GECCO-Triage stream ─► counts (16 per normal unit, integers) ─┬─► reference.py  (Python, the definition)
                                                               ├─► desktop build (same C++ on the PC)
                                                               ├─► simulated Uno  (real firmware, simavr, Linux)
                                                               └─► Arduino Uno over USB  (found automatically)
        laptop sends "D,<9 counts>"  ──►  device answers "R,<t>,<log-odds bits>,<θ bits>,alarm,water,sensor,mask,top,µs,…"
        when an alarm episode ends   ◄──  the laptop answers "V,<false alarms>" (the operator's verdict)
                                                               │
                                     compare every minute with the reference · score the device's own alarms
```

**Sized for 2 KB.** The laptop sends each reading as an integer count of 1/16 of the sensor's normal
unit (from S4). So all sums are exact integers, and one normal unit is 16 counts on every sensor.
The Uno keeps the last 16 minutes plus sums of the last four complete quarter-hours. That is 705
bytes for 9 sensors. The network weights and limits stay in flash.

**Every minute, on the device** (`edge_core.h`, same steps as `reference.py`):

1. Carry a missing reading forward up to 5 minutes.
2. Per sensor: `z` (vs. the previous 15 minutes), `d60` (vs. the last four quarter-hours), `r15`
   (range of the previous 15 minutes), all in normal units; the move ratio `m = |z| / move limit`.
3. Detector: 29 → 16 → 8 → 1 network, int8 weights (712 bytes), 32-bit float arithmetic.
4. Adaptive threshold (S5): θ drops a little every minute. When an alarm episode ends the device asks
   for the verdict, and each false alarm raises θ by 0.5.
5. Fault checks (S4): stuck, dropout, spike. Lone-jump rule: one sensor jumps alone means the sensor,
   not the water.
6. Triage: a suspect stays suspect for 60 minutes. A water alarm is withdrawn when no other sensor
   moved beyond normal. Output: water alarm, sensor alert (which sensors), top sensor.

**Bit-identical.** Everything is integer up to one conversion, then 32-bit floats in a fixed order,
with no fused multiply-add. The Python reference does the same operations in `numpy.float32`, so
results agree to the last bit (checked, section 5). Floats travel as their 8 hex digits, not as text.

**Detect, then choose.** Each part is used only if it answers. Otherwise the computer does its job and
says so in one line:

| Part | If it answers | If it is missing or stops |
|---|---|---|
| Uno | runs the edge logic | the laptop runs the same logic (identical answers) |
| Board unplugged mid-run | – | replay stops cleanly, the minutes done are compared, the laptop's answers stand for the rest |
| GECCO files | real data | the **simulated station** (`data.fallback: simulated`), said on screen |
| C++ compiler / simavr | PC build / simulated Uno checks | skipped (optional) |
| Probe, touch, DHT11 | live demo (S9) | replay / simulated stream, keyboard button, left out |
| RFID | – | not used (its pins are not soldered); optional later |
| Ollama (S7) | LLM reports | template reports |

`python -m ewsv2 board` shows this decision for your machine.

## 3. Files added or changed

| Path | Purpose |
|---|---|
| `ewsv2/edge/reference.py` | The Uno algorithm in Python (counts, float32) – the definition |
| `ewsv2/edge/export.py` | Counts, limits, network training, int8 weights, TFLite size check, `edge_model.h` writer |
| `ewsv2/edge/hil.py` | Uno over USB (port found automatically), PC build, simulated Uno, replay, comparison |
| `ewsv2/edge/board.py` | Finds the Arduino, asks which sensors answer, prints what the system will use |
| `ewsv2/edge/run.py` | The S8 experiment and report |
| `ewsv2/edge/model/` | Shipped edge models (2018, 2019, simulated) + TFLite files |
| `ewsv2/edge/firmware/ewsv2_edge/` | **The Arduino sketch**: `ewsv2_edge.ino`, `edge_core.h`, `edge_protocol.h`, `edge_sensors.h`, `edge_model.h` |
| `ewsv2/edge/firmware/desktop/desktop_main.cpp` | The same C++ for a PC |
| `ewsv2/edge/tools/uno_sim.c` | Simulated Uno with simulated DHT11, touch and probe (Linux, simavr) |
| `ewsv2/hardware/` | `HARDWARE_CHECK.md`, `wiring_guide.html`, the check sketch `ewsv2_hwcheck` |
| `ewsv2/data/simulated.py` | The simulated station (source `simulated`) |
| `ewsv2/cli.py` | Commands `edge`, `board`; `data.fallback` |
| `ewsv2/configs/s8_edge.yaml` | Targets, port (`auto`), minutes, fallback |
| `ewsv2/tests/test_edge.py` | 12 tests (129 in total): bit-identical C++, simulated Uno, unplugged board, no board, fallback data |

## 4. Run it (PowerShell)

Keep the wiring exactly as it is from the hardware check (touch D3/D4, DHT11 A1, probe A0/A2).

### 4a. Python part (no board needed)

```powershell
cd "C:\Users\THE GOD\Documents\Github\water-quality-early-warning"
venv2\Scripts\Activate.ps1

# 1. Clean install of the S8 code
Remove-Item -Recurse -Force ewsv2
Expand-Archive -Path "$HOME\Downloads\ewsv2_S8.zip" -DestinationPath . -Force
pip install pyserial

# 2. Check: must print  ewsv2 2.0.0.dev0  (built up to segment S8)
python -m ewsv2 info
python -m pytest ewsv2\tests -q        # 125 passed, 4 skipped (the 4 need g++ / simavr – fine on Windows)

# 3. Edge algorithm on both stations, without the board (about 1–2 min each)
python -m ewsv2 edge --config s8_edge
python -m ewsv2 edge --config s8_edge --set data.source=gecco2019
```

On Windows the `desktop` and `sim` lines say `skipped`. That is expected: the cloud ran them (section 5).

### 4b. Put the early-warning program on the Uno (5 minutes)

1. Plug in the Uno. Arduino IDE: **File → Open** →
   `ewsv2\edge\firmware\ewsv2_edge\ewsv2_edge.ino` (four more tabs open with it; that is right).
2. **Tools → Board → Arduino Uno**, **Tools → Port → COM7**, **Upload**. Expect roughly 18–20 KB of
   program storage (about 60%) and about 1,150 bytes of dynamic memory (about 56%).
3. **Close the Serial Monitor.** Only one program can use the port, and now it is Python.

This replaces the hardware check program on the board. To run the check again later, upload
`ewsv2\hardware\ewsv2_hwcheck\ewsv2_hwcheck.ino`.

### 4c. Which parts answer

```powershell
python -m ewsv2 board
```

Expected (your numbers will differ):

```
Looking for the Arduino ...
  board on COM7
  parts: probe yes, touch yes, dht yes, rfid no
  probe 1009 (off 0) | DHT11 32 C, 87 % | touches 0
  ...
What the system will use:
  board  Uno runs the edge logic
  probe  live water readings from the probe
  touch  touch sensor = operator's 'false alarm' button
  dht    station temperature and humidity
  rfid   not used (needs soldered pins; optional)
```

Try it: pull the DHT11's white wire and run it again. The DHT11 line switches to "left out", and nothing else changes.

### 4d. Hardware-in-the-loop: GECCO data through the real Uno

```powershell
# finds the Uno by itself; one week of the test period, about 10 minutes
python -m ewsv2 edge --config s8_edge

# quicker check (one day, ~1.5 min) or the whole test period (139,566 minutes, ~1.5 h)
python -m ewsv2 edge --config s8_edge --set edge.minutes=1440
python -m ewsv2 edge --config s8_edge --set edge.minutes=0

# save (the hardware guides are in ewsv2\hardware and go to GitHub with this)
git add ewsv2
git commit -m "ewsv2 S8: Arduino Uno edge, bit-identical to the laptop, every part optional"
git push
```

**Robustness test (optional):** while the one-day run is going, pull the USB cable. The run must
end with `partial: stopped after N of 1440 minutes ... – the laptop's answers stand for the rest`
and still write its report. Plug the board back in afterwards.

Paste the console output of 4a, 4c and 4d to me. The `uno` line is the result for the paper.

## 5. Expected output

### 5a. The edge algorithm vs. the S5 laptop pipeline (same on every computer)

```
Edge algorithm on gecco2018: 625 weights, int8 712 B, float32 2500 B, TFLite int8 3504 B
Uno firmware: 20264 B flash of 32256, 1153 B RAM of 2048 before the stack
  system                                          real  contam. faults->sensor ->water waterFA/d sensorFA/d
  edge int8 + fixed                             41/46    22/30           0/24       17      2.71       0.00
  edge int8 + fixed + edge triage               41/46    22/30          18/24        4      1.13       1.76
  edge int8 + adaptive                          41/46    17/30           0/24       14      1.08       0.00
  edge int8 + adaptive + edge triage            41/46    17/30          18/24        2      0.48       0.91
  edge float32 + adaptive + edge triage         41/46    17/30          18/24        2      0.47       0.93
  S5 laptop pipeline (stacking (learned))       34/46    18/30          18/24        1      0.45       0.82
  int8 vs float32: alarm agreement 0.99977

Edge algorithm on gecco2019: 481 weights, int8 568 B
  edge int8 + adaptive + edge triage             9/12     1/18          17/24        0      0.05       1.65
  S5 laptop pipeline (C alone)                  12/12     7/18          13/24        1      0.33       1.87
```

(The console prints all 8 edge rows for each station.)

### 5b. Targets (cloud)

```
gecco2018
  desktop  139566 min: alarm 1.00000 water 1.00000 sensor 1.00000 top 1.00000 | bit-identical 1.00000 | 3 us/min
  sim       10080 min: alarm 1.00000 water 1.00000 sensor 1.00000 top 1.00000 | bit-identical 1.00000 | 25763 us/min (max 26648), state 705 B, RAM never used 369 B
gecco2019
  desktop   31650 min: ... all 1.00000 | bit-identical 1.00000
  sim       10080 min: ... all 1.00000 | bit-identical 1.00000 | 19317 us/min (max 20056), state 477 B, RAM never used 669 B
```

`sim` is the compiled Uno firmware (the same file you upload) running on a cycle-exact simulated
ATmega328P at 16 MHz. Its timing is what the real chip needs.

On your laptop with the Uno, expect

```
  uno      10080 min: alarm 1.00000 water 1.00000 sensor 1.00000 top 1.00000 | bit-identical 1.00000 | ~25800 us/min (max ~26600), state 705 B, RAM never used ~350-400 B, ~40-60 ms round trip
```

### What the numbers say

1. **C9 – the decision chain fits on an Arduino Uno and agrees with the laptop to the last bit.** In
   every one of 191,376 minutes checked (both test periods on the PC build, two weeks of the real
   firmware on the simulated Uno), the log-odds, the threshold θ, the water alarm, the sensor alert,
   the faulty-sensor mask and the top sensor are identical. On the Uno it uses 20 KB of 32 KB
   flash and 1.15 KB of 2 KB RAM; even at its deepest the stack leaves about 370 bytes never touched. It
   needs 26 ms per minute of data, so it is busy 0.04% of the time.
   The network has 625 weights, 497 times fewer than paper 1's BiLSTM.
2. **On 2018 the Uno chain catches more than the laptop pipeline, at a similar false-alarm rate.**
   41/46 real events and 17/30 contaminants at 0.48 false alarms/day, against S5's 34/46 and 18/30 at
   0.45. Faults are handled as well: 18/24 go to the maintenance queue and only 2 raise a water alarm.
3. **Triage makes it usable.** Without it the same detector sends 14 of 24 faults to the water alarm.
   With the edge fault checks and the lone-jump rule that drops to 2, and false alarms halve
   (1.08 → 0.48/day), without losing an event.
4. **The adaptive threshold keeps the promise on the chip.** A fixed 1/day threshold set in summer
   gives 1.13/day in winter. The adaptive one gives 0.48, at the cost of 5 contaminants (22 → 17).
5. **int8 costs nothing measurable.** 99.98% of alarm minutes agree with float32 weights; the same
   events are caught.
6. **2019 is the honest limit.** A network this small, trained on 2019's few pre-test events, needs a
   very high threshold to stay within the budget: 9/12 real events and 1/18 contaminants, against
   12/12 and 7/18 for the laptop's classifier ensemble. Its triage still works (17/24 faults, 0.05 water
   false alarms/day). A 625-weight edge detector is a station-specific model.
7. **Nothing breaks.** Without the GECCO files the same command runs on the simulated station and
   says so. Without the board, a compiler or simavr, those checks are skipped with one line each. An
   unplugged board ends the replay cleanly (tested).

## 6. Check it worked

- `python -m ewsv2 info` prints `built up to segment S8`.
- The 5a tables match exactly.
- `python -m ewsv2 board` finds the Uno and lists the parts.
- The `uno` line shows 1.00000 everywhere and a `RAM never used` number.

| Problem | Fix |
|---|---|
| `no Arduino found on any USB port` | USB cable in? Driver (CH340) as in the hardware check; try `--set edge.port=COM7` |
| `COM7: cannot open (... Access is denied ...)` | Close the Arduino Serial Monitor (or any program using the port) |
| `no answer from the ewsv2_edge sketch (it runs another program ...)` | Upload `ewsv2_edge.ino` (4b); the hardware check program is still on the board |
| `the board runs the gecco2019 model` | Upload the sketch from this zip again (its `edge_model.h` is the 2018 one) |
| Upload error `stk500_getsync(): not in sync` | Board *Arduino Uno*, port COM7, Serial Monitor closed; press reset as the upload starts |
| `pyserial is missing` | `pip install pyserial` |
| `partial: stopped after N ...` | The cable moved or the board reset; plug in and run again (the report is still written) |

## 7. Viva notes

**Why an 8-bit Arduino and not a Raspberry Pi?** An Uno is the most common, cheapest controller
(₹350), and it can only run code that is truly small. If the decision chain fits here, it fits
in any station controller. A Pi is a small Linux computer and proves much less.

**How does 2 KB of RAM hold a 60-minute history?** It does not keep 60 minutes. It keeps 16 minutes
of 2-byte counts plus the sums of four quarter-hours. The 60-minute mean comes from those sums,
which is exact in integers. That is 705 bytes for 9 sensors. Weights and limits sit in flash.

**What does "bit-identical" mean, and why does it matter?** The 32-bit pattern of every log-odds value
and threshold from the chip equals the laptop's. That is checked, not assumed: the PC build over
both test periods and the real firmware on a simulated Uno for two weeks of data. So every result
measured on the laptop is also a result for the chip. It works because the sums are exact integers
and all float operations happen in the same order on both, with no fused multiply-add.

**Is a simulated Uno a real test?** It runs the same compiled file you upload, instruction by
instruction at 16 MHz, including the AVR's own floating-point routines. Your board run (4d) is the
physical confirmation.

**Where do the operator's verdicts come from?** In the replay, the laptop answers from the labels at
the moment the device asks. In the live demo (S9) the touch sensor is the "false alarm" button. With
no verdict at all, θ only drifts down slowly: more sensitive, never blind.

**What if a sensor dies on the station?** Its own fault checks flag it within minutes (stuck, dropout,
spike) and raise a sensor alert. Alarms keep working on the other sensors. In the demo, pulling a
probe wire shows exactly this.

**What if the GECCO files are gone in five months?** The commands fall back to the simulated station
and say so. The S10 hand-over also puts the data and exact packages on the pendrive.

## 8. Next segment

**S9 – Dashboard and live demo.** One page for the operator: the replayed GECCO stream or the live Uno
(probe in a glass of water: salt = event, pulled wire = sensor fault), water alarms and sensor alerts,
the top sensor, the S7 incident report, and the touch sensor (or a button) sending the verdict back.
Every part optional, chosen by detection.
