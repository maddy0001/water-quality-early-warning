# Hardware check – is every part working?

**Time:** about 45 minutes, done once. **Cost:** ₹0. **Needs:** the Arduino Uno clone, its USB cable, the
water probe (fork + its small blue board with the screw knob), the touch sensor, the DHT11, jumper wires,
the strip of male header pins, a glass of tap water and a spoon of salt.

**Not needed:** the RFID reader and tags (dropped from the project), the Raspberry Pi, the pendrive.

This check runs its own small program on the Uno. It needs **no Python, no ewsv2 code and no Arduino
library**. It was compiled for the Uno's chip and run on a simulated Uno with simulated sensors in the
cloud, so the program is known to work. If a part does not respond, the report names it and the program
keeps running.

---

## 0. Safety rules (read once)

1. **Unplug the USB cable whenever you change a wire.** Plug it back in only after checking the wiring.
2. **Never connect 5V straight to GND** (red to black with nothing in between). It can damage the board or
   the laptop's USB port.
3. **Only the fork goes into water.** Keep the blue boards, the Uno and the DHT11 dry. Hold the fork by
   its top (the black part with the 2 pins) so the pins and wires stay out of the water.
4. Rinse and dry the fork after each test. Water and current corrode it slowly. The program switches
   the probe on for only a moment every 2 seconds, so it corrodes less.

---

## 1. Install the Arduino software (one time, about 10 minutes)

1. Download **Arduino IDE 2** for Windows from https://www.arduino.cc/en/software and install it (all
   default options; say **Yes** to any driver question).
2. **CH340 driver.** Your Uno talks to the laptop through a CH340 chip. Plug the Uno in with the USB
   cable. Open **Device Manager** (Windows key, type *Device Manager*) and look under **Ports (COM & LPT)**:
   - `USB-SERIAL CH340 (COM3)` or similar: the driver is already there. Note the **COM number**.
   - Nothing there, or an unknown device with a yellow mark: install the driver *CH341SER.EXE* from the
     chip maker's site (https://www.wch-ic.com/downloads/CH341SER_EXE.html), click **INSTALL**, unplug and
     re-plug the Uno, and check again.
3. When the Uno is plugged in, its **ON** (power) LED lights. On a new board the **L** LED usually blinks.

## 2. Load the check program (5 minutes)

1. The program is in the repository: `ewsv2\hardware\ewsv2_hwcheck\ewsv2_hwcheck.ino` (the folder and file
   names must match; keep them as they are).
2. Arduino IDE: **File → Open** → `ewsv2\hardware\ewsv2_hwcheck\ewsv2_hwcheck.ino`.
3. **Tools → Board → Arduino AVR Boards → Arduino Uno.**
4. **Tools → Port →** the COM number from step 1.2.
5. Click **Upload** (the → arrow). The bottom panel ends with `Done uploading.`; on some versions it shows
   `avrdude done. Thank you.` instead.
6. **Tools → Serial Monitor.** At the right of the Serial Monitor set **115200 baud**.

**Step A – board alone (nothing wired yet).** You should see this every 2 seconds, and the L LED blinks:

```
=== ewsv2 hardware check VERSION 2 (Arduino Uno) ===
Board OK: you can read this, and the small L LED blinks.
...
PROBE  value 0 (off: 0)  -> NOT DETECTED: check AO on A0, VCC on A2, GND on GND
TOUCH  now not touched, touches counted: 0
DHT11  NO ANSWER: check DATA on A1, VCC on 5V, GND on GND
```

That is correct: nothing is connected yet, and the program keeps running anyway. While nothing is wired,
the PROBE numbers sit around 490 and TOUCH may say TOUCHED: an empty pin reads random values. Both
settle once the parts are connected. **The board works.**

---

## 3. How to connect the wires

**Follow `wiring_guide.html`** (same folder): it has a drawing of the Uno with the 9 holes marked and one drawing per
part. Open it in the browser. In short:

- Both ends of your jumper wires are holes, and so are the Uno's edges. Snap 9 single pins off the black
  pin strip and push one into each Uno hole you use. The wire's hole then slides onto the pin.
- The sensor boards already have pins, so the wires slide straight on.
- Turn the Uno so the USB socket is on the left. Go by the names printed next to the holes and next to
  each sensor's pins, never by position.

| Wire (colour on the student's parts) | From (sensor pin) | To (Uno hole) |
|---|---|---|
| white (soldered) | Touch **GND** | **GND** – top edge, just left of 13 |
| black (soldered) | Touch **VCC** | **4** – top edge |
| yellow (soldered) | Touch **SIG** | **3** – top edge |
| brown (soldered) | DHT11 **VCC** | **5V** – bottom edge |
| red (soldered) | DHT11 **GND** – red is GND here | **GND** – bottom edge (first) |
| white (soldered) | DHT11 **DATA** | **A1** – bottom edge |
| grey | Probe board **GND** | **GND** – bottom edge (second) |
| purple | Probe board **VCC** | **A2** – bottom edge (the program switches the probe on only while reading) |
| blue | Probe board **AO** | **A0** – bottom edge (DO stays empty) |

The colours mean nothing electrically; they only help you keep track.

## 4. Test each part

Unplug the USB, wire **one part**, check the wires against the table, plug the USB back in and watch the
Serial Monitor (it reconnects by itself; if it doesn't, close and reopen it). Then add the next part.
Keep earlier parts connected.

### Step B – DHT11 (temperature and humidity of the air)

Expected within 4 seconds:

```
DHT11  28 C, 65 % humidity  -> OK
```

- Room temperature should be about right (25–35 C in Chennai); its red LED should be on.
- **Breathe on it for 3 seconds** from close by: humidity rises by 10–30 % in the next reports, then
  falls back. That proves it measures.
- `NO ANSWER` every time → S is not on A1, or + / − are swapped (unplug at once if it gets warm).
  One `checksum error` now and then is normal.

### Step C – touch sensor

```
TOUCH  now TOUCHED, touches counted: 3
```

- Touch the round pad with a finger: its small LED lights and `touches counted` goes up by 1 for each touch.
- Count rising by itself without any touch → the SIG wire is loose (the pin "floats").
- LED never lights → VCC is not on D4 or GND is loose.

### Step D – water probe

The probe's 4-pin side goes to the Uno, and the fork stays on the 2-pin side.

1. **Fork in air:** `PROBE value 1000+ ... connected, DRY (in air)`. The board's power LED flashes every 2 s.
2. **Fork in a glass of tap water** (only the prongs, about 2 cm deep): within 2 reports the value
   drops: `... connected, WET (in water)`. **Write the number down** (tap water: ________).
3. **Add a spoon of salt, stir, wait 10 s:** the value drops further (salty: ________). This is the
   "conductivity rises" event the demo will use: dissolved salt = more ions = more current.
4. **Pull the fork out of its 2 pins** while it is in water: the value jumps back to dry. This is the
   "sensor broke" case the system must detect.
5. Rinse and dry the fork.

`NOT DETECTED` → check AO→A0, VCC→A2 and GND→GND. If the value never changes from dry, check the
two fork wires on the 2-pin side.

---

## 5. Send me the result

Select the Serial Monitor text (click in it, **Ctrl+A**, **Ctrl+C**) after each step, or copy this table
and fill it in:

| Part | Works? | Values seen |
|---|---|---|
| Uno + upload | yes / no | |
| DHT11 | yes / no | ___ C, ___ % (after breathing: ___ %) |
| Touch | yes / no | touches counted after 5 touches: ___ |
| Probe | yes / no | air ___, tap water ___, salt water ___, fork unplugged ___ |

A part that fails is not a problem: the system is built so that every hardware part is optional
(section 7).

---

## 6. Problems

| What you see | Fix |
|---|---|
| No COM port in Arduino IDE | CH340 driver (step 1.2); another USB cable (some are charge-only); another USB socket |
| Upload: `avrdude: ser_open(): can't open device` | Wrong port, or the Serial Monitor/another program is using it: close it and upload again |
| Upload: `stk500_getsync(): not in sync` | Board must be **Arduino Uno**; correct port; nothing connected to pins 0 and 1; press the Uno's reset button just as the upload starts |
| Serial Monitor shows rubbish characters | Set **115200 baud** |
| Serial Monitor shows nothing | Press the Uno's reset button (next to the USB socket); close and reopen the Serial Monitor |
| Uno's ON LED goes off or the laptop says "USB device not recognised" when a part is connected | A short circuit: unplug at once and check 5V and GND against the table |

---

## 7. What happens with these parts in the project

Every part is optional. If one is missing or broken, the system says so and carries on:

| Part | Role in the project | If it is missing or broken |
|---|---|---|
| Uno | Runs the early-warning logic, fed the GECCO data over USB (the hardware check for the paper) | The same logic runs on the laptop (Python), identical results |
| Probe | Live demo: the "water" sensor (salt = event, fork pulled = sensor fault) | The laptop simulates the stream |
| Touch | Operator's "false alarm" button for the self-adjusting threshold | Keyboard or dashboard button |
| DHT11 | Station temperature in the live demo | Left out of the demo |
| RFID | – (not used) | – |
