// ewsv2 hardware check – Arduino Uno (or clone). No extra libraries needed.
//
// Wiring (step by step with pictures in the wiring guide):
//   Touch sensor TTP223 (top edge of the Uno):   I/O -> D3,   VCC -> D4,  GND -> GND (next to pin 13)
//   DHT11 module (bottom edge):                  DATA -> A1,  VCC -> 5V, GND -> GND
//   Probe board LM393 (bottom edge):             AO -> A0,    VCC -> A2,  GND -> GND   (DO not used)
// Every part is optional: unplugged parts are reported, nothing stops.
//
// Open Tools -> Serial Monitor, set 115200 baud. A report appears every 2 seconds.
// The probe board is powered from A2 only while it is read, so the fork corrodes less in water.

#include <Arduino.h>

const uint8_t PIN_DHT = A1;            // analog pins work as digital pins too
const uint8_t PIN_TOUCH = 3;
const uint8_t PIN_TOUCH_POWER = 4;      // the touch sensor needs < 5 mA: a pin can power it
const uint8_t PIN_PROBE_POWER = A2;    // powers the probe board only while it is read
const uint8_t PIN_PROBE = A0;

unsigned long lastReport = 0;
unsigned int touches = 0;
bool lastTouch = false;
unsigned long lastChange = 0;
unsigned int reportNo = 0;

// ---------------------------------------------------------------- probe
struct ProbeReading { int on; int off; };

ProbeReading readProbe() {
  ProbeReading r;
  digitalWrite(PIN_PROBE_POWER, LOW);
  delay(5);
  long s = 0;
  for (int i = 0; i < 8; i++) s += analogRead(PIN_PROBE);
  r.off = s / 8;                       // board switched off: should read near 0
  digitalWrite(PIN_PROBE_POWER, HIGH);
  delay(50);                           // let the board settle
  s = 0;
  for (int i = 0; i < 16; i++) s += analogRead(PIN_PROBE);
  r.on = s / 16;                       // board switched on: 1023 = dry ... lower = wetter / saltier
  digitalWrite(PIN_PROBE_POWER, LOW);
  return r;
}

// ---------------------------------------------------------------- DHT11 (own reader, no library)
// returns 0 = ok, 1 = no answer (wiring), 2 = answer broken off, 3 = checksum wrong
uint16_t pulse(volatile uint8_t* port, uint8_t mask, bool level) {
  uint16_t n = 0;
  while (((*port & mask) != 0) == level) {
    if (++n >= 20000) return 0xFFFF;   // a few ms: no change, give up
  }
  return n;
}

uint8_t readDHT11(int& temperature, int& humidity) {
  static uint16_t cycles[80];
  uint8_t data[5] = {0, 0, 0, 0, 0};
  volatile uint8_t* port = portInputRegister(digitalPinToPort(PIN_DHT));
  uint8_t mask = digitalPinToBitMask(PIN_DHT);

  pinMode(PIN_DHT, INPUT_PULLUP);
  delay(2);
  pinMode(PIN_DHT, OUTPUT);            // start signal: pull the line low for 20 ms
  digitalWrite(PIN_DHT, LOW);
  delay(20);
  noInterrupts();
  pinMode(PIN_DHT, INPUT_PULLUP);        // release the line; the sensor answers after 20-40 us
  uint8_t status = 0;
  // wait for the answer (line pulled low), then its 80 us low and 80 us high
  if (pulse(port, mask, true) == 0xFFFF || pulse(port, mask, false) == 0xFFFF ||
      pulse(port, mask, true) == 0xFFFF) {
    status = 1;
  } else {
    for (uint8_t i = 0; i < 80; i += 2) {
      cycles[i] = pulse(port, mask, false);       // 50 us low before every bit
      cycles[i + 1] = pulse(port, mask, true);    // then high: ~27 us = 0, ~70 us = 1
    }
  }
  interrupts();
  if (status) return status;
  for (uint8_t i = 0; i < 40; i++) {
    uint16_t low = cycles[2 * i], high = cycles[2 * i + 1];
    if (low == 0xFFFF || high == 0xFFFF) return 2;
    data[i / 8] <<= 1;
    if (high > low) data[i / 8] |= 1;
  }
  if ((uint8_t)(data[0] + data[1] + data[2] + data[3]) != data[4]) return 3;
  humidity = data[0];
  temperature = data[2];
  return 0;
}

// ---------------------------------------------------------------- touch
void pollTouch() {
  bool now = digitalRead(PIN_TOUCH) == HIGH;
  if (now != lastTouch && millis() - lastChange > 30) {   // 30 ms debounce
    lastChange = millis();
    if (now) touches++;
    lastTouch = now;
  }
}

void setup() {
  Serial.begin(115200);
  pinMode(LED_BUILTIN, OUTPUT);
  pinMode(PIN_PROBE_POWER, OUTPUT);
  digitalWrite(PIN_PROBE_POWER, LOW);
  pinMode(PIN_TOUCH_POWER, OUTPUT);
  digitalWrite(PIN_TOUCH_POWER, HIGH);
  pinMode(PIN_TOUCH, INPUT);
  pinMode(PIN_DHT, INPUT_PULLUP);
  delay(1500);                         // DHT11 needs ~1 s after power-up
  Serial.println(F(""));
  Serial.println(F("=== ewsv2 hardware check VERSION 2 (Arduino Uno) ==="));
  Serial.println(F("Board OK: you can read this, and the small L LED blinks."));
  Serial.println(F("Pins: touch SIG=D3 VCC=D4 | DHT11 DATA=A1 VCC=5V | probe AO=A0 VCC=A2"));
  Serial.println(F("Report every 2 s. Touch the pad, dip the probe, breathe on the DHT11."));
}

void loop() {
  pollTouch();
  if (millis() - lastReport < 2000) return;
  lastReport = millis();
  reportNo++;
  digitalWrite(LED_BUILTIN, reportNo & 1);

  ProbeReading p = readProbe();
  int t = 0, h = 0;
  uint8_t d = readDHT11(t, h);

  Serial.println(F("------------------------------------------------------------"));
  Serial.print(F("report ")); Serial.print(reportNo);
  Serial.print(F("   uptime ")); Serial.print(millis() / 1000); Serial.println(F(" s"));

  Serial.print(F("PROBE  value ")); Serial.print(p.on);
  Serial.print(F(" (off: ")); Serial.print(p.off); Serial.print(F(")  -> "));
  if (p.on - p.off < 50) Serial.println(F("NOT DETECTED: check AO on A0, VCC on A2, GND on GND"));
  else if (p.on > 900) Serial.println(F("connected, DRY (in air)"));
  else if (p.on > 600) Serial.println(F("connected, DAMP"));
  else Serial.println(F("connected, WET (in water)"));

  Serial.print(F("TOUCH  now ")); Serial.print(lastTouch ? F("TOUCHED") : F("not touched"));
  Serial.print(F(", touches counted: ")); Serial.println(touches);

  Serial.print(F("DHT11  "));
  if (d == 0) {
    Serial.print(t); Serial.print(F(" C, ")); Serial.print(h); Serial.print(F(" % humidity  -> "));
    if (t >= 0 && t <= 50 && h >= 5 && h <= 100) Serial.println(F("OK"));
    else Serial.println(F("odd values: check the DATA pin"));
  } else if (d == 1) {
    Serial.println(F("NO ANSWER: check DATA on A1, VCC on 5V, GND on GND"));
  } else if (d == 2) {
    Serial.println(F("answer broken off: loose wire? (one error now and then is normal)"));
  } else {
    Serial.println(F("checksum error (one now and then is normal; every time = loose wire)"));
  }
}
