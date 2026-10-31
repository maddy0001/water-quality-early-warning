// ewsv2 S8 – the Uno's own sensors, all optional. Wiring (see ewsv2/hardware/wiring_guide.html):
//   touch TTP223: SIG -> D3, VCC -> D4            DHT11: DATA -> A1, VCC -> 5V, GND -> GND
//   probe board (LM393): AO -> A0, VCC -> A2, GND -> GND (powered only while it is read)
// A part that is not connected is reported as 0 in the HW line; nothing waits for it.
#pragma once
#include <Arduino.h>

const uint8_t PIN_TOUCH = 3, PIN_TOUCH_POWER = 4, PIN_DHT = A1, PIN_PROBE_POWER = A2, PIN_PROBE = A0;

static unsigned int touches = 0;
static bool touch_now = false, touch_seen_low = false;
static unsigned long touch_change = 0;

inline void sensors_begin() {
  pinMode(PIN_TOUCH_POWER, OUTPUT);
  digitalWrite(PIN_TOUCH_POWER, HIGH);
  pinMode(PIN_TOUCH, INPUT_PULLUP);       // an unconnected pin then reads HIGH, a connected sensor drives it LOW
  pinMode(PIN_PROBE_POWER, OUTPUT);
  digitalWrite(PIN_PROBE_POWER, LOW);
  pinMode(PIN_DHT, INPUT_PULLUP);
}

inline void sensors_poll() {             // call often: counts touches
  bool now = digitalRead(PIN_TOUCH) == HIGH;
  if (!now) touch_seen_low = true;
  if (now != touch_now && millis() - touch_change > 30) {
    touch_change = millis();
    if (now && touch_seen_low) touches++;
    touch_now = now;
  }
}

inline void probe_read(int& on, int& off) {
  digitalWrite(PIN_PROBE_POWER, LOW);
  delay(5);
  long s = 0;
  for (uint8_t i = 0; i < 8; i++) s += analogRead(PIN_PROBE);
  off = s / 8;
  digitalWrite(PIN_PROBE_POWER, HIGH);
  delay(50);
  s = 0;
  for (uint8_t i = 0; i < 16; i++) s += analogRead(PIN_PROBE);
  on = s / 16;
  digitalWrite(PIN_PROBE_POWER, LOW);
}

static uint16_t dht_pulse(volatile uint8_t* port, uint8_t mask, bool level) {
  uint16_t n = 0;
  while (((*port & mask) != 0) == level) {
    if (++n >= 20000) return 0xFFFF;
  }
  return n;
}

// 0 = ok, 1 = no answer, 2 = broken off, 3 = checksum
inline uint8_t dht_read(int& temperature, int& humidity) {
  uint8_t data[5] = {0, 0, 0, 0, 0};
  volatile uint8_t* port = portInputRegister(digitalPinToPort(PIN_DHT));
  uint8_t mask = digitalPinToBitMask(PIN_DHT);
  pinMode(PIN_DHT, INPUT_PULLUP);
  delay(2);
  pinMode(PIN_DHT, OUTPUT);
  digitalWrite(PIN_DHT, LOW);
  delay(20);
  noInterrupts();
  pinMode(PIN_DHT, INPUT_PULLUP);
  uint8_t status = 0;
  if (dht_pulse(port, mask, true) == 0xFFFF || dht_pulse(port, mask, false) == 0xFFFF ||
      dht_pulse(port, mask, true) == 0xFFFF) {
    status = 1;
  } else {
    for (uint8_t i = 0; i < 40 && !status; i++) {
      uint16_t low = dht_pulse(port, mask, false);
      uint16_t high = dht_pulse(port, mask, true);
      if (low == 0xFFFF || high == 0xFFFF) status = 2;
      data[i / 8] <<= 1;
      if (high > low) data[i / 8] |= 1;
    }
  }
  interrupts();
  if (status) return status;
  if ((uint8_t)(data[0] + data[1] + data[2] + data[3]) != data[4]) return 3;
  humidity = data[0];
  temperature = data[2];
  return 0;
}

inline int edge_hw_line(char* out, int n) {
  int on, off, t = 0, h = 0;
  probe_read(on, off);
  uint8_t d = dht_read(t, h);
  if (d) { delay(1100); d = dht_read(t, h); }      // one retry: the DHT11 needs 1 s between reads
  return esnprintf(out, n, EFMT("HW,touch=%d,dht=%d,probe=%d,rfid=0"), touch_seen_low ? 1 : 0, d == 0 ? 1 : 0,
                  (on - off) >= 50 ? 1 : 0);
}

inline int edge_sensor_line(char* out, int n) {
  int on, off, t = 0, h = 0;
  probe_read(on, off);
  uint8_t d = dht_read(t, h);
  return esnprintf(out, n, EFMT("S,%d,%d,%d,%d,%d,%u,%d"), on, off, d, t, h, touches, touch_now ? 1 : 0);
}
