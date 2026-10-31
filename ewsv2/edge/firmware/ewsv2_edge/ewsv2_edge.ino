// ewsv2 S8 – Arduino Uno sketch: the edge early-warning logic + the optional live sensors.
//
// Open this folder in Arduino IDE 2, Tools -> Board -> Arduino Uno, your COM port, Upload.
// Close the Serial Monitor, then on the laptop:
//     python -m ewsv2 board                       (finds the Uno, says which sensors answer)
//     python -m ewsv2 edge --config s8_edge       (replays GECCO to the Uno, checks every answer)
// Protocol: edge_protocol.h. Sensors (all optional): edge_sensors.h.
#include "edge_protocol.h"
#include "edge_sensors.h"

static EdgeCore core;
static EdgeStats stats;
static char line[100];
static uint8_t len = 0;
static char out[110];

unsigned long edge_now_us() { return micros(); }

// Free RAM = bytes between the heap and the deepest the stack has ever reached. At start the free
// area is painted with 0xA5; bytes the stack has used are no longer 0xA5.
extern int __heap_start, *__brkval;
static uint8_t* heap_end() { return (uint8_t*)(__brkval == 0 ? (int)&__heap_start : (int)__brkval); }

static void paint_stack() {
  uint8_t here;
  for (uint8_t* p = heap_end(); p < &here - 16; ++p) *p = 0xA5;
}

long edge_free_ram() {
  uint8_t* p = heap_end();
  long n = 0;
  while (*p == 0xA5) { ++p; ++n; }
  return n;
}

void setup() {
  paint_stack();
  Serial.begin(115200);
  sensors_begin();
  core.reset();
  edge_ready(core, out, sizeof(out));
  Serial.println(out);
}

void loop() {
  sensors_poll();
  while (Serial.available()) {
    char ch = (char)Serial.read();
    if (ch == '\n') {
      line[len] = 0;
      if (edge_handle(core, stats, line, out, sizeof(out)) > 0) Serial.println(out);
      len = 0;
    } else if (len < sizeof(line) - 1) {
      line[len++] = ch;
    }
  }
}
