// ewsv2 S8 – the same edge code built for a PC (no board needed): protocol lines on stdin/stdout.
//   g++ -O2 -std=c++17 -ffp-contract=off -o ewsv2_edge_desktop desktop_main.cpp
#include <chrono>
#include <cstdio>

#include "../ewsv2_edge/edge_protocol.h"

unsigned long edge_now_us() {
  using namespace std::chrono;
  return (unsigned long)duration_cast<microseconds>(steady_clock::now().time_since_epoch()).count();
}
long edge_free_ram() { return -1; }
int edge_hw_line(char* out, int n) { return snprintf(out, n, "HW,touch=0,dht=0,probe=0,rfid=0"); }
int edge_sensor_line(char* out, int n) { return snprintf(out, n, "S,-1,-1,1,0,0,0,0"); }

int main() {
  static EdgeCore core;
  static EdgeStats stats;
  static char line[4096];
  static char out[512];
  edge_ready(core, out, sizeof(out));
  std::printf("%s\n", out);
  std::fflush(stdout);
  while (std::fgets(line, sizeof(line), stdin)) {
    if (edge_handle(core, stats, line, out, sizeof(out)) > 0) {
      std::printf("%s\n", out);
      std::fflush(stdout);
    }
  }
  return 0;
}
