// ewsv2 S8 – line protocol between the laptop and the edge code (Uno, PC build or simulated Uno).
//
//   D,<c1>,...,<ck>      one minute of counts (empty field = missing)
//                        -> R,<t>,<logit hex>,<theta hex>,<alarm>,<water>,<sensor>,<mask>,<top>,<us>,<ask>,<ep_start>,<last_on>
//   V,<n>                operator verdict after an R line with ask=1   -> OK,V
//   C,<int8>,<adaptive>  configure and restart                         -> READY,...
//   X                    restart                                       -> READY,...
//   I                    information                                   -> INFO,...
//   H                    which sensors answer                          -> HW,touch=..,dht=..,probe=..,rfid=0
//   S                    read the sensors now                          -> S,<probe on>,<probe off>,<dht status>,<temp>,<hum>,<touches>,<touch now>
// Floats are sent as the 8 hex digits of their IEEE bits, so the laptop gets them exactly.
#pragma once
#include <stdio.h>
#include "edge_core.h"

#if defined(__AVR__)                     // keep the format strings in flash on the Uno (RAM is 2 KB)
#define EFMT(s) PSTR(s)
#define esnprintf snprintf_P
#else
#define EFMT(s) s
#define esnprintf snprintf
#endif

unsigned long edge_now_us();   // provided by the platform
long edge_free_ram();          // bytes free (-1 if unknown)
int edge_hw_line(char* out, int n);       // "HW,..." – sensors that answer
int edge_sensor_line(char* out, int n);   // "S,..." – one reading of every sensor

struct EdgeStats {
  unsigned long max_us = 0;
  unsigned long sum_us = 0;
  long n = 0;
};

static inline unsigned long edge_bits(float v) {
  uint32_t b;
  memcpy(&b, &v, 4);
  return (unsigned long)b;
}

inline int edge_ready(EdgeCore& c, char* out, int n) {
  return esnprintf(out, n, EFMT("READY,ewsv2-edge,%s,%d,%d,%d,%d,%ld,%u"), EDGE_DATASET, EDGE_K, EDGE_PARAMS, (int)c.int8,
                  (int)c.adaptive, edge_free_ram(), (unsigned)sizeof(EdgeCore));
}

inline int edge_handle(EdgeCore& c, EdgeStats& st, char* line, char* out, int n) {
  int len = (int)strlen(line);
  while (len && (line[len - 1] == '\r' || line[len - 1] == '\n' || line[len - 1] == ' ')) line[--len] = 0;
  if (len == 0) return 0;
  if (line[0] == 'D' && line[1] == ',') {
    char* p = line + 1;
    int16_t raw[EDGE_K];
    for (uint8_t s = 0; s < EDGE_K; ++s) {
      raw[s] = EDGE_MISSING;
      if (*p == ',') {
        p++;
        if (*p != ',' && *p != 0) {
          char* e;
          long v = strtol(p, &e, 10);
          if (e != p) raw[s] = (int16_t)(v < -32000 ? -32000 : (v > 32000 ? 32000 : v));
          p = e;
          while (*p && *p != ',') p++;
        }
      }
    }
    unsigned long t0 = edge_now_us();
    EdgeOut o = c.push(raw);
    unsigned long us = edge_now_us() - t0;
    if (us > st.max_us) st.max_us = us;
    st.sum_us += us;
    st.n++;
    return esnprintf(out, n, EFMT("R,%ld,%08lx,%08lx,%d,%d,%d,%u,%d,%lu,%d,%ld,%ld"), (long)o.t, edge_bits(o.logit),
                    edge_bits(o.theta), o.alarm, o.water, o.sensor, (unsigned)o.mask, o.top, us, o.ask,
                    (long)o.ep_start, (long)o.last_on);
  }
  if (line[0] == 'V' && line[1] == ',') {
    c.verdict(atoi(line + 2));
    return esnprintf(out, n, EFMT("OK,V,%08lx"), edge_bits(c.theta()));
  }
  if (line[0] == 'C' && line[1] == ',') {
    c.int8 = line[2] != '0';
    c.adaptive = !(len >= 5 && line[4] == '0');
    c.reset();
    st = EdgeStats();
    return edge_ready(c, out, n);
  }
  if (line[0] == 'X') {
    c.reset();
    st = EdgeStats();
    return edge_ready(c, out, n);
  }
  if (line[0] == 'I') {
    return esnprintf(out, n, EFMT("INFO,minutes=%ld,sum_us=%lu,max_us=%lu,free_ram=%ld,state_bytes=%u"), (long)c.minutes(),
                    st.sum_us, st.max_us, edge_free_ram(), (unsigned)sizeof(EdgeCore));
  }
  if (line[0] == 'H') return edge_hw_line(out, n);
  if (line[0] == 'S') return edge_sensor_line(out, n);
  return esnprintf(out, n, EFMT("ERR,unknown"));
}
