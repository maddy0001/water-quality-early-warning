// ewsv2 S8 – the edge early-warning algorithm in C++, sized for an Arduino Uno (2 KB RAM).
//
// Same algorithm as ewsv2/edge/reference.py, step by step: integer counts (16 per normal unit),
// exact integer sums, 32-bit floats in a fixed order. The Uno, a PC build of this file and the
// Python reference give bit-identical answers. The model and limits live in flash (edge_model.h).
// Uses only `float` (on the Uno `double` is the same 32-bit type; on a PC it is not).
#pragma once
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(__AVR__)
#include <avr/pgmspace.h>
#define EDGE_FLASH PROGMEM
#define RD_F(a, i) pgm_read_float(&(a)[i])
#define RD_I8(a, i) ((int8_t)pgm_read_byte(&(a)[i]))
#define RD_I16(a, i) ((int16_t)pgm_read_word(&(a)[i]))
#define RD_I32(a, i) ((int32_t)pgm_read_dword(&(a)[i]))
#else
#define EDGE_FLASH
#define RD_F(a, i) ((a)[i])
#define RD_I8(a, i) ((a)[i])
#define RD_I16(a, i) ((a)[i])
#define RD_I32(a, i) ((a)[i])
#endif

#ifdef EDGE_MODEL_FILE              // PC builds for another station pass -DEDGE_MODEL_FILE="<path>"
#include EDGE_MODEL_FILE
#else
#include "edge_model.h"
#endif

#define EDGE_MISSING (-32768)
#define EDGE_UNIT 16                // counts per normal unit
#define EDGE_HOLD 60
#define EDGE_CARRY 5
#define EDGE_GAP 5

struct EdgeOut {
  int32_t t;
  float logit, theta;
  uint8_t has, alarm, water, sensor, top, ask;
  uint16_t mask;
  int32_t ep_start, last_on;
};

class EdgeCore {
 public:
  bool int8 = true;
  bool adaptive = true;

  void reset() {
    t_ = 0;
    theta_ = EDGE_THETA0;
    ep_start_ = -1;
    last_on_ = -1000000L;
    pending_ = false;
    nblk_ = 0;
    bpos_ = 0;
    for (uint8_t s = 0; s < EDGE_K; ++s) {
      for (uint8_t i = 0; i < 16; ++i) ring_[i][s] = EDGE_MISSING;
      for (uint8_t b = 0; b < 4; ++b) { bsum_[b][s] = 0; bcnt_[b][s] = 0; }
      csum_[s] = 0;
      ccnt_[s] = 0;
      last_[s] = EDGE_MISSING;
      since_[s] = 255;
      run_[s] = 0;
      drop_[s] = 0;
      lastflag_[s] = -1000000L;
      jon_[s] = 0;
      jstart_[s][0] = jstart_[s][1] = 0;
    }
  }

  EdgeCore() { reset(); }

  // One minute of counts (EDGE_MISSING = no reading).
  EdgeOut push(const int16_t* raw) {
    if (pending_) verdict(0);                 // no verdict came back: count no false alarm
    EdgeOut o;
    memset(&o, 0, sizeof(o));
    o.t = t_;
    if (t_ > 0 && t_ % 15 == 0) {            // a quarter-hour is complete: keep its sums
      for (uint8_t s = 0; s < EDGE_K; ++s) { bsum_[bpos_][s] = csum_[s]; bcnt_[bpos_][s] = ccnt_[s]; csum_[s] = 0; ccnt_[s] = 0; }
      bpos_ = (bpos_ + 1) & 3;
      if (nblk_ < 4) nblk_++;
    }
    int16_t x[EDGE_K];
    bool has = false;
    for (uint8_t s = 0; s < EDGE_K; ++s) {   // carry forward up to 5 minutes
      if (raw[s] != EDGE_MISSING) { x[s] = raw[s]; last_[s] = raw[s]; since_[s] = 0; }
      else if (since_[s] < EDGE_CARRY) { since_[s]++; x[s] = last_[s]; }
      else { if (since_[s] < 255) since_[s]++; x[s] = EDGE_MISSING; }
      if (x[s] != EDGE_MISSING) has = true;
    }
    o.has = has;

    // ---- features ------------------------------------------------------------------------
    float F[EDGE_IN];
    float m[EDGE_K];
    float maxm = -1.0f, z1 = -1.0f, z2 = -1.0f;
    uint8_t breadth = 0, top = 0, lead = 0;
    for (uint8_t s = 0; s < EDGE_K; ++s) {
      int32_t sum = 0, n = 0;
      int16_t mx = -32768, mn = 32767;
      for (uint8_t i = 1; i <= 15 && (int32_t)i <= t_; ++i) {
        int16_t v = ring_[(t_ - i) & 15][s];
        if (v == EDGE_MISSING) continue;
        sum += v; n++;
        if (v > mx) mx = v;
        if (v < mn) mn = v;
      }
      int32_t s60 = 0, n60 = 0;
      for (uint8_t b = 0; b < nblk_; ++b) { s60 += bsum_[b][s]; n60 += bcnt_[b][s]; }
      float z = 0.0f, d = 0.0f, r = 0.0f;
      if (n >= 5 && x[s] != EDGE_MISSING) z = (float)(n * (int32_t)x[s] - sum) / (float)(n * EDGE_UNIT);
      if (n60 >= 20 && x[s] != EDGE_MISSING) d = (float)(n60 * (int32_t)x[s] - s60) / (float)(n60 * EDGE_UNIT);
      if (n >= 5) r = (float)((int32_t)mx - mn) / 16.0f;
      z = clip(z, -50.0f, 50.0f);
      d = clip(d, -50.0f, 50.0f);
      r = clip(r, 0.0f, 50.0f);
      F[s] = z;
      F[EDGE_K + s] = d;
      F[2 * EDGE_K + s] = r;
      float az = fabsf(z);
      m[s] = az / RD_F(EDGE_MOVE_LIMIT, s);
      if (m[s] >= 1.0f) breadth++;
      if (m[s] > maxm) { maxm = m[s]; top = s; }
      if (az > z1) { z2 = z1; z1 = az; lead = s; }
      else if (az > z2) z2 = az;
    }
    F[3 * EDGE_K] = (float)breadth;
    F[3 * EDGE_K + 1] = maxm;
    o.top = top;

    // ---- detector and threshold ---------------------------------------------------------------
    float lp = forward(F);                   // F is not needed afterwards
    o.logit = lp;
    o.theta = NAN;
    bool alarm = false;
    if (has) {
      o.theta = theta_;
      alarm = lp >= theta_;
      if (alarm) {
        if (t_ - last_on_ > EDGE_GAP) ep_start_ = t_;
        last_on_ = t_;
      }
      if (adaptive) {
        theta_ = clampt(theta_ - EDGE_DECAY);
        if (!alarm && ep_start_ >= 0 && t_ - last_on_ > EDGE_GAP) {
          o.ask = 1;                         // episode over: ask the operator for the verdict
          o.ep_start = ep_start_;
          o.last_on = last_on_;
          pending_ = true;
          ep_start_ = -1;
        }
      } else if (!alarm && ep_start_ >= 0 && t_ - last_on_ > EDGE_GAP) {
        ep_start_ = -1;
      }
    }
    o.alarm = alarm;

    // ---- lone jumps -------------------------------------------------------------------------
    int32_t step[EDGE_K];
    for (uint8_t s = 0; s < EDGE_K; ++s) {
      int16_t prev = t_ >= 1 ? ring_[(t_ - 1) & 15][s] : (int16_t)EDGE_MISSING;
      step[s] = (x[s] == EDGE_MISSING || prev == EDGE_MISSING) ? 0 : (int32_t)x[s] - prev;
      for (uint8_t g = 0; g < 2; ++g) {        // open suspicions: expire, or end on a further step
        if (!(jon_[s] & (1 << g))) continue;
        int32_t sign = g == 0 ? 1 : -1;
        if (t_ > jstart_[s][g] + EDGE_HOLD || step[s] * sign >= 2 * EDGE_UNIT) jon_[s] &= ~(1 << g);
      }
    }
    if (EDGE_K < 2) z2 = 0.0f;
    float jl = (float)labs(step[lead]) / 16.0f;
    bool lone = alarm && jl >= 6.0f && z2 < 3.0f && jl / (z1 + 1.0f) >= 0.7f;
    if (lone) {
      uint8_t g = step[lead] > 0 ? 0 : 1;
      jon_[lead] |= (1 << g);
      jstart_[lead][g] = t_;
    }

    // ---- fault checks and triage ------------------------------------------------------------
    uint16_t mask = 0;
    bool any_suspect = false;
    float clean = 0.0f;
    for (uint8_t s = 0; s < EDGE_K; ++s) {
      int16_t v = x[s];
      int16_t prev = t_ >= 1 ? ring_[(t_ - 1) & 15][s] : (int16_t)EDGE_MISSING;
      if (v != EDGE_MISSING && prev != EDGE_MISSING && v == prev) { if (run_[s] < 65535u) run_[s]++; }
      else run_[s] = v == EDGE_MISSING ? 0 : 1;
      bool flag = run_[s] >= (uint16_t)RD_I16(EDGE_STUCK, s);
      int16_t low = RD_I16(EDGE_LOW, s);
      bool below = v != EDGE_MISSING && v <= low;
      if (!below) {
        drop_[s] = 0;
      } else if (!drop_[s] && t_ > 0 && prev != EDGE_MISSING && prev > low) {
        int32_t need = labs(RD_I32(EDGE_BASE, s) + (int32_t)prev);
        if (need < 2 * 10 * EDGE_UNIT) need = 2 * 10 * EDGE_UNIT;
        if (2 * ((int32_t)prev - v) >= need) drop_[s] = 1;
      }
      if (drop_[s]) flag = true;
      int32_t med2;
      if (median2(s, med2) && prev != EDGE_MISSING && v != EDGE_MISSING) {
        if ((float)labs(2 * (int32_t)prev - med2) > RD_F(EDGE_SPIKE2, s) && labs(2 * (int32_t)v - med2) < 2 * 3 * EDGE_UNIT)
          flag = true;
      }
      if (!has) flag = false;
      if (flag) { mask |= (uint16_t)(1u << s); lastflag_[s] = t_; }
      bool suspect = (t_ - lastflag_[s] <= EDGE_HOLD) || jon_[s] != 0;
      if (suspect) any_suspect = true;
      else if (m[s] > clean) clean = m[s];
    }
    if (lone) mask |= (uint16_t)(1u << lead);
    o.mask = mask;
    o.sensor = mask != 0;
    o.water = alarm && !lone && !(any_suspect && clean < 1.0f);

    // ---- history -------------------------------------------------------------------------------
    for (uint8_t s = 0; s < EDGE_K; ++s) {
      ring_[t_ & 15][s] = x[s];
      if (x[s] != EDGE_MISSING) { csum_[s] += x[s]; ccnt_[s]++; }
    }
    t_++;
    return o;
  }

  // The operator's verdict for the episode that just ended: number of false alarms in it.
  void verdict(int n_false) {
    if (adaptive && n_false > 0) theta_ = clampt(theta_ + 0.5f * (float)n_false);
    pending_ = false;
  }

  int32_t minutes() const { return t_; }
  float theta() const { return theta_; }

 private:
  int16_t ring_[16][EDGE_K];        // the last 16 minutes (carried counts)
  int32_t bsum_[4][EDGE_K];         // the last 4 complete quarter-hours
  uint8_t bcnt_[4][EDGE_K];
  int32_t csum_[EDGE_K];            // the current quarter-hour
  uint8_t ccnt_[EDGE_K];
  int16_t last_[EDGE_K];
  uint8_t since_[EDGE_K];
  uint16_t run_[EDGE_K];
  uint8_t drop_[EDGE_K];
  int32_t lastflag_[EDGE_K];
  uint8_t jon_[EDGE_K];             // lone-jump suspicion, one bit per direction
  int32_t jstart_[EDGE_K][2];
  int32_t t_, ep_start_, last_on_;
  float theta_;
  bool pending_;
  uint8_t nblk_, bpos_;

  static float clip(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }
  static float clampt(float v) { return v < -10.0f ? -10.0f : (v > 15.0f ? 15.0f : v); }

  // 2 x median of the readings 16 .. 2 minutes ago (needs >= 5), exact in integers
  bool median2(uint8_t s, int32_t& out) const {
    int16_t w[15];
    uint8_t n = 0;
    for (uint8_t i = 2; i <= 16 && (int32_t)i <= t_; ++i) {
      int16_t v = ring_[(t_ - i) & 15][s];
      if (v != EDGE_MISSING) w[n++] = v;
    }
    if (n < 5) return false;
    for (uint8_t a = 1; a < n; ++a) {          // insertion sort of at most 15 values
      int16_t key = w[a];
      int8_t b = a - 1;
      while (b >= 0 && w[b] > key) { w[b + 1] = w[b]; b--; }
      w[b + 1] = key;
    }
    out = (n & 1) ? 2 * (int32_t)w[n / 2] : (int32_t)w[n / 2 - 1] + w[n / 2];
    return true;
  }

  float weight(const float* W, const int8_t* Q, float S, uint16_t i) const {
    return int8 ? (float)RD_I8(Q, i) * S : RD_F(W, i);
  }

  // float32 network; returns the log-odds. Sums in input order (as the reference does).
  float forward(float* h0) const {   // standardises the features in place (saves 116 bytes of stack)
    float h1[EDGE_H1], h2[EDGE_H2];
    for (uint8_t i = 0; i < EDGE_IN; ++i) h0[i] = (h0[i] - RD_F(EDGE_MU, i)) / RD_F(EDGE_SD, i);
    for (uint8_t j = 0; j < EDGE_H1; ++j) {
      float a = 0.0f;
      for (uint8_t i = 0; i < EDGE_IN; ++i) a = a + h0[i] * weight(EDGE_W1, EDGE_W1Q, EDGE_W1S, (uint16_t)i * EDGE_H1 + j);
      a = a + RD_F(EDGE_B1, j);
      h1[j] = a > 0.0f ? a : 0.0f;
    }
    for (uint8_t j = 0; j < EDGE_H2; ++j) {
      float a = 0.0f;
      for (uint8_t i = 0; i < EDGE_H1; ++i) a = a + h1[i] * weight(EDGE_W2, EDGE_W2Q, EDGE_W2S, (uint16_t)i * EDGE_H2 + j);
      a = a + RD_F(EDGE_B2, j);
      h2[j] = a > 0.0f ? a : 0.0f;
    }
    float a = 0.0f;
    for (uint8_t i = 0; i < EDGE_H2; ++i) a = a + h2[i] * weight(EDGE_W3, EDGE_W3Q, EDGE_W3S, i);
    return a + RD_F(EDGE_B3, 0);
  }
};
