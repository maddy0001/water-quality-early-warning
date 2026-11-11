/* ewsv2 S8 – a simulated Arduino Uno (ATmega328P, 16 MHz) for checking the real firmware without
   a board (Linux, simavr). The firmware's serial port is bridged to stdin/stdout, one line each way,
   so the laptop talks to it exactly as to the real Uno. Simulated parts: DHT11 on A1 (27 C, 61 %),
   touch sensor on D3 (pressed for 1 s every 5 s), probe board on A0 powered from A2.
   Build:  gcc -O2 -o uno_sim uno_sim.c -lsimavr -lelf        Run: ./uno_sim firmware.elf            */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <simavr/sim_avr.h>
#include <simavr/sim_elf.h>
#include <simavr/sim_irq.h>
#include <simavr/sim_cycle_timers.h>
#include <simavr/avr_uart.h>
#include <simavr/avr_ioport.h>
#include <simavr/avr_adc.h>

static avr_t* avr;
static avr_irq_t *uart_in, *dht_pin, *touch_pin, *adc0;
static int seq_level[100], seq_us[100], seq_n, seq_i, last_dir = 0;
static char inq[8192];
static int inq_len = 0, inq_pos = 0, newlines = 0, xoff = 0;

static void uart_out(struct avr_irq_t* irq, uint32_t v, void* p) {
  (void)irq; (void)p;
  putchar((int)v);
  if (v == '\n') { fflush(stdout); newlines++; }
}
static void on_xon(struct avr_irq_t* irq, uint32_t v, void* p) { (void)irq; (void)v; (void)p; xoff = 0; }
static void on_xoff(struct avr_irq_t* irq, uint32_t v, void* p) { (void)irq; (void)v; (void)p; xoff = 1; }

static avr_cycle_count_t feed(avr_t* a, avr_cycle_count_t when, void* p) {
  (void)a; (void)p;
  if (inq_pos < inq_len && !xoff) avr_raise_irq(uart_in, (uint8_t)inq[inq_pos++]);
  return inq_pos < inq_len ? when + 16 * 100 : 0;          /* one byte per 100 us */
}

static avr_cycle_count_t dht_step(avr_t* a, avr_cycle_count_t when, void* p) {
  (void)a; (void)when; (void)p;
  if (seq_i >= seq_n) return 0;
  avr_raise_irq(dht_pin, seq_level[seq_i]);
  int us = seq_us[seq_i++];
  if (seq_i < seq_n) avr_cycle_timer_register_usec(avr, us, dht_step, NULL);
  return 0;
}
static void push(int level, int us) { seq_level[seq_n] = level; seq_us[seq_n] = us; seq_n++; }
static void dirC(struct avr_irq_t* irq, uint32_t v, void* p) {
  (void)irq; (void)p;
  int dir = (v >> 1) & 1;                                     /* A1 = PC1 */
  if (last_dir == 1 && dir == 0) {                            /* the firmware released the line */
    uint8_t d[5] = {61, 0, 27, 0, 0};
    d[4] = d[0] + d[1] + d[2] + d[3];
    seq_n = seq_i = 0;
    push(1, 30); push(0, 80); push(1, 80);
    for (int i = 0; i < 40; i++) { push(0, 50); push(1, ((d[i / 8] >> (7 - i % 8)) & 1) ? 70 : 27); }
    push(0, 50); push(1, 10);
    avr_cycle_timer_register_usec(avr, 1, dht_step, NULL);
  }
  last_dir = dir;
}
static void a2(struct avr_irq_t* irq, uint32_t v, void* p) { (void)irq; (void)p; avr_raise_irq(adc0, v ? 2400 : 0); }
static avr_cycle_count_t touch_tick(avr_t* a, avr_cycle_count_t when, void* p) {
  static int state = 0;
  (void)a; (void)p;
  state = !state;
  avr_raise_irq(touch_pin, state);
  return when + (avr_cycle_count_t)(state ? 1000000 : 4000000) * 16;
}

int main(int argc, char** argv) {
  elf_firmware_t f;
  memset(&f, 0, sizeof(f));
  if (argc < 2) { fprintf(stderr, "usage: uno_sim firmware.elf\n"); return 1; }
  if (elf_read_firmware(argv[1], &f)) { fprintf(stderr, "cannot read %s\n", argv[1]); return 1; }
  avr = avr_make_mcu_by_name("atmega328p");
  avr_init(avr);
  f.frequency = 16000000;
  avr_load_firmware(avr, &f);
  avr->frequency = 16000000;
  avr->log = 0;
  uart_in = avr_io_getirq(avr, AVR_IOCTL_UART_GETIRQ('0'), UART_IRQ_INPUT);
  avr_irq_register_notify(avr_io_getirq(avr, AVR_IOCTL_UART_GETIRQ('0'), UART_IRQ_OUTPUT), uart_out, NULL);
  avr_irq_register_notify(avr_io_getirq(avr, AVR_IOCTL_UART_GETIRQ('0'), UART_IRQ_OUT_XON), on_xon, NULL);
  avr_irq_register_notify(avr_io_getirq(avr, AVR_IOCTL_UART_GETIRQ('0'), UART_IRQ_OUT_XOFF), on_xoff, NULL);
  dht_pin = avr_io_getirq(avr, AVR_IOCTL_IOPORT_GETIRQ('C'), 1);
  touch_pin = avr_io_getirq(avr, AVR_IOCTL_IOPORT_GETIRQ('D'), 3);
  adc0 = avr_io_getirq(avr, AVR_IOCTL_ADC_GETIRQ, ADC_IRQ_ADC0);
  avr_irq_register_notify(avr_io_getirq(avr, AVR_IOCTL_IOPORT_GETIRQ('C'), IOPORT_IRQ_DIRECTION_ALL), dirC, NULL);
  avr_irq_register_notify(avr_io_getirq(avr, AVR_IOCTL_IOPORT_GETIRQ('C'), 2), a2, NULL);
  avr_raise_irq(dht_pin, 1);
  avr_raise_irq(touch_pin, 0);
  avr_cycle_timer_register_usec(avr, 3000000, touch_tick, NULL);

  int wanted = 1;                 /* the READY line after reset */
  avr_cycle_count_t waited_from = 0;
  char buf[4096];
  for (;;) {
    int state = avr_run(avr);
    if (state == cpu_Done || state == cpu_Crashed) { fprintf(stderr, "firmware stopped (%d)\n", state); return 2; }
    /* the answer to the last line is complete (or 10 s simulated without one): read the next line */
    if (newlines >= wanted || avr->cycle - waited_from > (avr_cycle_count_t)16000000 * 10) {
      if (!fgets(buf, sizeof(buf), stdin)) return 0;
      size_t n = strlen(buf);
      if (n == 0) continue;
      memcpy(inq, buf, n);
      inq_len = (int)n;
      inq_pos = 0;
      wanted = newlines + 1;
      waited_from = avr->cycle;
      avr_cycle_timer_register(avr, 16 * 100, feed, NULL);
    }
  }
}
