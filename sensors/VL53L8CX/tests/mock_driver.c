/* SPDX-License-Identifier: BSD-3-Clause */
/* CLI lifecycle tests use no file descriptors or physical I2C devices. */
#include "vl53l8cx_api.h"
#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned polls;
static int initialized, configured, started;
static int scenario(const char *name)
{ const char *mode = getenv("TOF_MOCK"); return mode && !strcmp(mode, name); }

int platform_open(VL53L8CX_Platform *p, const char *path, unsigned address,
                  unsigned chunk, volatile sig_atomic_t *cancelled)
{
    (void)path;
    assert(address == 0x29);
    memset(p, 0, sizeof(*p)); p->fd = -1; p->chunk = chunk; p->cancelled = cancelled;
    fputs("MOCK_OPEN\n", stderr);
    return 0;
}
void platform_begin(VL53L8CX_Platform *p, unsigned timeout_ms)
{ (void)timeout_ms; p->error = 0; }
void platform_close(VL53L8CX_Platform *p)
{ (void)p; fputs("MOCK_CLOSE\n", stderr); }
uint8_t WaitMs(VL53L8CX_Platform *p, uint32_t ms)
{
    (void)ms;
    if (scenario("timeout")) { p->error = ETIMEDOUT; return 255; }
    return 0;
}
uint8_t vl53l8cx_is_alive(VL53L8CX_Configuration *p, uint8_t *alive)
{ (void)p; *alive = !scenario("wrong_id"); return 0; }
uint8_t vl53l8cx_init(VL53L8CX_Configuration *p)
{
    (void)p; initialized = 1;
    if (scenario("init_error")) return 5;
    if (scenario("init_signal")) raise(SIGINT);
    return 0;
}
uint8_t vl53l8cx_set_resolution(VL53L8CX_Configuration *p, uint8_t v)
{ (void)p; assert(initialized && v == 64); configured = 1; return 0; }
uint8_t vl53l8cx_set_ranging_mode(VL53L8CX_Configuration *p, uint8_t v)
{ (void)p; assert(configured && v == 3); return 0; }
uint8_t vl53l8cx_set_integration_time_ms(VL53L8CX_Configuration *p, uint32_t v)
{ (void)p; assert(configured && v == 5); return 0; }
uint8_t vl53l8cx_set_ranging_frequency_hz(VL53L8CX_Configuration *p, uint8_t v)
{ (void)p; assert(configured && v == 5); return 0; }
uint8_t vl53l8cx_set_target_order(VL53L8CX_Configuration *p, uint8_t v)
{ (void)p; assert(configured && v == 2); return 0; }
uint8_t vl53l8cx_start_ranging(VL53L8CX_Configuration *p)
{ (void)p; assert(configured); started = 1; return scenario("start_error") ? 255 : 0; }
uint8_t vl53l8cx_stop_ranging(VL53L8CX_Configuration *p)
{ (void)p; assert(started); fputs("MOCK_STOP\n", stderr); return scenario("stop_error") ? 255 : 0; }
uint8_t vl53l8cx_check_data_ready(VL53L8CX_Configuration *p, uint8_t *ready)
{
    (void)p; assert(started); polls++; *ready = !scenario("timeout");
    if (scenario("signal")) raise(SIGINT);
    return 0;
}
uint8_t vl53l8cx_get_ranging_data(VL53L8CX_Configuration *p, VL53L8CX_ResultsData *r)
{
    (void)p; assert(polls > 0);
    if (scenario("read_error")) return 255;
    for (unsigned i = 0; i < 64; i++) {
        r->distance_mm[i] = (int16_t)(500 + i);
        r->target_status[i] = 5; r->nb_target_detected[i] = 1;
    }
    r->target_status[1] = 9; r->target_status[2] = 3;
    r->nb_target_detected[3] = 0;
    return 0;
}
