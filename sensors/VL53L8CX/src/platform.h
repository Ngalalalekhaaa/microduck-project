/* SPDX-License-Identifier: BSD-3-Clause */
#ifndef MICRODUCK_VL53L8CX_PLATFORM_H
#define MICRODUCK_VL53L8CX_PLATFORM_H

#include <stdint.h>
#include <signal.h>

typedef struct {
    uint16_t address; /* ST's eight-bit convention: Linux address << 1. */
    int fd;
    uint32_t chunk;
    int error; /* Sticky errno: no further transfers after a failed operation. */
    int64_t deadline_ms;
    volatile sig_atomic_t *cancelled;
} VL53L8CX_Platform;

#define VL53L8CX_NB_TARGET_PER_ZONE 1U
#define VL53L8CX_DISABLE_AMBIENT_PER_SPAD
#define VL53L8CX_DISABLE_NB_SPADS_ENABLED
#define VL53L8CX_DISABLE_SIGNAL_PER_SPAD
#define VL53L8CX_DISABLE_RANGE_SIGMA_MM
#define VL53L8CX_DISABLE_REFLECTANCE_PERCENT
#define VL53L8CX_DISABLE_MOTION_INDICATOR
/* Keep number of targets, distance, and status together. */

int64_t platform_now_ms(void);
int platform_open(VL53L8CX_Platform *p, const char *path, unsigned address,
                  unsigned chunk, volatile sig_atomic_t *cancelled);
void platform_begin(VL53L8CX_Platform *p, unsigned timeout_ms);
void platform_close(VL53L8CX_Platform *p);
uint8_t RdByte(VL53L8CX_Platform *p, uint16_t reg, uint8_t *value);
uint8_t WrByte(VL53L8CX_Platform *p, uint16_t reg, uint8_t value);
uint8_t RdMulti(VL53L8CX_Platform *p, uint16_t reg, uint8_t *values, uint32_t size);
uint8_t WrMulti(VL53L8CX_Platform *p, uint16_t reg, uint8_t *values, uint32_t size);
void SwapBuffer(uint8_t *buffer, uint16_t size);
uint8_t WaitMs(VL53L8CX_Platform *p, uint32_t ms);
#endif
