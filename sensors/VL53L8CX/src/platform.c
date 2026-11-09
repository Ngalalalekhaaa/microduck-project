/* SPDX-License-Identifier: BSD-3-Clause */
#define _POSIX_C_SOURCE 200809L
#include "platform.h"
#include <errno.h>
#include <fcntl.h>
#include <linux/i2c.h>
#include <linux/i2c-dev.h>
#include <string.h>
#include <sys/file.h>
#include <sys/ioctl.h>
#include <time.h>
#include <unistd.h>

#define MAX_CHUNK 4096U

int64_t platform_now_ms(void)
{
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts)) return -1;
    return (int64_t)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

static uint8_t fail(VL53L8CX_Platform *p, int error)
{
    if (!p->error) p->error = error;
    return 255;
}

static uint8_t check(VL53L8CX_Platform *p)
{
    if (p->error) return 255;
    if (p->cancelled && *p->cancelled) return fail(p, EINTR);
    int64_t now = platform_now_ms();
    if (now < 0) return fail(p, EIO);
    if (p->deadline_ms && now >= p->deadline_ms) return fail(p, ETIMEDOUT);
    return 0;
}

int platform_open(VL53L8CX_Platform *p, const char *path, unsigned address,
                  unsigned chunk, volatile sig_atomic_t *cancelled)
{
    memset(p, 0, sizeof(*p));
    p->fd = -1;
    if (address < 0x08 || address > 0x77 || !chunk || chunk > MAX_CHUNK) {
        errno = EINVAL;
        return -1;
    }
    p->address = (uint16_t)(address << 1);
    p->chunk = chunk;
    p->cancelled = cancelled;
    p->fd = open(path, O_RDWR | O_CLOEXEC);
    if (p->fd < 0) return -1;
    unsigned long funcs = 0;
    /* Advisory lock prevents two copies of this tool from using the bus. */
    if (flock(p->fd, LOCK_EX | LOCK_NB) < 0 ||
        ioctl(p->fd, I2C_FUNCS, &funcs) < 0) goto error;
    if (!(funcs & I2C_FUNC_I2C)) { errno = ENOTSUP; goto error; }
    /* Reject kernel-owned addresses; deliberately do not use I2C_SLAVE_FORCE. */
    if (ioctl(p->fd, I2C_SLAVE, (unsigned long)address) < 0) goto error;
    return 0;
error:;
    int saved = errno;
    close(p->fd);
    p->fd = -1;
    errno = saved;
    return -1;
}

void platform_begin(VL53L8CX_Platform *p, unsigned timeout_ms)
{
    p->error = 0;
    int64_t now = platform_now_ms();
    p->deadline_ms = now + timeout_ms;
    if (now < 0) p->error = EIO;
}

void platform_close(VL53L8CX_Platform *p)
{
    if (p->fd >= 0) close(p->fd);
    p->fd = -1;
}

static uint8_t transfer(VL53L8CX_Platform *p, struct i2c_msg *msgs, unsigned n)
{
    if (check(p)) return 255;
    struct i2c_rdwr_ioctl_data data = { .msgs = msgs, .nmsgs = n };
    int result = ioctl(p->fd, I2C_RDWR, &data);
    /* Partial transfers must never be treated as complete. */
    if (result != (int)n) return fail(p, result < 0 ? errno : EIO);
    return check(p);
}

static uint8_t validate(VL53L8CX_Platform *p, uint16_t reg,
                        const uint8_t *values, uint32_t size)
{
    if (!values || !p->chunk || p->chunk > MAX_CHUNK ||
        size > 0x10000U - reg) return fail(p, EINVAL);
    return check(p);
}

uint8_t RdMulti(VL53L8CX_Platform *p, uint16_t reg, uint8_t *values, uint32_t size)
{
    if (validate(p, reg, values, size)) {
        /* ULD sometimes inspects a read buffer after a failed callback. */
        if (values && size <= 0x10000U) memset(values, 0, size);
        return 255;
    }
    memset(values, 0, size);
    for (uint32_t offset = 0; offset < size;) {
        uint32_t count = size - offset;
        if (count > p->chunk) count = p->chunk;
        uint16_t index = (uint16_t)(reg + offset);
        uint8_t bytes[2] = { (uint8_t)(index >> 8), (uint8_t)index };
        struct i2c_msg msgs[2] = {
            { .addr = p->address >> 1, .flags = 0, .len = 2, .buf = bytes },
            { .addr = p->address >> 1, .flags = I2C_M_RD,
              .len = (uint16_t)count, .buf = values + offset }
        };
        if (transfer(p, msgs, 2)) { memset(values, 0, size); return 255; }
        offset += count;
    }
    return 0;
}

uint8_t WrMulti(VL53L8CX_Platform *p, uint16_t reg, uint8_t *values, uint32_t size)
{
    if (validate(p, reg, values, size)) return 255;
    uint8_t bytes[MAX_CHUNK + 2];
    for (uint32_t offset = 0; offset < size;) {
        uint32_t count = size - offset;
        if (count > p->chunk) count = p->chunk;
        uint16_t index = (uint16_t)(reg + offset);
        bytes[0] = (uint8_t)(index >> 8);
        bytes[1] = (uint8_t)index;
        memcpy(bytes + 2, values + offset, count);
        struct i2c_msg msg = { .addr = p->address >> 1, .flags = 0,
                              .len = (uint16_t)(count + 2), .buf = bytes };
        if (transfer(p, &msg, 1)) return 255;
        offset += count;
    }
    return 0;
}

uint8_t RdByte(VL53L8CX_Platform *p, uint16_t reg, uint8_t *value)
{ return RdMulti(p, reg, value, 1); }

uint8_t WrByte(VL53L8CX_Platform *p, uint16_t reg, uint8_t value)
{ return WrMulti(p, reg, &value, 1); }

void SwapBuffer(uint8_t *buffer, uint16_t size)
{
    for (uint32_t i = 0; i + 3 < size; i += 4) {
        uint8_t a = buffer[i], b = buffer[i + 1];
        buffer[i] = buffer[i + 3]; buffer[i + 1] = buffer[i + 2];
        buffer[i + 2] = b; buffer[i + 3] = a;
    }
}

uint8_t WaitMs(VL53L8CX_Platform *p, uint32_t ms)
{
    if (check(p)) return 255;
    struct timespec remaining = { .tv_sec = ms / 1000,
                                  .tv_nsec = (long)(ms % 1000) * 1000000 };
    while (nanosleep(&remaining, &remaining) < 0) {
        if (errno != EINTR) return fail(p, errno);
        if (check(p)) return 255;
    }
    return check(p);
}
