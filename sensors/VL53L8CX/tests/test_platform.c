/* SPDX-License-Identifier: BSD-3-Clause */
#include "platform.h"
#include <assert.h>
#include <errno.h>
#include <linux/i2c.h>
#include <linux/i2c-dev.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

static unsigned calls, offset, base, read_mode, partial, chunk;
static uint8_t source[0x8000];

int __wrap_ioctl(int fd, unsigned long request, ...)
{
    (void)fd;
    assert(request == I2C_RDWR);
    va_list args;
    va_start(args, request);
    struct i2c_rdwr_ioctl_data *data = va_arg(args, struct i2c_rdwr_ioctl_data *);
    va_end(args);
    calls++;
    if (partial) return (int)data->nmsgs - 1;
    struct i2c_msg *first = &data->msgs[0];
    assert(first->addr == 0x29 && first->flags == 0);
    assert(((unsigned)first->buf[0] << 8 | first->buf[1]) == base + offset);
    if (read_mode) {
        assert(data->nmsgs == 2 && first->len == 2);
        struct i2c_msg *read = &data->msgs[1];
        assert(read->addr == 0x29 && read->flags == I2C_M_RD);
        assert(read->len > 0 && read->len <= chunk);
        memcpy(read->buf, source + offset, read->len);
        offset += read->len;
    } else {
        assert(data->nmsgs == 1 && first->len > 2 && first->len <= chunk + 2);
        assert(!memcmp(first->buf + 2, source + offset, first->len - 2));
        offset += first->len - 2;
    }
    return (int)data->nmsgs;
}

int main(void)
{
    VL53L8CX_Platform p = { .fd = -1, .address = 0x52, .chunk = 1024 };
    for (unsigned i = 0; i < sizeof(source); i++) source[i] = (uint8_t)(i * 13);
    chunk = p.chunk;
    assert(!WrMulti(&p, 0, source, sizeof(source)));
    assert(calls == 32 && offset == sizeof(source));
    uint8_t values[2051];
    calls = offset = 0; read_mode = 1; base = 0x1234;
    assert(!RdMulti(&p, (uint16_t)base, values, sizeof(values)));
    assert(calls == 3 && offset == sizeof(values));
    assert(!memcmp(values, source, sizeof(values)));
    partial = 1; calls = 0;
    assert(RdMulti(&p, 0, values, sizeof(values)) == 255);
    assert(p.error == EIO && calls == 1);
    for (unsigned i = 0; i < sizeof(values); i++) assert(values[i] == 0);
    assert(WrByte(&p, 0, 4) == 255 && calls == 1); /* sticky failure */
    partial = 0; p.error = 0;
    volatile sig_atomic_t signal_number = SIGINT;
    p.cancelled = &signal_number;
    assert(WrByte(&p, 0, 4) == 255 && p.error == EINTR && calls == 1);
    p.cancelled = NULL; p.error = 0; p.deadline_ms = 1;
    assert(WaitMs(&p, 1) == 255 && p.error == ETIMEDOUT);
    p.deadline_ms = 0; p.error = 0;
    assert(WrMulti(&p, 0xffff, source, 2) == 255 && p.error == EINVAL);
    uint8_t swap[] = {0, 1, 2, 3, 4, 5, 6, 7};
    uint8_t expected[] = {3, 2, 1, 0, 7, 6, 5, 4};
    SwapBuffer(swap, sizeof(swap));
    assert(!memcmp(swap, expected, sizeof(swap)));
    puts("platform mock: firmware chunking, repeated-start reads, address bytes, partial I/O, cancellation, timeout and byte swap OK");
    return 0;
}
