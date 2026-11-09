/* SPDX-License-Identifier: BSD-3-Clause */
#define _POSIX_C_SOURCE 200809L
#include "vl53l8cx_api.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static volatile sig_atomic_t interrupted;
static void on_signal(int signal_number) { interrupted = signal_number; }

static void usage(FILE *out)
{
    fputs("VL53L8CX standalone Linux test (ST ULD " VL53L8CX_API_REVISION ")\n"
          "Usage: vl53l8cx_test probe|range [options]\n"
          "  --bus PATH             default /dev/i2c-3\n"
          "  --address NUMBER       Linux 7-bit address, default 0x29 (NOT 0x52)\n"
          "  --frames NUMBER        range frames, 1..100000, default 10\n"
          "  --hz NUMBER            8x8 frame rate, 1..15, default 5\n"
          "  --chunk NUMBER         I2C payload bytes, 1..4096, default 1024\n"
          "  --init-timeout-ms N     initialization timeout, default 30000\n"
          "  --frame-timeout-ms N    per-frame timeout, default 5000\n"
          "  --csv                  raw distance/status/count per zone as CSV\n"
          "  --help                 no hardware access\n"
          "probe checks only the specified address using ST's ID check; it writes\n"
          "the register-page selector. range uploads firmware and emits 8x8 frames.\n"
          "Confirm the chosen connector, module power/I2C/control pins before use.\n"
          "Configured HAT J6 uses --bus /dev/i2c-10; original J5 uses /dev/i2c-3.\n",
          out);
}

static int number(const char *text, unsigned min, unsigned max, unsigned *out)
{
    if (!*text || *text == '-') return -1;
    char *end;
    errno = 0;
    unsigned long value = strtoul(text, &end, 0);
    if (errno || *end || value < min || value > max) return -1;
    *out = (unsigned)value;
    return 0;
}

static int failed(VL53L8CX_Configuration *dev, const char *operation, uint8_t status)
{
    if (!status && !dev->platform.error && !interrupted) return 0;
    fprintf(stderr, "%s failed: ULD status=%u", operation, status);
    if (dev->platform.error) fprintf(stderr, "; %s", strerror(dev->platform.error));
    if (interrupted) fprintf(stderr, "; signal %d", interrupted);
    fputc('\n', stderr);
    return 1;
}

static void print_frame(const VL53L8CX_ResultsData *result, unsigned frame, int csv)
{
    if (csv) {
        for (unsigned zone = 0; zone < 64; zone++) {
            unsigned status = result->target_status[zone];
            unsigned count = result->nb_target_detected[zone];
            const char *quality = !count ? "none" : status == 5 ? "valid" :
                                  (status == 6 || status == 9) ? "reduced" : "invalid";
            printf("%u,%u,%u,%u,%d,%u,%u,%s\n", frame, zone, zone / 8, zone % 8,
                   result->distance_mm[zone], status, count, quality);
        }
    } else {
        printf("\nFrame %u; zone order 0..63; mm (~ = reduced confidence, ---- = invalid)\n", frame);
        for (unsigned zone = 0; zone < 64; zone++) {
            unsigned status = result->target_status[zone];
            if (!result->nb_target_detected[zone]) printf(" %6s", "----");
            else if (status == 5) printf(" %6d", result->distance_mm[zone]);
            else if (status == 6 || status == 9) printf(" ~%5d", result->distance_mm[zone]);
            else printf(" %6s", "----");
            if (zone % 8 == 7) putchar('\n');
        }
        puts("Target status (5=valid; 6/9=reduced; others not accepted):");
        for (unsigned zone = 0; zone < 64; zone++) {
            printf(" %6u", result->target_status[zone]);
            if (zone % 8 == 7) putchar('\n');
        }
    }
    fflush(stdout);
}

int main(int argc, char **argv)
{
    if (argc == 1) { usage(stdout); return 0; }
    if (!strcmp(argv[1], "--help")) { usage(stdout); return 0; }
    int range = !strcmp(argv[1], "range");
    if (!range && strcmp(argv[1], "probe")) {
        usage(stderr); return 2;
    }
    const char *bus = "/dev/i2c-3";
    unsigned address = 0x29, frames = 10, hz = 5, chunk = 1024;
    unsigned init_timeout = 30000, frame_timeout = 5000;
    int csv = 0;
    for (int i = 2; i < argc; i++) {
        const char *option = argv[i];
        if (!strcmp(option, "--help")) { usage(stdout); return 0; }
        if (!strcmp(option, "--csv")) { csv = 1; continue; }
        if (i + 1 >= argc) {
            fprintf(stderr, "Missing value for %s\n", option); return 2;
        }
        const char *value = argv[++i];
        int error = 0;
        if (!strcmp(option, "--bus")) { bus = value; error = !*value; }
        else if (!strcmp(option, "--address")) error = number(value, 0x08, 0x77, &address);
        else if (!strcmp(option, "--frames")) error = number(value, 1, 100000, &frames);
        else if (!strcmp(option, "--hz")) error = number(value, 1, 15, &hz);
        else if (!strcmp(option, "--chunk")) error = number(value, 1, 4096, &chunk);
        else if (!strcmp(option, "--init-timeout-ms")) error = number(value, 1000, 300000, &init_timeout);
        else if (!strcmp(option, "--frame-timeout-ms")) error = number(value, 100, 60000, &frame_timeout);
        else error = 1;
        if (error) { fprintf(stderr, "Invalid option/value: %s %s\n", option, value); return 2; }
    }
    struct sigaction action = {0};
    action.sa_handler = on_signal;
    sigemptyset(&action.sa_mask);
    if (sigaction(SIGINT, &action, NULL) || sigaction(SIGTERM, &action, NULL) ||
        sigaction(SIGPIPE, &action, NULL)) {
        perror("sigaction"); return 1;
    }
    /* Single-threaded, large objects outside the stack. */
    static VL53L8CX_Configuration dev;
    static VL53L8CX_ResultsData result;
    if (platform_open(&dev.platform, bus, address, chunk, &interrupted) < 0) {
        fprintf(stderr, "Cannot open/use %s at 7-bit 0x%02x: %s\n", bus, address, strerror(errno));
        return interrupted ? 128 + interrupted : 1;
    }
    int rc = 1, start_attempted = 0;
    uint8_t alive = 0, status;
    platform_begin(&dev.platform, 3000);
    status = vl53l8cx_is_alive(&dev, &alive);
    if (failed(&dev, "probe", status)) goto cleanup;
    if (!alive) {
        fputs("Device/revision ID does not match VL53L8CX (expected F0/0C).\n", stderr);
        goto cleanup;
    }
    fprintf(stderr, "VL53L8CX detected at %s, Linux 7-bit 0x%02x; ULD %s\n",
            bus, address, VL53L8CX_API_REVISION);
    if (!range) { rc = 0; goto cleanup; }
    fprintf(stderr, "Uploading ST firmware and initializing (timeout %u ms)...\n", init_timeout);
    platform_begin(&dev.platform, init_timeout);
    if (failed(&dev, "init", vl53l8cx_init(&dev))) goto cleanup;
    platform_begin(&dev.platform, 5000);
    if (failed(&dev, "set_resolution", vl53l8cx_set_resolution(&dev, VL53L8CX_RESOLUTION_8X8))) goto cleanup;
    if (failed(&dev, "set_mode", vl53l8cx_set_ranging_mode(&dev, VL53L8CX_RANGING_MODE_AUTONOMOUS))) goto cleanup;
    if (failed(&dev, "set_integration", vl53l8cx_set_integration_time_ms(&dev, 5))) goto cleanup;
    if (failed(&dev, "set_frequency", vl53l8cx_set_ranging_frequency_hz(&dev, (uint8_t)hz))) goto cleanup;
    if (failed(&dev, "set_target_order", vl53l8cx_set_target_order(&dev, VL53L8CX_TARGET_ORDER_STRONGEST))) goto cleanup;
    start_attempted = 1; /* A failed start may still have reached the hardware. */
    if (failed(&dev, "start", vl53l8cx_start_ranging(&dev))) goto cleanup;
    if (csv) puts("frame,zone,row,column,distance_mm,target_status,targets,quality");
    for (unsigned frame = 1; frame <= frames; frame++) {
        platform_begin(&dev.platform, frame_timeout);
        uint8_t ready = 0;
        do {
            if (failed(&dev, "check_data_ready", vl53l8cx_check_data_ready(&dev, &ready))) goto cleanup;
            if (!ready && failed(&dev, "wait_for_frame", WaitMs(&dev.platform, 10))) goto cleanup;
        } while (!ready);
        memset(&result, 0, sizeof(result));
        if (failed(&dev, "get_ranging_data", vl53l8cx_get_ranging_data(&dev, &result))) goto cleanup;
        print_frame(&result, frame, csv);
    }
    rc = 0;
cleanup:
    if (start_attempted) {
        /* Permit a bounded best-effort stop even after SIGINT or I/O failure. */
        dev.platform.cancelled = NULL;
        platform_begin(&dev.platform, 6000);
        status = vl53l8cx_stop_ranging(&dev);
        if (status || dev.platform.error) {
            fprintf(stderr, "Stop failed (ULD=%u, %s); sensor state is unconfirmed.\n",
                    status, dev.platform.error ? strerror(dev.platform.error) : "driver error");
            rc = 1;
        }
    }
    platform_close(&dev.platform);
    return interrupted ? 128 + interrupted : rc;
}
