CC ?= cc
CPPFLAGS += -Isrc -Ivendor/vl53l8cx
CFLAGS ?= -O2 -g
CFLAGS += -std=c11 -Wall -Wextra

.PHONY: all clean test
all: build/vl53l8cx_test

build:
	mkdir -p $@

build/platform.o: src/platform.c src/platform.h | build
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

build/vl53l8cx_test.o: src/vl53l8cx_test.c src/platform.h vendor/vl53l8cx/vl53l8cx_api.h | build
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

build/vl53l8cx_api.o: vendor/vl53l8cx/vl53l8cx_api.c vendor/vl53l8cx/vl53l8cx_api.h vendor/vl53l8cx/vl53l8cx_buffers.h src/platform.h | build
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

build/vl53l8cx_test: build/platform.o build/vl53l8cx_test.o build/vl53l8cx_api.o
	$(CC) $(LDFLAGS) $^ $(LDLIBS) -o $@

build/test_platform: tests/test_platform.c build/platform.o | build
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -Wl,--wrap=ioctl $(LDFLAGS) $(LDLIBS) -o $@

build/test_cli_mock: src/vl53l8cx_test.c tests/mock_driver.c | build
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ $(LDFLAGS) $(LDLIBS) -o $@

test: all build/test_platform build/test_cli_mock
	./build/test_platform
	python3 tests/test_cli.py

clean:
	rm -rf build
