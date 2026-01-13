# Makefile for tinygpu shim

CC = clang
CFLAGS = -Wall -Wextra -O2
FRAMEWORKS = -framework IOKit -framework CoreFoundation

all: shim

shim: shim.c
	$(CC) $(CFLAGS) $(FRAMEWORKS) -o $@ $<

clean:
	rm -f shim

sign:
	codesign --force --sign - shim

.PHONY: all clean sign
