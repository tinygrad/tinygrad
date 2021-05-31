#include <stdint.h>
#define reg_leds (*(volatile uint32_t*)0x02000000)

void delay();

int main() {
  while (1) {
    for (int i = 1; i < 0x100; i <<= 1) {
      reg_leds = i;
      delay();
    }
  }
}

void __attribute__ ((noinline)) delay() {
  asm ("lui t0, 0x100\n"
       "lop:"
       "addi t0,t0,-0x1\n"
       "bne t0,zero,lop\n"
       ::);
}

