#include <stdint.h>
#define reg_leds (*(volatile uint32_t*)0x02000000)
#define reg_uart_clkdiv (*(volatile uint32_t*)0x02000004)
#define reg_uart_data (*(volatile uint32_t*)0x02000008)

void delay();

int main() {
  // 50 mhz clock
  reg_uart_clkdiv = 434;
  while (1) {
    for (int i = 1; i < 0x100; i <<= 1) {
      if (i == 4) {
        sdelay();
        reg_uart_data = 'h';
        sdelay();
        reg_uart_data = 'e';
        sdelay();
        reg_uart_data = 'l';
        sdelay();
        reg_uart_data = 'l';
        sdelay();
        reg_uart_data = 'o';
      }
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

void __attribute__ ((noinline)) sdelay() {
  asm ("lui t0, 0x10\n"
       "lop2:"
       "addi t0,t0,-0x1\n"
       "bne t0,zero,lop2\n"
       ::);
}

