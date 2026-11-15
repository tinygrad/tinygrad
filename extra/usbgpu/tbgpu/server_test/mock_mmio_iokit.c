// Minimal IOKit replacement for MMIO reply tests; BAR contents persist across connections.
#include <stdint.h>
#include <mach/mach.h>
#include <IOKit/IOKitLib.h>
#include <IOKit/IOMessage.h>

#define KR_BAD_ARG 0xE00002C2u
#define KR_UNSUPPORTED 0xE00002C7u
static uint8_t bar_mem[6][1 << 20];
const mach_port_t kIOMainPortDefault = 0;

CFMutableDictionaryRef IOServiceNameMatching(const char *name) { (void)name; return NULL; }
io_service_t IOServiceGetMatchingService(mach_port_t p, CFDictionaryRef m) { (void)p; (void)m; return 1; }
IONotificationPortRef IONotificationPortCreate(mach_port_t p) { (void)p; return (IONotificationPortRef)1; }
void IONotificationPortSetDispatchQueue(IONotificationPortRef n, dispatch_queue_t q) { (void)n; (void)q; }
kern_return_t IOServiceAddInterestNotification(IONotificationPortRef n, io_service_t s, const io_name_t t, IOServiceInterestCallback c, void *r, io_object_t *o) {
  (void)n; (void)s; (void)t; (void)c; (void)r; *o = 1; return 0;
}
kern_return_t IOObjectRelease(io_object_t o) { (void)o; return 0; }
kern_return_t IOServiceOpen(io_service_t s, task_port_t t, uint32_t type, io_connect_t *c) { (void)s; (void)t; (void)type; *c = 1; return 0; }
kern_return_t IOServiceClose(io_connect_t c) { (void)c; return 0; }
kern_return_t IOConnectMapMemory64(io_connect_t c, uint32_t type, task_port_t t, mach_vm_address_t *a, mach_vm_size_t *s, IOOptionBits o) {
  (void)c; (void)t; (void)o;
  if (type >= 6) return KR_BAD_ARG;
  *a = (mach_vm_address_t)bar_mem[type]; *s = sizeof(bar_mem[type]);
  return 0;
}
kern_return_t IOConnectUnmapMemory64(io_connect_t c, uint32_t type, task_port_t t, mach_vm_address_t a) { (void)c; (void)type; (void)t; (void)a; return 0; }

kern_return_t IOConnectCallMethod(mach_port_t c, uint32_t sel, const uint64_t *in, uint32_t in_cnt, const void *is, size_t isc, uint64_t *out, uint32_t *out_cnt, void *os, size_t *osc) {
  (void)c; (void)is; (void)isc; (void)os; (void)osc;
  if (sel != 0) return KR_UNSUPPORTED;
  if (in_cnt != 2 || *out_cnt < 1) return KR_BAD_ARG;
  out[0] = in[0] == 0 ? 0x1002 : 0;
  *out_cnt = 1;
  return 0;
}

kern_return_t IOConnectCallStructMethod(mach_port_t c, uint32_t sel, const void *in, size_t in_sz, void *out, size_t *out_sz) {
  (void)c; (void)sel; (void)in; (void)in_sz; (void)out; (void)out_sz;
  return KR_UNSUPPORTED;
}
