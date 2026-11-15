// Shared IOKit test double: no dext or GPU. BARs are ordinary memory. Optional DMA emulation supports the release/import tests:
// an IOVA arena of MOCK_ARENA_MIB (default 2047), lowest-free slots, and (RELEASE_MAGIC, slot) after the address table.
// All emulated mappings are freed when the connection closes; this does not imply the installed dext supports ReleaseDMA.
// Environment: MOCK_ARENA_MIB, MOCK_SEG_MIB (split every mapping into segments of that size), MOCK_OLD_DEXT=1 (a dext without ReleaseDMA:
// no magic, selector 4 unsupported), MOCK_FAIL_RELEASE=N (the Nth ReleaseDMA fails with kIOReturnBusy and leaves the mapping).
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <mach/mach.h>
#include <IOKit/IOKitLib.h>
#include <IOKit/IOMessage.h>

#define KR_NO_MEMORY 0xE00002BDu
#define KR_BAD_ARG 0xE00002C2u
#define KR_UNSUPPORTED 0xE00002C7u
#define KR_BUSY 0xE00002D5u
#define MAGIC 0x52454C444D410001ull
#define ARENA_BASE 0x80000000ull
#define PAGE (16ull << 10)
#define MAX_SLOTS 4096

static struct { int live; uint64_t iova, size; } slots[MAX_SLOTS];
static int nslots, release_calls;
static void *bar_mem[6];
const mach_port_t kIOMainPortDefault = 0;

static uint64_t env_u64(const char *n, uint64_t d) { const char *v = getenv(n); return v && *v ? strtoull(v, NULL, 0) : d; }

static int arena_alloc(uint64_t size, uint64_t *iova) { // lowest free run of `size` bytes
  uint64_t cap = env_u64("MOCK_ARENA_MIB", 2047) << 20, at = 0;
  for (int moved = 1; moved;) {
    moved = 0;
    for (int i = 0; i < nslots; i++) {
      uint64_t lo = slots[i].iova - ARENA_BASE, hi = lo + slots[i].size;
      if (slots[i].live && lo < at + size && hi > at) { at = hi; moved = 1; }
    }
  }
  if (at + size > cap) return -1;
  *iova = ARENA_BASE + at;
  return 0;
}

static void free_all(void) { memset(slots, 0, sizeof slots); nslots = 0; release_calls = 0; }

CFMutableDictionaryRef IOServiceNameMatching(const char *name) { (void)name; return NULL; }
io_service_t IOServiceGetMatchingService(mach_port_t p, CFDictionaryRef m) { (void)p; (void)m; return 1; }
IONotificationPortRef IONotificationPortCreate(mach_port_t p) { (void)p; return (IONotificationPortRef)1; }
void IONotificationPortSetDispatchQueue(IONotificationPortRef n, dispatch_queue_t q) { (void)n; (void)q; }
kern_return_t IOServiceAddInterestNotification(IONotificationPortRef n, io_service_t s, const io_name_t t, IOServiceInterestCallback c, void *r, io_object_t *o) {
  (void)n; (void)s; (void)t; (void)c; (void)r; *o = 1; return 0;
}
kern_return_t IOObjectRelease(io_object_t o) { (void)o; return 0; }
kern_return_t IOServiceOpen(io_service_t s, task_port_t t, uint32_t type, io_connect_t *c) { (void)s; (void)t; (void)type; free_all(); *c = 1; return 0; }
kern_return_t IOServiceClose(io_connect_t c) { (void)c; free_all(); return 0; } // Stop_Impl: CompleteDMA on every command
kern_return_t IOConnectMapMemory64(io_connect_t c, uint32_t type, task_port_t t, mach_vm_address_t *a, mach_vm_size_t *s, IOOptionBits o) {
  (void)c; (void)t; (void)o;
  if (type >= 6) return KR_BAD_ARG;
  if (!bar_mem[type]) bar_mem[type] = calloc(1, 1 << 20);
  *a = (mach_vm_address_t)bar_mem[type]; *s = 1 << 20;
  return 0;
}
kern_return_t IOConnectUnmapMemory64(io_connect_t c, uint32_t type, task_port_t t, mach_vm_address_t a) { (void)c; (void)type; (void)t; (void)a; return 0; }

kern_return_t IOConnectCallMethod(mach_port_t c, uint32_t sel, const uint64_t *in, uint32_t in_cnt, const void *is, size_t isc, uint64_t *out, uint32_t *out_cnt, void *os, size_t *osc) {
  (void)c; (void)is; (void)isc; (void)os; (void)osc;
  switch (sel) {
  case 0: if (in_cnt != 2 || *out_cnt < 1) return KR_BAD_ARG; out[0] = in[0] == 0 ? 0x1002 : 0; *out_cnt = 1; return 0; // CfgRead: vendor id
  case 1: case 2: return 0; // CfgWrite, Reset
  case 4: { // ReleaseDMA
    if (env_u64("MOCK_OLD_DEXT", 0)) return KR_UNSUPPORTED;
    if (in_cnt != 1 || in[0] >= (uint64_t)nslots || !slots[in[0]].live) return KR_BAD_ARG;
    if (++release_calls == (int)env_u64("MOCK_FAIL_RELEASE", 0)) return KR_BUSY;
    slots[in[0]].live = 0;
    return 0;
  }
  }
  return KR_UNSUPPORTED;
}

kern_return_t IOConnectCallStructMethod(mach_port_t c, uint32_t sel, const void *in, size_t in_sz, void *out, size_t *out_sz) {
  (void)c; (void)in;
  if (sel != 3) return KR_UNSUPPORTED;
  if (in_sz < 4097 || *out_sz < 4097 || in_sz % PAGE) return KR_BAD_ARG; // "buffers >= 4097 bytes" (and whole 16 KiB pages)
  int slot = -1;
  for (int i = 0; i < nslots; i++) if (!slots[i].live) { slot = i; break; }
  if (slot < 0) { if (nslots == MAX_SLOTS) return KR_NO_MEMORY; slot = nslots; }
  uint64_t iova;
  if (arena_alloc(in_sz, &iova)) return KR_NO_MEMORY;
  uint64_t *t = out, seg = env_u64("MOCK_SEG_MIB", 0) << 20, n = 0;
  memset(out, 0, *out_sz);
  for (uint64_t o = 0; o < in_sz; o += seg ? seg : in_sz, n++) {
    if (n == 32) return KR_NO_MEMORY; // more segments than the dext's array
    t[2 * n] = iova + o; t[2 * n + 1] = (seg && in_sz - o > seg) ? seg : in_sz - o;
  }
  if (!env_u64("MOCK_OLD_DEXT", 0)) { t[2 * n + 2] = MAGIC; t[2 * n + 3] = slot; }
  slots[slot].live = 1; slots[slot].iova = iova; slots[slot].size = in_sz;
  if (slot == nslots) nslots++;
  return 0;
}
