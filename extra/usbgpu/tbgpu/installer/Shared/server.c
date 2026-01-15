#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <errno.h>
#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/IOKitLib.h>
#include <mach/mach.h>

enum {
  CMD_MAP_BAR = 1, CMD_MAP_SYSMEM = 2, CMD_CFG_READ = 3, CMD_CFG_WRITE = 4,
  CMD_RESET = 5, CMD_MMIO_BULK_READ = 6, CMD_MMIO_BULK_WRITE = 7,
  CMD_SYSMEM_BULK_READ = 8, CMD_SYSMEM_BULK_WRITE = 9,
  RESP_OK = 0, RESP_ERR = 1,
};

#define RESP(call) { resp.status = (call) ? RESP_ERR : RESP_OK; break; }
#define RESP_VAL(call, v) { int _r = (call); resp.status = _r ? RESP_ERR : RESP_OK; if (!_r) resp.value = (v); break; }
#define RESP_ERR_BREAK() { resp.status = RESP_ERR; break; }
#define RESP_SEND_DATA(sz, data) { resp.status = RESP_OK; resp.value = (sz); send_response(fd, &resp, -1); send(fd, data, sz, 0); send_resp = 0; break; }
#define NO_RESP() { send_resp = 0; break; }

#define BULK_BUF_SIZE (16 * 1024 * 1024)
#define MAX_BARS 6
#define MAX_SYSMEM 64

typedef struct {
  uint8_t cmd, bar;
  uint64_t offset, size, value;
} __attribute__((packed)) request_t;

typedef struct {
  uint8_t status;
  uint64_t value, addr;
} __attribute__((packed)) response_t;

static uint8_t g_bulk_buf[BULK_BUF_SIZE];
static io_connect_t g_conn = IO_OBJECT_NULL;
static struct { mach_vm_address_t addr; mach_vm_size_t size; int mapped; } g_bars[MAX_BARS];
static struct { mach_vm_address_t addr; mach_vm_size_t size; uint64_t mem_type; int mapped; } g_sysmem[MAX_SYSMEM];
static int g_sysmem_count = 0, g_client_active = 0;

static io_connect_t open_tinygpu(void) {
  io_service_t svc = IOServiceGetMatchingService(kIOMainPortDefault, IOServiceNameMatching("tinygpu"));
  if (!svc) return IO_OBJECT_NULL;
  io_connect_t conn;
  kern_return_t kr = IOServiceOpen(svc, mach_task_self(), 0, &conn);
  IOObjectRelease(svc);
  return kr == KERN_SUCCESS ? conn : IO_OBJECT_NULL;
}

static int map_bar(uint32_t bar, response_t *resp) {
  if (bar >= MAX_BARS) return -1;
  if (g_bars[bar].mapped) { resp->addr = g_bars[bar].addr; resp->value = g_bars[bar].size; return 0; }
  mach_vm_address_t addr = 0;
  mach_vm_size_t size = 0;
  if (IOConnectMapMemory64(g_conn, bar, mach_task_self(), &addr, &size, kIOMapAnywhere) != KERN_SUCCESS) return -1;
  g_bars[bar].addr = addr; g_bars[bar].size = size; g_bars[bar].mapped = 1;
  resp->addr = addr; resp->value = size;
  printf("BAR%u: 0x%llx size 0x%llx\n", bar, addr, size);
  return 0;
}

static int map_sysmem(uint64_t size, response_t *resp) {
  if (g_sysmem_count >= MAX_SYSMEM) return -1;
  mach_vm_address_t addr = 0;
  mach_vm_size_t mapped_size = 0;
  if (IOConnectMapMemory64(g_conn, size, mach_task_self(), &addr, &mapped_size, kIOMapAnywhere) != KERN_SUCCESS) return -1;
  int idx = g_sysmem_count++;
  g_sysmem[idx].addr = addr; g_sysmem[idx].size = mapped_size; g_sysmem[idx].mem_type = size; g_sysmem[idx].mapped = 1;
  resp->addr = idx; resp->value = mapped_size;
  printf("sysmem[%d]: 0x%llx size 0x%llx\n", idx, addr, mapped_size);
  return 0;
}

static int cfg_write(uint64_t offset, uint64_t size, uint64_t value) {
  uint64_t in[3] = {offset, size, value}, out[16];
  uint32_t cnt = 16;
  return IOConnectCallMethod(g_conn, 1, in, 3, NULL, 0, out, &cnt, NULL, NULL) == KERN_SUCCESS ? 0 : -1;
}

static int validate_bar(uint8_t bar, uint64_t offset, uint64_t size) {
  return (bar < MAX_BARS && g_bars[bar].mapped && offset + size <= g_bars[bar].size && size <= BULK_BUF_SIZE) ? 0 : -1;
}

static int validate_sysmem(uint8_t idx, uint64_t offset, uint64_t size) {
  return (idx < g_sysmem_count && g_sysmem[idx].mapped && offset + size <= g_sysmem[idx].size && size <= BULK_BUF_SIZE) ? 0 : -1;
}

static void recvall(int fd, void *buf, size_t len) {
  for (size_t rcv = 0; rcv < len; ) {
    ssize_t r = recv(fd, (uint8_t*)buf + rcv, len - rcv, 0);
    if (r <= 0) break;
    rcv += r;
  }
}

static void drain(int fd, size_t len) {
  for (size_t rem = len; rem > 0; ) {
    ssize_t r = recv(fd, g_bulk_buf, rem > BULK_BUF_SIZE ? BULK_BUF_SIZE : rem, 0);
    if (r <= 0) break;
    rem -= r;
  }
}

static void mmio_copy(void *dst, void *src, size_t len) {
  volatile uint32_t *d4 = dst; volatile uint32_t *s4 = src;
  for (size_t i = 0; i < len / 4; i++) d4[i] = s4[i];
  volatile uint8_t *d1 = (volatile uint8_t*)(d4 + len / 4);
  volatile uint8_t *s1 = (volatile uint8_t*)(s4 + len / 4);
  for (size_t i = 0; i < len % 4; i++) d1[i] = s1[i];
}

static int send_response(int fd, response_t *resp, int send_fd) {
  struct msghdr msg = {0};
  struct iovec iov = {.iov_base = resp, .iov_len = sizeof(*resp)};
  msg.msg_iov = &iov; msg.msg_iovlen = 1;

  char cmsgbuf[CMSG_SPACE(sizeof(int))];
  if (send_fd >= 0) {
    msg.msg_control = cmsgbuf; msg.msg_controllen = sizeof(cmsgbuf);
    struct cmsghdr *cmsg = CMSG_FIRSTHDR(&msg);
    cmsg->cmsg_level = SOL_SOCKET; cmsg->cmsg_type = SCM_RIGHTS; cmsg->cmsg_len = CMSG_LEN(sizeof(int));
    memcpy(CMSG_DATA(cmsg), &send_fd, sizeof(int));
  }
  return sendmsg(fd, &msg, 0) > 0 ? 0 : -1;
}

static void cleanup(void) {
  for (int i = 0; i < MAX_BARS; i++) {
	if (g_bars[i].mapped) { IOConnectUnmapMemory64(g_conn, i, mach_task_self(), g_bars[i].addr); g_bars[i].mapped = 0; }
  }
  for (int i = 0; i < g_sysmem_count; i++) {
	if (g_sysmem[i].mapped) { IOConnectUnmapMemory64(g_conn, g_sysmem[i].mem_type, mach_task_self(), g_sysmem[i].addr); g_sysmem[i].mapped = 0; }
  }
  g_sysmem_count = 0;
  if (g_conn != IO_OBJECT_NULL) { IOServiceClose(g_conn); g_conn = IO_OBJECT_NULL; }
}

static void handle_client(int fd) {
  request_t req;
  response_t resp;
  int bufsize = BULK_BUF_SIZE;
  setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &bufsize, sizeof(bufsize));
  setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &bufsize, sizeof(bufsize));
  printf("client connected\n");

  while (1) {
    if (recv(fd, &req, sizeof(req), 0) != sizeof(req)) break;
    memset(&resp, 0, sizeof(resp));
    int send_resp = 1;

    switch (req.cmd) {
      case CMD_MAP_BAR: RESP(map_bar(req.bar, &resp));
      case CMD_MAP_SYSMEM: RESP(map_sysmem(req.size, &resp));
      case CMD_CFG_READ: {
        uint64_t in[2] = {req.offset, req.size}, out[16];
        uint32_t cnt = 16;
        RESP_VAL(IOConnectCallMethod(g_conn, 0, in, 2, NULL, 0, out, &cnt, NULL, NULL) != KERN_SUCCESS, out[0]);
      }
      case CMD_CFG_WRITE: {
        uint64_t in[3] = {req.offset, req.size, req.value}, out[16];
        uint32_t cnt = 16;
        RESP(IOConnectCallMethod(g_conn, 1, in, 3, NULL, 0, out, &cnt, NULL, NULL) != KERN_SUCCESS);
      }
      case CMD_RESET: {
        uint64_t out[16];
        uint32_t cnt = 16;
        RESP(IOConnectCallMethod(g_conn, 2, NULL, 0, NULL, 0, out, &cnt, NULL, NULL) != KERN_SUCCESS);
      }
      case CMD_MMIO_BULK_READ:
        if (validate_bar(req.bar, req.offset, req.size)) RESP_ERR_BREAK();
        mmio_copy(g_bulk_buf, (void*)(g_bars[req.bar].addr + req.offset), req.size);
        RESP_SEND_DATA(req.size, g_bulk_buf);
      case CMD_MMIO_BULK_WRITE:
        if (validate_bar(req.bar, req.offset, req.size)) { drain(fd, req.size); NO_RESP(); }
        recvall(fd, g_bulk_buf, req.size);
        mmio_copy((void*)(g_bars[req.bar].addr + req.offset), g_bulk_buf, req.size);
        NO_RESP();
      case CMD_SYSMEM_BULK_READ:
        if (validate_sysmem(req.bar, req.offset, req.size)) RESP_ERR_BREAK();
        RESP_SEND_DATA(req.size, (void*)(g_sysmem[req.bar].addr + req.offset));
      case CMD_SYSMEM_BULK_WRITE:
        if (validate_sysmem(req.bar, req.offset, req.size)) { drain(fd, req.size); NO_RESP(); }
        recvall(fd, (void*)(g_sysmem[req.bar].addr + req.offset), req.size);
        NO_RESP();
      default:
        RESP_ERR_BREAK();
    }
    if (send_resp) send_response(fd, &resp, -1);
  }

  printf("client disconnected\n");
  cleanup();
}

int run_server(const char *sock_path) {
  g_conn = open_tinygpu();
  if (g_conn == IO_OBJECT_NULL) { fprintf(stderr, "failed to connect to tinygpu driver\n"); return 1; }
  printf("connected to tinygpu driver\n");

  int server_fd = socket(AF_UNIX, SOCK_STREAM, 0);
  if (server_fd < 0) { perror("socket"); cleanup(); return 1; }

  struct sockaddr_un addr = {.sun_family = AF_UNIX};
  strncpy(addr.sun_path, sock_path, sizeof(addr.sun_path) - 1);
  unlink(sock_path);

  if (bind(server_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) { perror("bind"); close(server_fd); cleanup(); return 1; }
  if (listen(server_fd, 1) < 0) { perror("listen"); close(server_fd); cleanup(); return 1; }

  printf("listening on %s\n", sock_path);

  while (1) {
    int client_fd = accept(server_fd, NULL, NULL);
    if (client_fd < 0) { if (errno == EINTR) continue; perror("accept"); break; }
    if (g_client_active) { printf("rejected: client already connected\n"); close(client_fd); continue; }
    g_client_active = 1;
    handle_client(client_fd);
    g_client_active = 0;
    close(client_fd);
  }

  close(server_fd);
  unlink(sock_path);
  cleanup();
  return 0;
}
