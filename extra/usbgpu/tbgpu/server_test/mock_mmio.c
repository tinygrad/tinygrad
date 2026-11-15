#include <sys/socket.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

static size_t remaining;
static int finished, interrupted;

ssize_t mock_mmio_sendmsg(int fd, const struct msghdr *msg, int flags) {
  if (!remaining && !finished && msg->msg_iovlen && msg->msg_iov[0].iov_len == 17) {
    const uint8_t *header = msg->msg_iov[0].iov_base;
    uint64_t size;
    memcpy(&size, header + 1, sizeof(size));
    if (!header[0] && size == 4096) remaining = 17 + size;
  }
  if (!remaining) return sendmsg(fd, msg, flags);
  if (getenv("MOCK_MMIO_WAIT_CLOSE")) {
    char byte;
    while (recv(fd, &byte, 1, MSG_PEEK) < 0 && errno == EINTR);
  }
  if (getenv("MOCK_MMIO_EINTR") && !interrupted) {
    interrupted = 1;
    errno = EINTR;
    return -1;
  }
  if (getenv("MOCK_MMIO_ZERO") || getenv("MOCK_MMIO_ERROR")) {
    remaining = 0;
    finished = 1;
    errno = EIO;
    return getenv("MOCK_MMIO_ZERO") ? 0 : -1;
  }

  struct iovec iov[2];
  struct msghdr limited = *msg;
  size_t budget = getenv("MOCK_MMIO_LIMIT") ? strtoul(getenv("MOCK_MMIO_LIMIT"), NULL, 10) : remaining;
  limited.msg_iov = iov;
  limited.msg_iovlen = 0;
  for (size_t i = 0; i < msg->msg_iovlen && i < 2 && budget; i++) {
    iov[i] = msg->msg_iov[i];
    if (iov[i].iov_len > budget) iov[i].iov_len = budget;
    budget -= iov[i].iov_len;
    limited.msg_iovlen++;
  }
  ssize_t sent = sendmsg(fd, &limited, flags);
  if (sent > 0) {
    remaining -= (size_t)sent;
    if (!remaining) finished = 1;
  } else if (sent == 0 || errno != EINTR) {
    remaining = 0;
    finished = 1;
  }
  return sent;
}

ssize_t mock_mmio_send(int fd, const void *data, size_t size, int flags) {
  struct iovec iov = {(void*)data, size};
  struct msghdr msg = {.msg_iov = &iov, .msg_iovlen = 1};
  return mock_mmio_sendmsg(fd, &msg, flags);
}
