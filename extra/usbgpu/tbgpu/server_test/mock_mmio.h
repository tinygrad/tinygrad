#include <sys/socket.h>

ssize_t mock_mmio_send(int fd, const void *data, size_t size, int flags);
ssize_t mock_mmio_sendmsg(int fd, const struct msghdr *msg, int flags);
#define send mock_mmio_send
#define sendmsg mock_mmio_sendmsg
