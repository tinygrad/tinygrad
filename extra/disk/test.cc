#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

#include <chrono>

#define SZ (unsigned long long)(16*1000*1000)
#define CNT 100LL

void *test_read() {
  int f = open("/dev/nvme0n1", O_RDONLY|O_DIRECT);
  printf("open %d\n", f);

  // 16 MB
  void *buf = malloc(SZ);

  auto t1 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < CNT; i++) {
    read(f, buf, SZ);
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  printf("read %.2f GB, %.2f GB/s\n", SZ/1e9*CNT, (SZ*CNT)/(float)std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count());

  return buf;
}

void *test_mmap() {
  int f = open("/dev/nvme0n1", O_RDONLY|O_DIRECT);
  printf("open %d\n", f);

  // 16 MB
  void *buf = malloc(SZ);

  void *dat = mmap(NULL, SZ*CNT, PROT_READ, MAP_SHARED, f, 0);
  printf("mmap %p\n", dat);

  auto t1 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < CNT; i++) {
    memcpy(buf, (unsigned char*)dat+SZ*i, SZ);
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  printf("read %.2f GB, %.2f GB/s\n", SZ/1e9*CNT, (SZ*CNT)/(float)std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count());

  return buf;
}

int main() {
  system("sync; echo 1 > /proc/sys/vm/drop_caches");
  test_read();

  system("sync; echo 1 > /proc/sys/vm/drop_caches");
  free(test_mmap());
}

