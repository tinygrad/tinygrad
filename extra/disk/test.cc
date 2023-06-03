#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

#include <thread>
#include <chrono>

//#define FN "/dev/nvme0n1"
#define FN "../../weights/LLaMA/7B/consolidated.00.pth"

#define SZ (unsigned long long)(512*1024*1024)
#define CNT 10LL

void test_read() {
#ifdef O_DIRECT
  int f = open(FN, O_RDONLY|O_DIRECT);
#else
  int f = open(FN, O_RDONLY);
  //fcntl(f, F_NOCACHE, 1);
#endif
  printf("open %d\n", f);

  /*void *buf = malloc(CNT*SZ);
  printf("malloc %p\n", buf);
  mlock(buf, CNT*SZ);*/

  auto t0 = std::chrono::high_resolution_clock::now();
  void *buf = mmap(NULL, SZ*CNT, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);

  auto t1 = std::chrono::high_resolution_clock::now();
  mlock(buf, CNT*SZ);
  for (int i = 0; i < CNT; i++) {
    read(f, (unsigned char*)buf+SZ*i, SZ);
  }
  auto t2 = std::chrono::high_resolution_clock::now();

  //free(buf);
  printf("malloc %p\n", buf);
  float ns = (float)std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count();
  float pns = (float)std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count();
  printf("read %.2f GB in %.2f s (%.2f s to prepare), %.2f GB/s\n", SZ/1e9*CNT, ns*1e-9, pns*1e-9, (SZ*CNT)/ns);

  close(f);
  munmap(buf, SZ*CNT);
}

void test_mmap() {
#ifdef O_DIRECT
  int f = open(FN, O_RDONLY|O_DIRECT);
#else
  int f = open(FN, O_RDONLY);
#endif
  printf("open %d\n", f);

  void *dat = mmap(NULL, SZ*CNT, PROT_READ, MAP_PRIVATE, f, 0);

  auto t1 = std::chrono::high_resolution_clock::now();
  mlock(dat, SZ*CNT);
  auto t2 = std::chrono::high_resolution_clock::now();

  printf("mmap %p\n", dat);

  float ns = (float)std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count();
  printf("read %.2f GB in %.2f s, %.2f GB/s\n", SZ/1e9*CNT, ns*1e-9, (SZ*CNT)/ns);

  close(f);
  munlock(dat, SZ*CNT);
  munmap(dat, SZ*CNT);
}

int main() {
  //system("sync; echo 1 > /proc/sys/vm/drop_caches");
  //system("sudo purge");
  //test_mmap();

  //system("sync; echo 1 > /proc/sys/vm/drop_caches");
  system("sudo purge");
  test_read();
  test_read();
}

