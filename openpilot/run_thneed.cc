#include "thneed.h"
#include "common/timing.h"

int main(int argc, char *argv[]) {
  auto thneed = new Thneed(true);
  thneed->record = false;
  thneed->load(argv[1]);

  for (int i = 0; i < 5; i++) {
    uint64_t sb = nanos_since_boot();
    thneed->clexec();
    uint64_t et = nanos_since_boot();
    printf("run in %.2f ms\n", (et-sb)*1e-6);
  }
}