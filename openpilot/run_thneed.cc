#include "thneed.h"

int main(int argc, char *argv[]) {
  auto thneed = new Thneed(true);
  thneed->record = false;
  thneed->load(argv[1]);
  thneed->clexec();
}