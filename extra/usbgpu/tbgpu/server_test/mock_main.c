// the TinyGPU app's Swift code calls run_server(path); this is the same for the test build
#include <stdio.h>
int run_server(const char *sock_path);
int main(int argc, char **argv) {
  if (argc != 2) { fprintf(stderr, "usage: %s <socket path>\n", argv[0]); return 2; }
  return run_server(argv[1]);
}
