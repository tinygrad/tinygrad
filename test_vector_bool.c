typedef _Bool bool4 __attribute__((vector_size(4)));

int main() {
  bool4 a = {0, 1, 0, 1};
  _Bool b = a[1];
  return b;
}
