typedef int int4 __attribute__((vector_size(16)));

int main() {
  int4 a = {1, 2, 3, 4};
  int b = a[0];
  return b;
}
