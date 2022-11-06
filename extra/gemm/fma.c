int main() {
  float A[0x1000];
  float B[0x1000];
  float C[0x1000];
  for (int i = 0; i < 0x1000; i++) {
    C[i] += A[i] * B[i];
  }
}
