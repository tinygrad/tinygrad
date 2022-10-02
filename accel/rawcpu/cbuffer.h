class CBuffer {
public:
  CBuffer(int size_, void* dat = NULL) {
    size = size_;
    buf = (float*)malloc(size*4);
  }

  void copyin(void *dat) {
    memcpy(buf, dat, size*4);
  }

  void add(CBuffer *x, CBuffer *y) {
    for (int i = 0; i < size; i++) {
      buf[i] = x->buf[i] + y->buf[i];
    }
  }

  void mul(CBuffer *x, CBuffer *y) {
    for (int i = 0; i < size; i++) {
      buf[i] = x->buf[i] * y->buf[i];
    }
  }

  float *buf;
  int size;
};
