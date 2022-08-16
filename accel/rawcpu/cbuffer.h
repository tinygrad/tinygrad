class CBuffer {
public:
  CBuffer(int size, void* dat = NULL) {
    buf = (float*)malloc(size*4);
    if (dat != NULL) memcpy(buf, dat, size*4);
  }
private:
  float *buf;
};

