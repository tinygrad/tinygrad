//#include "selfdrive/modeld/thneed/include/msm_kgsl.h"

class GPUMalloc {
  public:
    GPUMalloc(int size, int fd);
    ~GPUMalloc();
    void *alloc(int size);
  private:
    uint64_t base;
    int remaining;
};

class CachedIoctl {
  public:
    virtual void exec() {}
};

class CachedSync: public CachedIoctl {
  public:
    CachedSync(Thneed *lthneed, string ldata) { thneed = lthneed; data = ldata; }
    void exec();
  private:
    Thneed *thneed;
    string data;
};

class CachedCommand: public CachedIoctl {
  public:
    CachedCommand(Thneed *lthneed, struct kgsl_gpu_command *cmd);
    void exec();
  private:
    void disassemble(int cmd_index);
    struct kgsl_gpu_command cache;
    unique_ptr<kgsl_command_object[]> cmds;
    unique_ptr<kgsl_command_object[]> objs;
    Thneed *thneed;
    vector<shared_ptr<CLQueuedKernel> > kq;
};