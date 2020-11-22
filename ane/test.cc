#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

// clang++ -std=c++17 test.cc -F /System/Library/PrivateFrameworks/ -framework ANEServices


namespace H11ANE {
  class H11ANEDevice;
  class H11ANEServicesThreadParams;
  typedef int ANEDeviceUsageType;
  struct H11ANEDeviceInfoStruct;

  class H11ANEDeviceController {
    public:
      H11ANEDeviceController(int (*callback)(H11ANE::H11ANEDeviceController*, void*, H11ANE::H11ANEDevice*), void *arg);
  };

  class H11ANEDevice {
    public:
      H11ANEDevice(H11ANE::H11ANEDeviceController *param_1, unsigned int param_2);
      unsigned long H11ANEDeviceOpen(int (*callback)(H11ANE::H11ANEDevice*, unsigned int, void*, void*), void *param_2, ANEDeviceUsageType param_3, H11ANEDeviceInfoStruct *param_4);
  };

  //unsigned long H11ANEServicesThreadStart(H11ANE::H11ANEServicesThreadParams *param_1);
  unsigned long CreateH11ANEDeviceController(H11ANE::H11ANEDeviceController**,
    int (*callback)(H11ANE::H11ANEDeviceController*, void*, H11ANE::H11ANEDevice*), void *arg);

};

using namespace H11ANE;

H11ANEDevice *device = NULL;

int MyH11ANEDeviceControllerNotification(H11ANEDeviceController *param_1, void *param_2, H11ANEDevice *param_3) {
  printf("callback %p %p %p\n", param_1, param_2, param_3);
  device = param_3;
  return 0;
}

extern "C" {
  // called
  int H11ANEDeviceOpen(unsigned long *param_1, void *param_2, unsigned long param_3, unsigned long param_4);

  // programRequest has size 0x2038
  int H11ANEProgramProcessRequestDirect(H11ANEDevice *pANEDevice, void *programRequest,void *requestCallback);

  // never called
  int H11InitializePlatformServices(void);
  int H11ANEProgramCreate(long param_1, long *param_2, long *param_3);

  int H11ANEProgramPrepare(long param_1,long *param_2, unsigned long param_3);
}

/*int MyH11ANEDeviceMessageNotification(H11ANEDevice *param_1, unsigned int param_2, void *param_3, void *param_4) {
  printf("here\n");
  return 0;
}*/

int main() {
  printf("hello %d\n", getpid());

  int ret2 = H11InitializePlatformServices();
  printf("init 0x%X\n", ret2);

  H11ANEDevice *dev = NULL;
  char settings[0x20] = {0};
  ret2 = H11ANEDeviceOpen((unsigned long *)&dev, settings, 0, 0);
  printf("open 0x%X %p\n", ret2, dev);

  /*H11ANEDeviceController *ret = NULL;
  CreateH11ANEDeviceController(&ret, MyH11ANEDeviceControllerNotification, NULL);
  printf("%p %p\n", ret, device);*/

  //dev->H11ANEServicesThreadStart(NULL);
  

  /*auto *tmp = new H11ANEDeviceController(callback, NULL);
  printf("%p\n", tmp);

  auto *dev = new H11ANEDevice(NULL, 0);
  printf("%p\n", dev);*/

  //ANECDumpAnalytics(NULL, 0);
}


