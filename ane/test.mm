#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

#import <IOSurface/IOSurfaceRef.h>

#import <Foundation/Foundation.h>
#import <CoreFoundation/CoreFoundation.h>

// clang++ -std=c++17 test.cc -F /System/Library/PrivateFrameworks/ -framework ANEServices

enum ANEDeviceUsageType {
  UsageNoProgram,
  UsageWithProgram,
  UsageCrazy
};

struct H11ANEDeviceInfoStruct {
  uint64_t nothing;
  uint64_t program_handle;
};

namespace H11ANE {
  class H11ANEDevice;
  class H11ANEServicesThreadParams;
  struct H11ANEProgramRequestArgsStruct;

  class H11ANEDeviceController {
    public:
      H11ANEDeviceController(int (*callback)(H11ANE::H11ANEDeviceController*, void*, H11ANE::H11ANEDevice*), void *arg);
  };

  class H11ANEDevice {
    public:
      H11ANEDevice(H11ANE::H11ANEDeviceController *param_1, unsigned int param_2);

      unsigned long H11ANEDeviceOpen(int (*callback)(H11ANE::H11ANEDevice*, unsigned int, void*, void*), void *param_2, ANEDeviceUsageType param_3, H11ANEDeviceInfoStruct *param_4);

      int ANE_IsPowered();

      int ANE_ProgramSendRequest(H11ANEProgramRequestArgsStruct*, unsigned int);

      int ANE_ReadANERegister(unsigned int param_1, unsigned int *param_2);
      int ANE_ForgetFirmware();
      int ANE_PowerOn();
      int ANE_PowerOff();

      void EnableDeviceMessages();
  };

  //unsigned long H11ANEServicesThreadStart(H11ANE::H11ANEServicesThreadParams *param_1);
  unsigned long CreateH11ANEDeviceController(H11ANE::H11ANEDeviceController**,
    int (*callback)(H11ANE::H11ANEDeviceController*, void*, H11ANE::H11ANEDevice*), void *arg);

};

using namespace H11ANE;

H11ANEDevice *device = NULL;

int MyH11ANEDeviceControllerNotification(H11ANEDeviceController *param_1, void *param_2, H11ANEDevice *param_3) {
  printf("MyH11ANEDeviceControllerNotification %p %p %p\n", param_1, param_2, param_3);
  device = param_3;
  return 0;
}

int MyH11ANEDeviceMessageNotification(H11ANE::H11ANEDevice* dev, unsigned int param_1, void* param_2, void* param_3) {
  printf("MyH11ANEDeviceMessageNotification %d %p %p\n", param_1, param_2, param_3);
  return 0;
}


extern "C" {
  // called
  // param_3 = _ANEDeviceController
  // param_4 = AppleNeuralEngineDeviceCallback
  int H11ANEDeviceOpen(unsigned long *param_1, void *param_2, unsigned long param_3, unsigned long param_4);

  // programRequest has size 0x2038
  // the compiler already ran by this point
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

struct programRequest {
  int zero;
  int num;
  IOSurfaceRef surf;
  
  // extra
  char junk[0x2038-0x10];
};

/* 
breakpoint set -r . -s ANEServices


H11ANEDeviceOpen
  CreateH11ANEDeviceController
    H11ANEThreadReadySyncer
    ...boring thread stuff...
  AllocateStatsBufferPools
  CreateH11ANEFrameReceiver
    H11ANEFrameReceiver
H11ANEProgramProcessRequestDirect
  H11ANE::H11ANEFrameReceiver::ProgramProcessRequest
    H11ANE::H11ANEDevice::ANE_ProgramSendRequest
    H11ANE::H11ANEFrameReceiver::startNoDataTimer

+[_ANEClient initialize]
  +[_ANELog framework]
+[_ANEClient sharedConnection]
  -[_ANEClient initWithRestrictedAccessAllowed:]
    +[_ANEDaemonConnection daemonConnection]
      -[_ANEDaemonConnection init]
        +[_ANEStrings machServiceName]
        -[_ANEDaemonConnection initWithMachServiceName:restricted:]
+[_ANEModel modelAtURL:key:]
  +[_ANEModel modelAtURL:key:modelAttributes:]
    -[_ANEModel initWithModelAtURL:key:modelAttributes:]
-[_ANEClient loadModel:options:qos:error:]
  -[_ANEClient doLoadModel:options:qos:error:]
    *** this happens for a long time, and I suspect runs the compiler ***
    *** the H11ANEDeviceOpen happens in here
  +[_ANEDeviceController controllerWithProgramHandle:]
    -[_ANEDeviceController initWithProgramHandle:priviledged:]
  -[_ANEDeviceController start]
-[_ANEClient evaluateWithModel:options:request:qos:error:] 
  *** the H11ANEProgramProcessRequestDirect happens in here

breakpoint set -r .::H11ANEDevice::. -s ANEServices

H11ANE::H11ANEDevice::H11ANEDevice(H11ANE::H11ANEDeviceController*, unsigned int)
  H11ANE::H11ANEDevice::EnableDeviceMessages()
H11ANE::H11ANEDevice::H11ANEDeviceOpen(int (*)(H11ANE::H11ANEDevice*, unsigned int, void*, void*), void*, ANEDeviceUsageType, H11ANEDeviceInfoStruct*)

H11ANE::H11ANEDevice::ANE_ProgramSendRequest(H11ANEProgramRequestArgsStruct*, unsigned int)

<in device open in my process>
  H11ANE::H11ANEDevice::ANE_GetVersion(H11ANEVersion*)
  H11ANE::H11ANEDevice::ANE_GetStatus(H11ANEStatusStruct*)
  H11ANE::H11ANEDevice::ANE_PowerOn()
    H11ANE::H11ANEDevice::ANE_IsPowered()

*/

int main() {
  NSBundle *aneBundle = [NSBundle bundleWithPath:@"/System/Library/PrivateFrameworks/AppleNeuralEngine.framework"];
  BOOL success = [aneBundle load];
  NSLog(@"load results: %d", success);

  printf("hello %d\n", getpid());
  int ret2, ret;

  // kern_return_t IOServiceOpen(io_service_t service, task_port_t owningTask, uint32_t type, io_connect_t *connect);

  /*io_connect_t connection;
  ret = IOServiceOpen(0x5303, mach_task_self(), 0, &connection);
  printf("service open: %d %p\n", ret, connection);
  exit(0);*/

  //ret2 = H11InitializePlatformServices();
  //printf("init 0x%X\n", ret2);



  /*uint64_t settings[4] = {0};
  // first two are some array thing (1 and unaligned pointer in real call)
  //settings[0] = 1;
  // this is the program handle
  //settings[1] = 0x00000029c2de11a2;
  settings[2] = 0;
  settings[3] = 5000;*/

  //H11ANEDeviceController device(NULL, NULL);

  /*Class _ANEDeviceController = NSClassFromString(@"_ANEDeviceController");
  NSObject *obj = [[_ANEDeviceController alloc] init];
  printf("%p\n", obj);*/

  /*ret2 = H11ANEDeviceOpen((unsigned long *)&dev, settings, (unsigned long)0, 0);
  printf("open 0x%X %p\n", ret2, dev);*/

  // this gets the connection port
  H11ANEDeviceController *dc = NULL;
  CreateH11ANEDeviceController(&dc, MyH11ANEDeviceControllerNotification, NULL);
  printf("%p %p\n", dc, device);

  H11ANEDevice *dev = device; //new H11ANEDevice(dc, 0xaabbcc);
  printf("construct %p\n", dev);

  dev->EnableDeviceMessages();

  char empty[0x90];
  H11ANEDeviceInfoStruct dis = {0};
  dis.program_handle = 0x0000004f4afdade2;
  ret = dev->H11ANEDeviceOpen(MyH11ANEDeviceMessageNotification, empty, UsageNoProgram, &dis);
  printf("open 0x%x\n", ret);

  int is_powered;

  ret = dev->ANE_PowerOn();
  printf("power on: %d\n", ret);

  // need moar privilege
  /*unsigned int reg = 0;
  ret = dev->ANE_ReadANERegister(0, &reg);
  printf("reg 0x%x %lx\n", ret, reg);*/

  /*for (int i =0 ; i < 5; i++) {
    is_powered = dev->ANE_IsPowered();
    printf("powered? %d\n", is_powered);
    sleep(1);
  }

  ret = dev->ANE_PowerOff();
  printf("power off: %d\n", ret);

  for (int i =0 ; i < 5; i++) {
    is_powered = dev->ANE_IsPowered();
    printf("powered? %d\n", is_powered);
    sleep(1);
  }*/

  /*ret = dev->ANE_ForgetFirmware();
  printf("forget? 0x%lx\n", ret);*/

  exit(0);



  exit(0);

  NSDictionary* dict = [NSDictionary dictionaryWithObjectsAndKeys:
                           [NSNumber numberWithInt:1], kIOSurfaceWidth,
                           [NSNumber numberWithInt:3], kIOSurfaceHeight,
                           [NSNumber numberWithInt:4], kIOSurfaceBytesPerElement,
                           nil];
  IOSurfaceRef surf = IOSurfaceCreate((CFDictionaryRef)dict);
  printf("we have surface %p\n", surf);

  //char programRequest[0x2038] = {0};
  struct programRequest pr = {0};
  pr.num = 1;
  pr.surf = surf;
  ret2 = H11ANEProgramProcessRequestDirect(dev, &pr, ^() { printf("callback"); } );
  printf("run 0x%X\n", ret2);


  //dev->H11ANEServicesThreadStart(NULL);
  

  /*auto *tmp = new H11ANEDeviceController(callback, NULL);
  printf("%p\n", tmp);

  auto *dev = new H11ANEDevice(NULL, 0);
  printf("%p\n", dev);*/

  //ANECDumpAnalytics(NULL, 0);
}


