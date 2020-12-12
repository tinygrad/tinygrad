#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

#import <IOSurface/IOSurfaceRef.h>

#import <Foundation/Foundation.h>
#import <CoreFoundation/CoreFoundation.h>

#include "h11ane.h"
using namespace H11ANE;

extern "C" {

// global vars
H11ANEDevice *dev = NULL;

int MyH11ANEDeviceControllerNotification(H11ANEDeviceController *param_1, void *param_2, H11ANEDevice *param_3) {
  printf("MyH11ANEDeviceControllerNotification %p %p %p\n", param_1, param_2, param_3);
  dev = param_3;
  return 0;
}

int MyH11ANEDeviceMessageNotification(H11ANE::H11ANEDevice* dev, unsigned int param_1, void* param_2, void* param_3) {
  printf("MyH11ANEDeviceMessageNotification %d %p %p\n", param_1, param_2, param_3);
  return 0;
}

int ANE_Open() {
  int ret;
  H11ANEDeviceController dc(MyH11ANEDeviceControllerNotification, NULL);
  dc.SetupDeviceController();
  assert(dev != NULL);
  dev->EnableDeviceMessages();

  char empty[0x90] = {0};
  H11ANEDeviceInfoStruct dis = {0};
  ret = dev->H11ANEDeviceOpen(MyH11ANEDeviceMessageNotification, empty, UsageCompile, &dis);
  printf("open 0x%x %p\n", ret, dev);

  ret = dev->ANE_PowerOn();
  printf("power on: %d\n", ret);

  ret = dev->ANE_IsPowered();
  printf("powered? %d\n", ret);

  return 0;
}

int stride_for_width(int width) {
  int ret = width*2;
  ret += (64-ret) % 64;
  return ret;
}

void *ANE_TensorCreate(int width, int height) {
  // all float16
  // input buffer

  NSDictionary* dict = [NSDictionary dictionaryWithObjectsAndKeys:
                           [NSNumber numberWithInt:width], kIOSurfaceWidth,
                           [NSNumber numberWithInt:height], kIOSurfaceHeight,
                           [NSNumber numberWithInt:2], kIOSurfaceBytesPerElement,
                           [NSNumber numberWithInt:stride_for_width(width)], kIOSurfaceBytesPerRow,
                           [NSNumber numberWithInt:1278226536], kIOSurfacePixelFormat,
                           nil];
  IOSurfaceRef in_surf = IOSurfaceCreate((CFDictionaryRef)dict);

  return (void *)in_surf;
}

void* ANE_TensorData(void *out_surf) {
  return (void *)IOSurfaceGetBaseAddress((IOSurfaceRef)out_surf);
}

uint64_t ANE_Compile(char *prog, int sz) {
  int ret;
  printf("ANE_Compile %p with size %d\n", prog, sz);
  H11ANEProgramCreateArgsStruct mprog = {0};
  mprog.program = prog;
  mprog.program_length = sz;

  H11ANEProgramCreateArgsStructOutput *out = new H11ANEProgramCreateArgsStructOutput;
  memset(out, 0, sizeof(H11ANEProgramCreateArgsStructOutput));
  ret = dev->ANE_ProgramCreate(&mprog, out);
  uint64_t program_handle = out->program_handle;
  delete out;
  printf("program create: %lx %lx\n", ret, program_handle);

  H11ANEProgramPrepareArgsStruct pas = {0};
  pas.program_handle = program_handle;
  pas.flags = 0x0000000100010001;
  ret = dev->ANE_ProgramPrepare(&pas);
  printf("program prepare: %lx\n", ret);

  return program_handle;
}

int ANE_Run(uint64_t program_handle, void *in_surf, void *out_surf) {
  int ret;
  H11ANEProgramRequestArgsStruct *pras = new H11ANEProgramRequestArgsStruct;
  memset(pras, 0, sizeof(H11ANEProgramRequestArgsStruct));

  // TODO: make real struct
  pras->args[0] = program_handle;
  pras->args[4] = 0x0000002100000003;

  // inputs
  int in_surf_id = IOSurfaceGetID((IOSurfaceRef)in_surf);
  int out_surf_id = IOSurfaceGetID((IOSurfaceRef)out_surf);

  pras->args[0x28/8] = 1;
  pras->args[0x128/8] = (long long)in_surf_id<<32LL;

  // outputs
  pras->args[0x528/8] = 1;
  // 0x628 = outputBufferSurfaceId
  pras->args[0x628/8] = (long long)out_surf_id<<32LL;

  mach_port_t recvPort = 0;
  IOCreateReceivePort(kOSAsyncCompleteMessageID, &recvPort);
  printf("recv port: 0x%x\n", recvPort);

  // run program
  ret = dev->ANE_ProgramSendRequest(pras, recvPort);
  printf("send 0x%x\n", ret);

  struct {
    mach_msg_header_t header;
    char data[256];
  } message;

  ret = mach_msg(&message.header,
          MACH_RCV_MSG,
          0, sizeof(message),
          recvPort,
          MACH_MSG_TIMEOUT_NONE,
          MACH_PORT_NULL);
  printf("got message: %d sz %d\n", ret, message.header.msgh_size);
  delete pras;

  return 0;
}

}

