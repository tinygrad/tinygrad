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
H11ANEDevice *dev;

int ANE_Open() {
  H11ANEDeviceController dc(MyH11ANEDeviceControllerNotification, NULL);
  dc.SetupDeviceController();
  assert(device != NULL);
  H11ANEDevice *dev = device;
  dev->EnableDeviceMessages();

  char empty[0x90] = {0};
  H11ANEDeviceInfoStruct dis = {0};
  ret = dev->H11ANEDeviceOpen(MyH11ANEDeviceMessageNotification, empty, UsageCompile, &dis);
  printf("open 0x%x %p\n", ret, dev);

  ret = dev->ANE_PowerOn();
  printf("power on: %d\n", ret);

  ret = dev->ANE_IsPowered();
  printf("powered? %d\n", ret);
}

int stride_for_width(int width) {
  int ret = width*2;
  ret += (64-ret) % 64;
  return ret;
}

int ANE_CreateTensor(int width, int height) {
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
  int in_surf_id = IOSurfaceGetID(in_surf);
  printf("we have surface %p with id 0x%x\n", in_surf, in_surf_id);

  return in_surf_id;
}

uint64_t ANE_Compile(char *prog, int sz) {
  H11ANEProgramCreateArgsStruct mprog = {0};
  mprog.program = prog;
  mprog.program_length = 0x8000;

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

void ANE_Run(uint64_t program_handle, int in_surf_id, int out_surf_id) {
  H11ANEProgramRequestArgsStruct *pras = new H11ANEProgramRequestArgsStruct;
  memset(pras, 0, sizeof(H11ANEProgramRequestArgsStruct));

  // TODO: make real struct
  pras->args[0] = out->program_handle;
  pras->args[4] = 0x0000002100000003;

  // inputs
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
}

}

