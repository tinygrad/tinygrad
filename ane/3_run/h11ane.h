enum ANEDeviceUsageType {
  UsageNoProgram,
  UsageWithProgram,  // used in running process
  UsageCompile       // used in aned
};

struct H11ANEDeviceInfoStruct {
  uint64_t nothing;
  uint64_t program_handle;
  uint64_t junk[0x100];
};

struct H11ANEProgramCreateArgsStruct {
  void *program;
  uint64_t program_length;
  uint64_t empty[4];
  char has_signature;
};

struct H11ANEProgramCreateArgsStructOutput {
  uint64_t program_handle;
  int unknown[0x2000];
};

struct H11ANEProgramPrepareArgsStruct {
  uint64_t program_handle;
  uint64_t flags;
  uint64_t empty[0x100];
};

struct H11ANEProgramRequestArgsStruct {
  uint64_t args[0x1000];
};

namespace H11ANE {
  class H11ANEDevice;

  class H11ANEDeviceController {
    public:
      H11ANEDeviceController(
        int (*callback)(H11ANE::H11ANEDeviceController*, void*, H11ANE::H11ANEDevice*),
        void *arg);
      int SetupDeviceController();
    private:   // size is 0x50
      CFArrayRef array_ref;
      mach_port_t *master_port;
      IONotificationPortRef port_ref;
      CFRunLoopSourceRef source_ref;
      int (*callback)(H11ANE::H11ANEDeviceController*, void*, H11ANE::H11ANEDevice*);
      void *callback_arg;
      CFRunLoopRef run_loop_ref;
      io_iterator_t io_iterator;
      pthread_t thread_self;
      uint64_t unused;
  };

  class H11ANEDevice {
    public:
      H11ANEDevice(H11ANE::H11ANEDeviceController *param_1, unsigned int param_2);

      unsigned long H11ANEDeviceOpen(
        int (*callback)(H11ANE::H11ANEDevice*, unsigned int, void*, void*),
        void *param_2, ANEDeviceUsageType param_3, H11ANEDeviceInfoStruct *param_4);

      void EnableDeviceMessages();

      // power management
      int ANE_IsPowered();
      int ANE_PowerOn();
      int ANE_PowerOff();

      // logging (e00002c7 error, needs PE_i_can_has_debugger)
      int ANE_CreateClientLoggingSession(unsigned int log_iosurface);
      int ANE_TerminateClientLoggingSession(unsigned int log_iosurface);
      int ANE_GetDriverLoggingFlags(unsigned int *flags);
      int ANE_SetDriverLoggingFlags(unsigned int flags);

      // program creation
      int ANE_ProgramCreate(H11ANEProgramCreateArgsStruct*,
                            H11ANEProgramCreateArgsStructOutput*);
      int ANE_ProgramPrepare(H11ANEProgramPrepareArgsStruct*);
      int ANE_ProgramSendRequest(H11ANEProgramRequestArgsStruct*, mach_port_t);

      // need PE_i_can_has_debugger
      int ANE_ReadANERegister(unsigned int param_1, unsigned int *param_2);
      int ANE_ForgetFirmware();


    private:   // size is 0x88 
      unsigned char unknown[0x88];
  };

};

