#include <os/log.h>
#include <stdio.h>
#import <CoreFoundation/CoreFoundation.h>

int ANECCompile(CFDictionaryRef param_1, CFDictionaryRef param_2, unsigned long param_3);

int main(int argc, char* argv[]) {
  os_log(OS_LOG_DEFAULT, "start compiler");

  CFTypeRef ikeys[2];
  ikeys[0] = CFSTR("NetworkPlistName");
  ikeys[1] = CFSTR("NetworkPlistPath");

  CFTypeRef ivalues[2];
  ivalues[0] = CFStringCreateWithCString(kCFAllocatorDefault, argv[1], kCFStringEncodingUTF8);
  ivalues[1] = CFSTR("./");
  
  CFDictionaryRef iDictionary = CFDictionaryCreate(kCFAllocatorDefault, ikeys, ivalues, 2, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
  CFArrayRef array = CFArrayCreate(kCFAllocatorDefault, (const void**)&iDictionary, 1, &kCFTypeArrayCallBacks);

  CFMutableDictionaryRef optionsDictionary = CFDictionaryCreateMutable(kCFAllocatorDefault, 0, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
  CFMutableDictionaryRef flagsDictionary = CFDictionaryCreateMutable(kCFAllocatorDefault, 0, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);

  CFDictionaryAddValue(optionsDictionary, CFSTR("InputNetworks"), array);
  CFDictionaryAddValue(optionsDictionary, CFSTR("OutputFilePath"), CFSTR("./"));
  //CFDictionaryAddValue(optionsDictionary, CFSTR("OptionsFilePath"), CFSTR("good.options"));

  // h11 (or anything?) works here too, and creates different outputs that don't run
  CFDictionaryAddValue(flagsDictionary, CFSTR("TargetArchitecture"), CFSTR("h13"));

  if (argc > 2) {
    CFDictionaryAddValue(optionsDictionary, CFSTR("OutputFileName"), CFSTR("debug/model.hwx"));
    CFDictionaryAddValue(flagsDictionary, CFSTR("DebugDetailPrint"), kCFBooleanTrue);
    int debug_mask = 0x7fffffff;
    CFDictionaryAddValue(flagsDictionary, CFSTR("DebugMask"), CFNumberCreate(kCFAllocatorDefault, 3, &debug_mask));
  } else {
    CFDictionaryAddValue(optionsDictionary, CFSTR("OutputFileName"), CFSTR("model.hwx"));
  }
  //CFDictionaryAddValue(flagsDictionary, CFSTR("DisableMergeScaleBias"), kCFBooleanTrue);
  //CFDictionaryAddValue(flagsDictionary, CFSTR("Externs"), CFSTR("swag"));

  //CFShow(optionsDictionary);
  //CFShow(flagsDictionary);

  printf("hello\n");
  int ret = ANECCompile(optionsDictionary, flagsDictionary, 0);
  printf("compile: %d\n", ret);

  return ret;
}
