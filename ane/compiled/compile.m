#include <stdio.h>
#import <CoreFoundation/CoreFoundation.h>

int ANECCompile(CFDictionaryRef param_1, CFDictionaryRef param_2, unsigned long param_3);

int main() {
  CFTypeRef ikeys[2];
  ikeys[0] = CFSTR("NetworkPlistName");
  ikeys[1] = CFSTR("NetworkPlistPath");

  CFTypeRef ivalues[2];
  ivalues[0] = CFSTR("net.plist");
  ivalues[1] = CFSTR("/Users/taylor/fun/tinygrad/ane/compiled");
  
  CFDictionaryRef iDictionary = CFDictionaryCreate(kCFAllocatorDefault, ikeys, ivalues, 2, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
  CFArrayRef array = CFArrayCreate(kCFAllocatorDefault, (void**)&iDictionary, 1, &kCFTypeArrayCallBacks);

  CFMutableDictionaryRef optionsDictionary = CFDictionaryCreateMutable(kCFAllocatorDefault, 0, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
  CFMutableDictionaryRef flagsDictionary = CFDictionaryCreateMutable(kCFAllocatorDefault, 0, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);

  CFDictionaryAddValue(optionsDictionary, CFSTR("InputNetworks"), array);
  CFDictionaryAddValue(optionsDictionary, CFSTR("OutputFileName"), CFSTR("model.hwx.tmp"));
  CFDictionaryAddValue(optionsDictionary, CFSTR("OutputFilePath"), CFSTR("/Users/taylor/fun/tinygrad/ane/compiled"));

  CFDictionaryAddValue(flagsDictionary, CFSTR("TargetArchitecture"), CFSTR("h13"));

  CFShow(optionsDictionary);
  CFShow(flagsDictionary);

  printf("hello\n");
  int ret = ANECCompile(optionsDictionary, flagsDictionary, 0);
  printf("compile: %d\n", ret);
}
