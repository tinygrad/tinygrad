// thanks to https://github.com/freedomtan/TestANE

// gcc -framework Foundation query.m && ./a.out

#import <Foundation/Foundation.h>
#import <objc/runtime.h>

int main (int argc, const char * argv[])
{
  NSBundle *aneBundle = [NSBundle bundleWithPath:@"/System/Library/PrivateFrameworks/AppleNeuralEngine.framework"];
  BOOL success = [aneBundle load];
  NSLog(@"load results: %d", success);

  Class _ANEDeviceInfo = NSClassFromString(@"_ANEDeviceInfo");
  NSLog(@"aneBundleLoaded: %s", success ? "yes" : "no");
  NSLog(@"buildVersion: %@", [_ANEDeviceInfo buildVersion]);
  NSLog(@"hasANE: %s", [_ANEDeviceInfo hasANE] ? "yes" : "no");
  NSLog(@"isInternalBuild: %s", [_ANEDeviceInfo isInternalBuild] ? "yes" : "no");
  NSLog(@"precompiledModelCheckDisabled: %s", [_ANEDeviceInfo precompiledModelChecksDisabled] ? "yes" : "no");

  Class _ANEClient = NSClassFromString(@"_ANEClient");
  [_ANEClient initialize];

  Class elem = [_ANEClient class]; 
  while (elem) {
    NSLog(@"%s", class_getName(elem));
    unsigned int numMethods = 0;
    Method *mList = class_copyMethodList(elem, &numMethods);
    if (mList) {
      for (int j = 0; j < numMethods; j++) {
        NSLog(@" %s", sel_getName(method_getName(mList[j])));
      }
      free(mList);
    }
    elem = class_getSuperclass(elem);
    // no superclas support
    break;
  }
}

