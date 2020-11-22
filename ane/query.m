#import <Foundation/Foundation.h>

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
}

