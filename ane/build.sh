#!/bin/bash
clang++ test.mm -F /System/Library/PrivateFrameworks/ -framework ANEServices -framework IOSurface -framework Foundation -framework IOKit
codesign --entitlements entitlements.xml -s "Taylor Swift Child" a.out

