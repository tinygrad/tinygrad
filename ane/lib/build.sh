#!/bin/bash
clang++ ane.mm --shared -F /System/Library/PrivateFrameworks/ -framework ANEServices -framework IOSurface -framework Foundation -framework IOKit -o ane.dylib
codesign --entitlements entitlements.xml -s "Taylor Swift Child" ane.dylib

