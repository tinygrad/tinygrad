#!/bin/bash
clang++ -std=c++17 test.cc -F /System/Library/PrivateFrameworks/ -framework ANEServices 
codesign --entitlements entitlements.xml -s "Taylor Swift Child" a.out

