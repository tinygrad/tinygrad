#!/bin/bash
gcc compile.m -F /System/Library/PrivateFrameworks/ -framework ANECompiler -framework CoreFoundation
rm -f model.hwx
./a.out net.plist
log show --process a.out --last 1m --info --debug

