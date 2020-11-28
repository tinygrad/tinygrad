#!/bin/bash
gcc compile.m -F /System/Library/PrivateFrameworks/ -framework ANECompiler -framework CoreFoundation
./a.out
log show --process a.out --last 1m --info --debug

