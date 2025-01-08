#!/bin/bash

INSTALL_PATH="/opt/homebrew/lib"
mkdir -p $INSTALL_PATH

curl -s https://api.github.com/repos/Qazalin/remu/releases/latest | \
    jq -r '.assets[] | select(.name == "libremu.dylib").browser_download_url' | \
    xargs curl -L -o $INSTALL_PATH/libremu.dylib

curl -s https://api.github.com/repos/nimlgen/amdcomgr_dylib/releases/latest | \
    jq -r '.assets[] | select(.name == "libamd_comgr.dylib").browser_download_url' | \
    xargs curl -L -o $INSTALL_PATH/libamd_comgr.dylib
