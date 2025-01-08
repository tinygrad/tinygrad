#!/bin/bash
INSTALL_PATH="${1:-/opt/homebrew/lib}"
[ ! -d "$INSTALL_PATH" ] && mkdir -p "$INSTALL_PATH"

# Download libremu.dylib
curl -s https://api.github.com/repos/Qazalin/remu/releases/latest | \
    jq -r '.assets[] | select(.name == "libremu.dylib").browser_download_url' | \
    xargs curl -L -o $INSTALL_PATH/libremu.dylib

# Download libamd_comgr.dylib
curl -s https://api.github.com/repos/nimlgen/amdcomgr_dylib/releases/latest | \
    jq -r '.assets[] | select(.name == "libamd_comgr.dylib").browser_download_url' | \
    xargs curl -L -o $INSTALL_PATH/libamd_comgr.dylib
