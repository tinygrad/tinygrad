#!/bin/bash
set -e

APP_PATH="/Applications/TinyGPU.app"
DEXT_ID="org.tinygrad.tinygpu.edriver"

# Install app if not present. TODO: url
if [[ ! -d "$APP_PATH" ]]; then
  echo "TinyGPU.app not found in /Applications"
  exit 1
fi

# Check if dext is running
dext_status=$(systemextensionsctl list 2>/dev/null | grep "$DEXT_ID" || true)
if echo "$dext_status" | grep -q "\[activated enabled\]"; then
  echo "TinyGPU driver extension is already installed and active."
  exit 0
fi

# Ask user to install
echo "TinyGPU driver extension is not installed."
read -n1 -p "Install now? [y/N] " answer
echo

if [[ "$answer" =~ ^[Yy]$ ]]; then
  "$APP_PATH/Contents/MacOS/TinyGPU" install
else
  echo "Skipped."
  exit 0
fi
