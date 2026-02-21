#!/bin/bash
# Quick setup + test script for CDNA4 FA emulator on Ubuntu x86_64
# Usage: ssh root@some-vps 'bash -s' < scripts/test_cdna4_fa.sh
# Or: rsync -av . root@vps:~/tinygrad-cdna4/ && ssh root@vps 'cd ~/tinygrad-cdna4 && bash scripts/test_cdna4_fa.sh'
set -euo pipefail

echo "=== System info ==="
uname -m
cat /etc/os-release 2>/dev/null | head -2

# Install ROCm comgr if not present
if [ ! -f /opt/rocm/lib/libamd_comgr.so ]; then
  echo "=== Installing ROCm comgr ==="
  apt-get update -qq
  apt-get install -y -qq wget gnupg2
  # Add ROCm repo
  mkdir -p /etc/apt/keyrings
  wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | gpg --dearmor -o /etc/apt/keyrings/rocm.gpg
  echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/6.4 noble main" > /etc/apt/sources.list.d/rocm.list
  apt-get update -qq
  apt-get install -y -qq amd-comgr-dev clang || {
    echo "ROCm repo failed, trying direct package..."
    # Fallback: install from ROCm 6.3 (Ubuntu 24.04)
    apt-get install -y -qq amd-comgr clang
  }
fi

echo "=== ROCm check ==="
ls -la /opt/rocm/lib/libamd_comgr.so* 2>/dev/null || echo "comgr NOT found"
clang --version 2>/dev/null | head -1 || echo "clang NOT found"

# Install Python deps
echo "=== Python setup ==="
python3 --version
pip3 install -q numpy 2>/dev/null || apt-get install -y -qq python3-numpy

# Run tests
echo ""
echo "=== Test 1: FA compile-only (should already work) ==="
NULL_ALLOW_COPYOUT=1 PYTHONPATH=. DEV=NULL EMULATE=AMD_CDNA4 python3 test/testextra/test_tk.py 2>&1 | tail -5

echo ""
echo "=== Test 2: FA emulator (the main test) ==="
AMD=1 MOCKGPU=1 MOCKGPU_ARCH=cdna4 PYTHONPATH=. python3 -m pytest test/testextra/test_tk.py -x --tb=short -v 2>&1 | tail -40

echo ""
echo "=== Test 3: Core tests (sanity check) ==="
AMD=1 MOCKGPU=1 MOCKGPU_ARCH=cdna4 SKIP_SLOW_TEST=1 AMD_LLVM=0 \
  python3 -m pytest -n=auto test/backend/test_ops.py test/backend/test_dtype.py test/backend/test_dtype_alu.py --tb=line -q 2>&1 | tail -10

echo ""
echo "=== Done ==="
