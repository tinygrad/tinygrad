#!/usr/bin/env bash
set -ex
mkdir -p out
cd out

BASE=/Users/taylor/fun/fpga

# yosys commit 82f5829aba108be4a3786e7a237fd7bcebe61eb6
# build normally
$BASE/yosys/yosys -p "synth_xilinx -flatten -nowidelut -abc9 -arch xc7 -top top; write_json attosoc.json" ../src/attosoc.v ../src/attosoc_top.v ../src/simpleuart.v

# nextpnr-xilinx 0be5cc19f3261101730ce9274720aaf3784f83e2
# cmake -DARCH=xilinx -DBUILD_GUI=no -DBUILD_PYTHON=no -DUSE_OPENMP=No .
# git submodule init && git submodule update
# python3 xilinx/python/bbaexport.py --device xc7a100tcsg324-1 --bba xilinx/xc7a100t.bba
# ./bbasm -l xilinx/xc7a100t.bba xilinx/xc7a100t.bin
$BASE/nextpnr-xilinx/nextpnr-xilinx --chipdb $BASE/nextpnr-xilinx/xilinx/xc7a100t.bin --xdc ../src/arty.xdc --json attosoc.json --write attosoc_routed.json --fasm attosoc.fasm

# if you want the GUI (still broken on osx)
# cmake -DARCH=xilinx -DUSE_OPENMP=No -DPYTHON_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())")  -DPYTHON_LIBRARY=$(python -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))") -DQt5_DIR=$(brew --prefix qt5)/lib/cmake/Qt5 .

XRAY_UTILS_DIR=$BASE/prjxray/utils
XRAY_TOOLS_DIR=$BASE/prjxray/build/tools
XRAY_DATABASE_DIR=$BASE/prjxray/database

"${XRAY_UTILS_DIR}/fasm2frames.py" --db-root "${XRAY_DATABASE_DIR}/artix7" --part xc7a100tcsg324-1 attosoc.fasm > attosoc.frames
"${XRAY_TOOLS_DIR}/xc7frames2bit" --part_file "${XRAY_DATABASE_DIR}/artix7/xc7a100tcsg324-1/part.yaml" --part_name xc7a100tcsg324-1 --frm_file attosoc.frames --output_file attosoc.bit

