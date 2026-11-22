#!/bin/bash
set -e
echo "options vfio-pci disable_idle_d3=1" | sudo tee /etc/modprobe.d/vfio.conf
echo "vfio-pci" | sudo tee /etc/modules-load.d/vfio.conf
echo "@sudo - memlock unlimited" | sudo tee /etc/security/limits.d/vfio.conf
sudo tee /etc/udev/rules.d/70-vfio.rules <<'EOF'
SUBSYSTEM=="pci", ATTR{vendor}=="0x1002", ATTR{class}=="0x03*", ATTR{reset_method}="bus"
SUBSYSTEM=="pci", ATTR{vendor}=="0x1002|0x10de", ATTR{class}=="0x03*|0x0403*", ENV{VFIO}="1"
SUBSYSTEM=="pci", ATTR{vendor}=="0x14e4", ATTR{device}=="0x1760", ENV{VFIO}="1"
SUBSYSTEM=="pci", ATTR{vendor}=="0x15b3", ATTR{class}=="0x02*", ENV{VFIO}="1"
ACTION=="add", ENV{VFIO}=="1", ATTR{driver_override}="vfio-pci", RUN+="/bin/sh -c 'echo $kernel > /sys$devpath/driver/unbind; echo $kernel > /sys/bus/pci/drivers_probe'"
SUBSYSTEM=="vfio", GROUP="sudo", MODE="0660"
EOF
