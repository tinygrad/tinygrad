GPU=0000:01:00.0              # your PCI address
echo 1 | sudo tee /sys/bus/pci/devices/$GPU/reset 2>/dev/null || {
    # FLR not supported – fall back to remove/rescan
    echo 1 | sudo tee /sys/bus/pci/devices/$GPU/remove
    sleep 1
    echo 1 | sudo tee /sys/bus/pci/rescan
}
