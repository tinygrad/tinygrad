#!/bin/bash
# repro: trigger 100 GPU resets via debugfs (gets stuck around 10 resets)
# rocm 7.1.1
for i in {1..100}; do sudo cat /sys/kernel/debug/dri/5/amdgpu_gpu_recover; done
