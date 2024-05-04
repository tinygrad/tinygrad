#!/bin/bash

rocm-smi --setprofile compute
rocm-smi --setmclk 3
rocm-smi --setperflevel high
