#!/bin/sh
python3 -c "from tinygrad.runtime.support.system import APLRemotePCIDevice; APLRemotePCIDevice.ensure_app()"
