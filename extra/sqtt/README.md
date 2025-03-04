# SQTT Profiling

## Getting SQ Thread Trace

Only supported on 7900XTX, requires either AM (`rmmod amdgpu`) or disabling power gating on AMD (`ppfeaturemask=0xffff3fff`, don't forget to rebuild initramfs)

SQTT is implemented on top of normal tinygrad PROFILE=1, `PROFILE=1 SQTT=1` to get profile pickle with sqtt data embedded in it.

`SQTT_ITRACE=0` to disable instruction tracing, this will reduce size of generated trace significantly \
`SQTT_BUFFER_SIZE=X` to change size of SQTT buffer (per shader engine, 6 SEs on 7900xtx) in megabytes, default 128.

## Converting pickled profile with SQTT data into RGP file

```bash
extra/sqtt/rgptool.py create "/tmp/profile.pkl.$USER" -o /tmp/gpu0.rgp
```

Then load gpu0.rgp into Radeon GPU Profiler. It works just fine both in wine (macos, native version available for linux) and via ssh X forwarding
