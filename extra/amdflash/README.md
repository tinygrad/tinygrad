# Navi31 flash tools

Utilities for reading and recovering the 2 MiB SPI flash on Navi31 boards.
Run them from the tinygrad repository root. No image is bundled; keep a verified
full-ROM backup before performing any write.

`fw_live.py` accesses BAR5 through tinygrad's `PCIDevice.map_bar()` abstraction
and supports either the custom ASM24 USB-PCIe bridge or native PCIe. Select the
transport before the subcommand:

```sh
python3 extra/amdflash/fw_live.py --transport usb probe
python3 extra/amdflash/fw_live.py --transport pci probe
```

The default, `--transport auto`, considers USB devices first and then native
PCI devices. Native PCI access requires the usual tinygrad PCI permissions and
an unbound kernel driver.

## Access paths and hardware state

The paths are state-dependent and are not interchangeable:

* **`romless.py`** drives SMUIO `ROM_SW_*` directly through the ASM24 bridge.
  Use it only when an empty or corrupt flash has stalled the PSP PBL. Healthy
  autonomous boot gates this engine; the usual gated status is
  `ROM_SW_STATUS=0x04000800`.
* **`fw_live.py probe`, `ifwi-step`, and `ifwi-all`** use the early PSP
  boot-firmware mailbox. They must run after autonomous PSP boot but before a
  host driver or `AMDev` initializes SOS.
* **`fw_live.py live-flash`** boots `AMDev`, stages an image in trained VRAM,
  and invokes the Linux PSP v13 live-update command sequence.
* **`fw_live.py dump`** reads an exact 2 MiB raw image through
  `ROM_INDEX/ROM_DATA`. It refuses devices where the raw SMUIO controller is
  unavailable; the NBIO SOC15 function-ROM aperture is not a physical SPI
  mapping and is deliberately not used as a fallback.

The tools do not reset or power-cycle the board.

## Raw ROM_SW recovery

Identification and read-only operations:

```sh
python3 extra/amdflash/romless.py info
python3 extra/amdflash/romless.py read 0 0x40
python3 extra/amdflash/romless.py dump spi.bin
python3 extra/amdflash/romless.py verify known-good.bin
```

Restore an exact 2 MiB image:

```sh
python3 extra/amdflash/romless.py flash known-good.bin --yes
```

If GD25 status-register bit `SR2.CMP` protects the complete array, clearing it
requires separate authorization:

```sh
python3 extra/amdflash/romless.py flash known-good.bin --clear-cmp --yes
```

Programming is sector-granular. Every written 4 KiB sector is immediately read
back and compared with the input. A range can be resumed independently:

```sh
python3 extra/amdflash/romless.py flash known-good.bin \
  --start-sector 128 --sector-count 64 --yes
```

Navi31 ROM_SW details used by the implementation:

* `ROM_SW_COMMAND = (address << 8) | opcode`
* TX data uses big-endian stream dwords
* `RETURN_DATA_EN` (bit 19) is clear for TX and set for RX
* the RX window exposes the preceding transaction, so reads are primed once

## Firmware-mediated access

Query the early mailbox without changing flash:

```sh
python3 extra/amdflash/fw_live.py probe
```

Stream one exact item only when its type matches the firmware request:

```sh
python3 extra/amdflash/fw_live.py stream 0x37 vbios-item.bin --yes
```

For a complete 2 MiB IFWI, either perform one requested step or follow requests
until firmware reports a `0x2xx` completion state:

```sh
python3 extra/amdflash/fw_live.py ifwi-step full-ifwi.bin --yes
python3 extra/amdflash/fw_live.py ifwi-all full-ifwi.bin --yes
```

A completion state requires a hard reset; it is not another item request. The
resolver supports recovery metadata types `0x01`-`0x08` and `0x80`-`0x89`,
including the firmware-selected inactive partition. This path is signature
enforcing and intentionally refuses mismatched item types.

The early protocol is:

* `START_TRANSFER`: `(size << 8) | item_type`
* `DATA_TRANSFER`: one little-endian image dword per mailbox command
* `END_TRANSFER`: `(bytes_sent << 8) | item_type`

Each dword requires a firmware acknowledgement, so large partition transfers
are slow through a USB-PCIe bridge.

The fully initialized PSP path and the healthy-state dump are:

```sh
python3 extra/amdflash/fw_live.py live-flash signed-update.bin --yes
python3 extra/amdflash/fw_live.py dump current-spi.bin
```

The PSP validates live-update inputs and may reject an image even when its size
and alignment are valid. `dump` produces exactly `0x200000` bytes, requires the
raw IFWI magic at offset zero, rejects mirrored 1 MiB apertures, and restores
the ROM controller/index state before writing output.

## Safety

All erase, program, and firmware-streaming commands require `--yes`. Read-only
commands still touch controller and mailbox registers but do not issue SPI
program/erase or PSP transfer-start commands. Preserve a known-good full dump
outside the repository.
