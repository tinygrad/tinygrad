#include "TinyGPUDriver.h"
#include "TinyGPUDriverUserClient.h"
#include <DriverKit/IOLib.h>
#include <PCIDriverKit/PCIDriverKit.h>

struct TinyGPUDriver_IVars
{
	IOPCIDevice *pci = nullptr;
};

bool TinyGPUDriver::init()
{
	os_log(OS_LOG_DEFAULT, "tinygpu: init");

	auto answer = super::init();
	if (!answer) {
		return false;
	}

	ivars = new TinyGPUDriver_IVars();
	if (ivars == nullptr) {
		return false;
	}

	return true;
}

void TinyGPUDriver::free()
{
	IOSafeDeleteNULL(ivars, TinyGPUDriver_IVars, 1);
	super::free();
}

void TinyGPUDriver::ProgramBARAddresses()
{
	// On Apple Silicon, macOS may not assign BAR addresses for Thunderbolt eGPUs.
	// The BAR registers in PCI config space remain 0x00000000, causing
	// _CopyDeviceMemoryWithIndex to fail because it reads from the IORegistry
	// "assigned-addresses" property (set at boot), not live config space.
	//
	// This function:
	// 1. Detects unassigned BARs (BAR0 = 0x00000000)
	// 2. Determines each BAR's size requirement (write 0xFFFFFFFF, read back)
	// 3. Allocates BAR addresses from a fixed Thunderbolt aperture range
	// 4. Writes the addresses to BAR registers via ConfigurationWrite32
	// 5. Re-enables Memory Space and Bus Master
	//
	// NOTE: _CopyDeviceMemoryWithIndex() still won't work after this because it
	// reads from the IORegistry "assigned-addresses" property which DriverKit cannot
	// modify (SetProperty for OSData is not available in the DriverKit SDK).
	// The tinygrad runtime should use ConfigurationRead32/Write32 for register access,
	// and DMA for large data transfers.

	os_log(OS_LOG_DEFAULT, "tinygpu: checking BAR assignments");

	// Check if BAR0 is already assigned by the OS
	uint32_t bar0 = 0;
	ivars->pci->ConfigurationRead32(kIOPCIConfigurationOffsetBaseAddress0, &bar0);
	if (bar0 != 0 && bar0 != 0xFFFFFFFF) {
		os_log(OS_LOG_DEFAULT, "tinygpu: BAR0 already assigned (0x%08x), skipping BAR programming", bar0);
		return;
	}

	os_log(OS_LOG_DEFAULT, "tinygpu: BAR0 unassigned (0x%08x), programming BAR addresses for eGPU", bar0);

	// Disable memory and I/O access while programming BARs
	uint16_t cmd = 0;
	ivars->pci->ConfigurationRead16(kIOPCIConfigurationOffsetCommand, &cmd);
	os_log(OS_LOG_DEFAULT, "tinygpu: Command register before: 0x%04x", cmd);
	ivars->pci->ConfigurationWrite16(kIOPCIConfigurationOffsetCommand, cmd & ~(kIOPCICommandIOSpace | kIOPCICommandMemorySpace));

	// Step 1: Determine BAR sizes
	uint64_t bar_sizes[6] = {};
	bool bar_64bit[6] = {};
	bool bar_valid[6] = {};
	uint32_t bar_origins[12] = {};

	for (int i = 0; i < 6; ) {
		uint32_t offset = kIOPCIConfigurationOffsetBaseAddress0 + i * 4;

		ivars->pci->ConfigurationRead32(offset, &bar_origins[i]);

		ivars->pci->ConfigurationWrite32(offset, 0xFFFFFFFF);
		uint32_t readback = 0;
		ivars->pci->ConfigurationRead32(offset, &readback);

		ivars->pci->ConfigurationWrite32(offset, bar_origins[i]);

		if (readback == 0 || readback == 0xFFFFFFFF) {
			bar_sizes[i] = 0;
			bar_64bit[i] = false;
			bar_valid[i] = false;
			i++;
			continue;
		}

		uint8_t bar_type = readback & 0x7;
		bar_64bit[i] = (bar_type == 0x4);
		bar_valid[i] = true;

		uint32_t mask = readback & ~0xF;
		uint64_t size = ~(uint64_t)mask + 1;

		if (bar_64bit[i] && i + 1 < 6) {
			uint32_t offset_hi = kIOPCIConfigurationOffsetBaseAddress0 + (i + 1) * 4;
			ivars->pci->ConfigurationRead32(offset_hi, &bar_origins[i + 1]);

			ivars->pci->ConfigurationWrite32(offset_hi, 0xFFFFFFFF);
			uint32_t readback_hi = 0;
			ivars->pci->ConfigurationRead32(offset_hi, &readback_hi);

			ivars->pci->ConfigurationWrite32(offset_hi, bar_origins[i + 1]);

			uint64_t size_lo = ~(uint64_t)mask + 1;
			uint64_t size_hi = ~(uint64_t)readback_hi;
			bar_sizes[i] = size_lo | (size_hi << 32);
			if (bar_sizes[i] == 0) bar_sizes[i] = ((uint64_t)1 << 32);
			bar_sizes[i + 1] = 0;
			bar_valid[i + 1] = false;
			os_log(OS_LOG_DEFAULT, "tinygpu: BAR%d: size=0x%llx 64bit=1", i, bar_sizes[i]);
			i += 2;
		} else {
			bar_sizes[i] = size;
			os_log(OS_LOG_DEFAULT, "tinygpu: BAR%d: size=0x%llx 64bit=0", i, size);
			i++;
		}
	}

	// Step 2: Use fixed Thunderbolt aperture range
	// On Apple Silicon M4 with Thunderbolt 4, the TB aperture for memory-mapped
	// devices starts at a known address. We use a safe offset that won't conflict
	// with the audio device's BAR (which is at ~0x16xxxxxxx range).
	// Aperture: 0x400000000 (16GB) to 0x10000000000 (1TB)
	uint64_t aperture_base = 0x400000000ULL;
	uint64_t aperture_limit = 0x10000000000ULL;

	os_log(OS_LOG_DEFAULT, "tinygpu: using fixed TB aperture base=0x%llx limit=0x%llx", aperture_base, aperture_limit);

	// Step 3: Allocate and program BAR addresses
	uint64_t next_addr = aperture_base;
	int assigned_count = 0;

	for (int i = 0; i < 6; ) {
		if (!bar_valid[i] || bar_sizes[i] == 0) {
			if (bar_64bit[i]) { i += 2; } else { i++; }
			continue;
		}

		uint64_t alignment = bar_sizes[i];
		uint64_t addr = (next_addr + alignment - 1) & ~(alignment - 1);

		if (addr + bar_sizes[i] > aperture_limit) {
			os_log(OS_LOG_DEFAULT, "tinygpu: ERROR: not enough aperture for BAR%d (need 0x%llx at 0x%llx, limit 0x%llx)", i, bar_sizes[i], addr, aperture_limit);
			break;
		}

		uint32_t bar_offset = kIOPCIConfigurationOffsetBaseAddress0 + i * 4;

		if (bar_64bit[i]) {
			uint32_t bar_lo_val = (uint32_t)(addr & ~0xF) | 0x04; // Preserve 64-bit flag
			uint32_t bar_hi_val = (uint32_t)(addr >> 32);
			ivars->pci->ConfigurationWrite32(bar_offset, bar_lo_val);
			ivars->pci->ConfigurationWrite32(bar_offset + 4, bar_hi_val);
			os_log(OS_LOG_DEFAULT, "tinygpu: BAR%d (64-bit): programmed addr=0x%llx size=0x%llx lo=0x%08x hi=0x%08x", i, addr, bar_sizes[i], bar_lo_val, bar_hi_val);

			next_addr = addr + bar_sizes[i];
			assigned_count++;
			i += 2;
		} else {
			uint32_t bar_val = (uint32_t)(addr & ~0xF);
			ivars->pci->ConfigurationWrite32(bar_offset, bar_val);
			os_log(OS_LOG_DEFAULT, "tinygpu: BAR%d (32-bit): programmed addr=0x%llx size=0x%llx val=0x%08x", i, addr, bar_sizes[i], bar_val);

			next_addr = addr + bar_sizes[i];
			assigned_count++;
			i++;
		}
	}

	// Step 4: Re-enable Memory Space and Bus Master
	uint16_t new_cmd = cmd | (kIOPCICommandMemorySpace | kIOPCICommandBusMaster);
	ivars->pci->ConfigurationWrite16(kIOPCIConfigurationOffsetCommand, new_cmd);
	os_log(OS_LOG_DEFAULT, "tinygpu: Command register: 0x%04x -> 0x%04x (Memory Space + Bus Master enabled)", cmd, new_cmd);

	// Verify BAR programming
	os_log(OS_LOG_DEFAULT, "tinygpu: BAR verification after programming:");
	for (int i = 0; i < 6; i++) {
		uint32_t offset = kIOPCIConfigurationOffsetBaseAddress0 + i * 4;
		uint32_t val = 0;
		ivars->pci->ConfigurationRead32(offset, &val);
		if (val != 0) {
			os_log(OS_LOG_DEFAULT, "tinygpu:   BAR%d = 0x%08x", i, val);
		}
	}

	os_log(OS_LOG_DEFAULT, "tinygpu: BAR address programming complete (%d BARs assigned)", assigned_count);
}

kern_return_t TinyGPUDriver::Start_Impl(IOService* in_provider)
{
	IOServiceName service_name;
	os_log(OS_LOG_DEFAULT, "tinygpu: on gpu detected");

	kern_return_t err = Start(in_provider, SUPERDISPATCH);
	if (err) return err;

	ivars->pci = OSDynamicCast(IOPCIDevice, in_provider);
	if (!ivars->pci) return kIOReturnNoDevice;

	err = ivars->pci->Open(this, 0);
	if (err) {
		os_log(OS_LOG_DEFAULT, "tinygpu: Open() failed 0x%08x", err);
		ivars->pci = nullptr;
		return err;
	}

	uint16_t ven = 0, dev = 0;
	ivars->pci->ConfigurationRead16(kIOPCIConfigurationOffsetVendorID, &ven);
	ivars->pci->ConfigurationRead16(kIOPCIConfigurationOffsetDeviceID, &dev);
	os_log(OS_LOG_DEFAULT, "tinygpu: opened device ven=0x%04x dev=0x%04x", ven, dev);

	// Enable Memory Space, Bus Master, and I/O Space
	uint16_t commandRegister;
	ivars->pci->ConfigurationRead16(kIOPCIConfigurationOffsetCommand, &commandRegister);
	os_log(OS_LOG_DEFAULT, "tinygpu: initial Command register = 0x%04x", commandRegister);
	commandRegister |= (kIOPCICommandIOSpace | kIOPCICommandBusMaster | kIOPCICommandMemorySpace);
	ivars->pci->ConfigurationWrite16(kIOPCIConfigurationOffsetCommand, commandRegister);

	// Program BAR addresses for eGPUs where macOS doesn't assign them
	ProgramBARAddresses();

	memcpy((void*)service_name, (void*)"tinygpu\0", 8);
	SetName(service_name);

	os_log(OS_LOG_DEFAULT, "tinygpu: will register service %s", service_name);
	RegisterService();

	os_log(OS_LOG_DEFAULT, "tinygpu: service started %s", service_name);
	return 0;
}

kern_return_t TinyGPUDriver::Stop_Impl(IOService* in_provider)
{
	ivars->pci->Close(this, 0);
	return 0;
}

kern_return_t TinyGPUDriver::NewUserClient_Impl(uint32_t in_type, IOUserClient** out_user_client)
{
	kern_return_t err = 0;

	IOService* user_client_service = nullptr;
	err = Create(this, "TinyGPUDriverUserClientProperties", &user_client_service);
	if (err) {
		os_log(OS_LOG_DEFAULT, "tinygpu: failed to create NewUserClient");
		goto error;
	}
	*out_user_client = OSDynamicCast(IOUserClient, user_client_service);
	os_log(OS_LOG_DEFAULT, "tinygpu: NewUserClient created");

error:
	return err;
}

kern_return_t TinyGPUDriver::MapBar(uint32_t bar, IOMemoryDescriptor** memory)
{
	uint8_t barMemoryIndex, barMemoryType;
	uint64_t barMemorySize;
	kern_return_t err = ivars->pci->GetBARInfo(bar, &barMemoryIndex, &barMemorySize, &barMemoryType);
	if (err) {
		os_log(OS_LOG_DEFAULT, "tinygpu: GetBARInfo(%d) failed: 0x%08x", bar, err);
		return err;
	}
	os_log(OS_LOG_DEFAULT, "tinygpu: bar mapping %d idx=%d size=0x%llx type=%d", bar, barMemoryIndex, barMemorySize, barMemoryType);

	err = ivars->pci->_CopyDeviceMemoryWithIndex(barMemoryIndex, memory, this);
	if (err) {
		os_log(OS_LOG_DEFAULT, "tinygpu: _CopyDeviceMemoryWithIndex(%d) failed: 0x%08x - BAR addresses may not be in IORegistry", barMemoryIndex, err);
	}
	return err;
}

static kern_return_t WriteDMASegments(IOMemoryDescriptor* mem, IOAddressSegment* segments, uint32_t segCount,
                                      uint64_t mapOffset = 0, uint64_t mapSize = 0)
{
	IOMemoryMap* map = nullptr;
	kern_return_t err = mem->CreateMapping(0, 0, 0, mapOffset, mapSize, &map);
	if (err || !map) return err ?: kIOReturnError;

	uint64_t* out = (uint64_t*)map->GetAddress();
	for (uint32_t i = 0; i < segCount; i++) { out[i * 2] = segments[i].address; out[i * 2 + 1] = segments[i].length; }
	out[segCount * 2] = 0; out[segCount * 2 + 1] = 0;
	map->release();
	return 0;
}

kern_return_t TinyGPUDriver::SetupDMA(IOMemoryDescriptor* memory, uint64_t size, IODMACommand** outCmd,
                                       IOAddressSegment* segments, uint32_t* segCount)
{
	IODMACommandSpecification dmaSpec = {.options = 0, .maxAddressBits = 40};
	IODMACommand* dmaCmd = nullptr;

	kern_return_t err = IODMACommand::Create(ivars->pci, kIODMACommandCreateNoOptions, &dmaSpec, &dmaCmd);
	if (err) { os_log(OS_LOG_DEFAULT, "tinygpu: DMA create failed err=%d", err); return err; }

	uint64_t flags = kIOMemoryDirectionInOut;
	err = dmaCmd->PrepareForDMA(kIODMACommandPrepareForDMANoOptions, memory, 0, size, &flags, segCount, segments);
	if (err) { os_log(OS_LOG_DEFAULT, "tinygpu: PrepareForDMA failed err=%d", err); dmaCmd->release(); return err; }

	*outCmd = dmaCmd;
	return 0;
}

kern_return_t TinyGPUDriver::CreateDMA(size_t size, TinyGPUCreateDMAResp* dmaDesc)
{
	IOBufferMemoryDescriptor* sharedBuf = nullptr;
	kern_return_t err = IOBufferMemoryDescriptor::Create(kIOMemoryDirectionInOut, size, IOVMPageSize, &sharedBuf);
	if (err) { os_log(OS_LOG_DEFAULT, "tinygpu: alloc failed err=%d", err); return err; }

	IODMACommand* dmaCmd = nullptr;
	IOAddressSegment segments[32];
	uint32_t segCount = 32;
	err = SetupDMA(sharedBuf, size, &dmaCmd, segments, &segCount);
	if (err) { sharedBuf->release(); return err; }

	err = WriteDMASegments(sharedBuf, segments, segCount, IOVMPageSize, IOVMPageSize);
	if (err) { dmaCmd->CompleteDMA(kIODMACommandCompleteDMANoOptions); dmaCmd->release(); sharedBuf->release(); return err; }

	dmaDesc->sharedBuf = sharedBuf;
	dmaDesc->dmaCmd = dmaCmd;
	os_log(OS_LOG_DEFAULT, "tinygpu: CreateDMA size=0x%zx segs=%u", size, segCount);
	return 0;
}

kern_return_t TinyGPUDriver::CfgRead(uint32_t off, uint32_t size, uint32_t* outVal)
{
  if (!ivars->pci || !outVal) return kIOReturnNotReady;

  if (size == 1) {
	uint8_t v8 = 0;
	ivars->pci->ConfigurationRead8(off, &v8);
	*outVal = v8;
  } else if (size == 2) {
	uint16_t v16 = 0;
	ivars->pci->ConfigurationRead16(off, &v16);
	*outVal = v16;
  } else if (size == 4) {
	uint32_t v32 = 0;
	ivars->pci->ConfigurationRead32(off, &v32);
	*outVal = v32;
  }
  return 0;
}

kern_return_t TinyGPUDriver::CfgWrite(uint32_t off, uint32_t size, uint32_t val)
{
  if (!ivars->pci) return kIOReturnNotReady;
  if (size == 1) ivars->pci->ConfigurationWrite8 (off, (uint8_t)val);
  else if (size == 2) ivars->pci->ConfigurationWrite16(off, (uint16_t)val);
  else if (size == 4) ivars->pci->ConfigurationWrite32(off, (uint32_t)val);
  return 0;
}

kern_return_t TinyGPUDriver::ResetDevice()
{
	if (!ivars->pci) return kIOReturnNotReady;
	kern_return_t ret = ivars->pci->Reset(kIOPCIDeviceResetTypeFunctionReset);
	return ret == kIOReturnSuccess ? ret : ivars->pci->Reset(kIOPCIDeviceResetTypeHotReset);
}

IOPCIDevice* TinyGPUDriver::GetPCI()
{
	return ivars->pci;
}
