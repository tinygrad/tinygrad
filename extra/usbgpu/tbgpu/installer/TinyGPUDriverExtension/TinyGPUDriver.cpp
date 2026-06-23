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
	// _CopyDeviceMemoryWithIndex to return mappings that read as 0xFFFFFFFF.
	//
	// This function checks if BARs are unassigned and programs them by:
	// 1. Determining each BAR's size requirement (write 0xFFFFFFFF, read back)
	// 2. Reading the parent Thunderbolt bridge's memory aperture
	// 3. Allocating BAR addresses from the aperture
	// 4. Writing the addresses to BAR registers
	// 5. Re-enabling Memory Space

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
	ivars->pci->ConfigurationWrite16(kIOPCIConfigurationOffsetCommand, cmd & ~(kIOPCICommandIOSpace | kIOPCICommandMemorySpace));

	// Step 1: Determine BAR sizes
	uint64_t bar_sizes[6] = {};
	bool bar_64bit[6] = {};
	uint32_t bar_origins[6] = {};

	for (int i = 0; i < 6; ) {
		uint32_t offset = kIOPCIConfigurationOffsetBaseAddress0 + i * 4;

		// Save original
		ivars->pci->ConfigurationRead32(offset, &bar_origins[i]);

		// Write all 1s to determine size
		ivars->pci->ConfigurationWrite32(offset, 0xFFFFFFFF);
		uint32_t readback = 0;
		ivars->pci->ConfigurationRead32(offset, &readback);

		// Restore original
		ivars->pci->ConfigurationWrite32(offset, bar_origins[i]);

		if (readback == 0 || readback == 0xFFFFFFFF) {
			bar_sizes[i] = 0;
			bar_64bit[i] = false;
			i++;
			continue;
		}

		// Decode BAR type
		uint8_t bar_type = readback & 0x7;
		bar_64bit[i] = (bar_type == 0x4);

		// Compute size: mask off type/info bits, invert, add 1
		uint32_t mask = readback & ~0xF;
		uint64_t size = ~(uint64_t)mask + 1;

		if (bar_64bit[i] && i + 1 < 6) {
			// For 64-bit BARs, also probe upper 32 bits
			uint32_t offset_hi = kIOPCIConfigurationOffsetBaseAddress0 + (i + 1) * 4;
			uint32_t orig_hi = 0;
			ivars->pci->ConfigurationRead32(offset_hi, &orig_hi);

			ivars->pci->ConfigurationWrite32(offset_hi, 0xFFFFFFFF);
			uint32_t readback_hi = 0;
			ivars->pci->ConfigurationRead32(offset_hi, &readback_hi);

			ivars->pci->ConfigurationWrite32(offset_hi, orig_hi);

			bar_origins[i + 1] = orig_hi;
			// 64-bit size = lower 32 inverted + upper 32 inverted << 32
			uint64_t size_hi = (~(uint64_t)readback_hi) << 32;
			bar_sizes[i] = (size & 0xFFFFFFFF) | size_hi;
			if (bar_sizes[i] == 0) bar_sizes[i] = ((uint64_t)1 << 32);
			bar_sizes[i + 1] = 0; // Upper half is part of the same BAR
			os_log(OS_LOG_DEFAULT, "tinygpu: BAR%d: size=0x%llx 64bit=1", i, bar_sizes[i]);
			i += 2;
		} else {
			bar_sizes[i] = size;
			os_log(OS_LOG_DEFAULT, "tinygpu: BAR%d: size=0x%llx 64bit=0", i, size);
			i++;
		}
	}

	// Step 2: Read parent bridge memory aperture
	// Thunderbolt bridges expose memory windows at standard PCI bridge config offsets
	// Offset 0x20: Memory Base (16-bit, lower 28 bits of base address, 1MB aligned)
	// Offset 0x22: Memory Limit (16-bit, lower 28 bits of limit address)
	// Offset 0x24: Prefetchable Memory Base Lower (16-bit)
	// Offset 0x28: Prefetchable Memory Base Upper (32-bit)
	// Offset 0x2C: Prefetchable Memory Limit Upper (32-bit)

	OSObject* parentObj = nullptr;
	ivars->pci->CopyParent(&parentObj);
	IOPCIDevice* bridge = OSDynamicCast(IOPCIDevice, parentObj);

	uint64_t mem_base = 0, mem_limit = 0;
	if (bridge) {
		bridge->Open(this, 0);
		uint16_t mem_base_reg = 0, mem_limit_reg = 0;
		bridge->ConfigurationRead16(0x20, &mem_base_reg);
		bridge->ConfigurationRead16(0x22, &mem_limit_reg);

		// Decode: Memory Base/Limit are in units of 1MB, aligned to 1MB boundaries
		mem_base = ((uint64_t)(mem_base_reg & 0xFFF0) << 16);
		mem_limit = (((uint64_t)(mem_limit_reg & 0xFFF0) << 16) | 0xFFFFF);

		os_log(OS_LOG_DEFAULT, "tinygpu: bridge mem window 0x%llx - 0x%llx (base_reg=0x%04x limit_reg=0x%04x)", mem_base, mem_limit, mem_base_reg, mem_limit_reg);

		// Also check prefetchable memory window for 64-bit BARs
		uint16_t pref_base_low = 0, pref_limit_low = 0;
		uint32_t pref_base_hi = 0, pref_limit_hi = 0;
		bridge->ConfigurationRead16(0x24, &pref_base_low);
		bridge->ConfigurationRead16(0x26, &pref_limit_low);
		bridge->ConfigurationRead32(0x28, &pref_base_hi);
		bridge->ConfigurationRead32(0x2C, &pref_limit_hi);

		uint64_t pref_base = ((uint64_t)(pref_base_low & 0xFFF0) << 16) | ((uint64_t)pref_base_hi << 32);
		uint64_t pref_limit = (((uint64_t)(pref_limit_low & 0xFFF0) << 16) | 0xFFFFF) | ((uint64_t)pref_limit_hi << 32);

		os_log(OS_LOG_DEFAULT, "tinygpu: bridge pref mem window 0x%llx - 0x%llx", pref_base, pref_limit);

		// Use prefetchable window for 64-bit BARs if available
		if (pref_base != 0 && pref_limit > pref_base) {
			// Use non-prefetchable for 32-bit BARs, prefetchable for 64-bit BARs
			// For now, we allocate everything from the larger window
			if (pref_limit - pref_base > mem_limit - mem_base) {
				mem_base = pref_base;
				mem_limit = pref_limit;
				os_log(OS_LOG_DEFAULT, "tinygpu: using pref mem window for BAR allocation (larger)");
			}
		}

		bridge->Close(this, 0);
	} else {
		os_log(OS_LOG_DEFAULT, "tinygpu: WARNING: parent is not IOPCIDevice, cannot determine bridge aperture");
	}

	if (parentObj) parentObj->release();

	// Step 3: Allocate and program BAR addresses
	if (mem_base == 0 && mem_limit == 0) {
		os_log(OS_LOG_DEFAULT, "tinygpu: WARNING: bridge memory window not available, cannot program BARs");
		ivars->pci->ConfigurationWrite16(kIOPCIConfigurationOffsetCommand, cmd);
		return;
	}

	uint64_t next_addr = mem_base;
	for (int i = 0; i < 6; ) {
		if (bar_sizes[i] == 0) { i++; continue; }

		// Align to BAR size (PCI BARs must be naturally aligned)
		uint64_t alignment = bar_sizes[i];
		uint64_t addr = (next_addr + alignment - 1) & ~(alignment - 1);

		if (addr + bar_sizes[i] > mem_limit) {
			os_log(OS_LOG_DEFAULT, "tinygpu: ERROR: not enough space for BAR%d (need 0x%llx at 0x%llx, limit 0x%llx)", i, bar_sizes[i], addr, mem_limit);
			break;
		}

		uint32_t offset = kIOPCIConfigurationOffsetBaseAddress0 + i * 4;

		if (bar_64bit[i]) {
			uint32_t bar_lo = (uint32_t)(addr & ~0xF) | 0x04; // 64-bit, prefetchable bit preserved
			uint32_t bar_hi = (uint32_t)(addr >> 32);
			ivars->pci->ConfigurationWrite32(offset, bar_lo);
			ivars->pci->ConfigurationWrite32(offset + 4, bar_hi);
			os_log(OS_LOG_DEFAULT, "tinygpu: BAR%d (64-bit): addr=0x%llx size=0x%llx lo=0x%08x hi=0x%08x", i, addr, bar_sizes[i], bar_lo, bar_hi);
			next_addr = addr + bar_sizes[i];
			i += 2;
		} else {
			uint32_t bar_val = (uint32_t)(addr & ~0xF) | 0x00; // 32-bit, non-prefetchable
			ivars->pci->ConfigurationWrite32(offset, bar_val);
			os_log(OS_LOG_DEFAULT, "tinygpu: BAR%d (32-bit): addr=0x%llx size=0x%llx val=0x%08x", i, addr, bar_sizes[i], bar_val);
			next_addr = addr + bar_sizes[i];
			i++;
		}
	}

	// Step 4: Re-enable Memory Space and Bus Master
	ivars->pci->ConfigurationWrite16(kIOPCIConfigurationOffsetCommand, cmd);

	// Verify BAR programming
	os_log(OS_LOG_DEFAULT, "tinygpu: BAR verification:");
	for (int i = 0; i < 6; i++) {
		uint32_t offset = kIOPCIConfigurationOffsetBaseAddress0 + i * 4;
		uint32_t val = 0;
		ivars->pci->ConfigurationRead32(offset, &val);
		if (val != 0) {
			os_log(OS_LOG_DEFAULT, "tinygpu:   BAR%d = 0x%08x", i, val);
		}
	}

	os_log(OS_LOG_DEFAULT, "tinygpu: BAR address programming complete");
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

	uint16_t commandRegister;
	ivars->pci->ConfigurationRead16(kIOPCIConfigurationOffsetCommand, &commandRegister);
	commandRegister |= (kIOPCICommandIOSpace | kIOPCICommandBusMaster | kIOPCICommandMemorySpace);
	ivars->pci->ConfigurationWrite16(kIOPCIConfigurationOffsetCommand, commandRegister);

	// Program BAR addresses for eGPUs where macOS doesn't assign them.
	// On Apple Silicon Thunderbolt, BAR registers may remain 0x00000000,
	// causing _CopyDeviceMemoryWithIndex to fail silently.
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
	if (err) return err;
	os_log(OS_LOG_DEFAULT, "tinygpu: bar mapping %d idx=%d", bar, barMemoryIndex);
	return ivars->pci->_CopyDeviceMemoryWithIndex(barMemoryIndex, memory, this);
}

static kern_return_t WriteDMASegments(IOMemoryDescriptor* mem, IOAddressSegment* segments, uint32_t segCount,
                                      uint64_t mapOffset = 0, uint64_t mapSize = 0)
{
	// write dma segments to mapped memory as [addr0, len0, addr1, len1, ..., 0, 0]

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