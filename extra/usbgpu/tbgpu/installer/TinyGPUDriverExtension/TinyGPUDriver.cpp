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

	memcpy((void*)service_name, (void*)"tinygpu\0", 8);
	SetName(service_name);

	os_log(OS_LOG_DEFAULT, "tinygpu: will register service %s", service_name);
	RegisterService();

	os_log(OS_LOG_DEFAULT, "tinygpu: service started %s", service_name);
	return 0;
}

kern_return_t TinyGPUDriver::Stop_Impl(IOService* in_provider)
{
	// x1476 fork: hand the stop to the superclass. Upstream returned 0 here without
	// SUPERDISPATCH, so DriverKit never finished terminating the service and every
	// re-enumeration of the GPU left one more dext process behind (25 observed).
	os_log(OS_LOG_DEFAULT, "tinygpu: stop");
	if (ivars->pci) {
		ivars->pci->Close(this, 0);
		ivars->pci = nullptr;
	}
	return Stop(in_provider, SUPERDISPATCH);
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

// NVIDIA Blackwell (GB20x): the RTX 50 series never returns from a Function-Level Reset (a
// documented firmware defect; Linux carries quirk_no_flr for 10de:2b85/2b87/2b8c). A hot
// reset — secondary bus reset from the upstream bridge, what Linux falls back to — is the
// reset that works for these parts.
static bool IsNvidiaBlackwell(uint32_t vendorDevice)
{
	if ((vendorDevice & 0xffff) != 0x10de) return false;
	uint32_t family = (vendorDevice >> 16) & 0xff00;
	return family == 0x2b00 || family == 0x2c00 || family == 0x2d00 || family == 0x2f00;
}

static const uint32_t kTinyGPUBarCount = 6;
static const uint32_t kTinyGPUResetPollMs = 10;
static const uint32_t kTinyGPUResetDefaultTimeoutMs = 30000;

kern_return_t TinyGPUDriver::ResetDeviceWait(uint32_t type, uint32_t options, uint32_t timeoutMs, uint64_t* status)
{
	if (!status) return kIOReturnBadArgument;
	*status = kTinyGPUResetFailed;
	if (!ivars->pci) return kIOReturnNotReady;
	if (timeoutMs == 0) timeoutMs = kTinyGPUResetDefaultTimeoutMs;

	uint32_t vendorDevice = 0;
	ivars->pci->ConfigurationRead32(kIOPCIConfigurationOffsetVendorID, &vendorDevice);
	if (type == 0) type = IsNvidiaBlackwell(vendorDevice) ? kIOPCIDeviceResetTypeHotReset : kIOPCIDeviceResetTypeFunctionReset;

	// IOPCIFamily saves and restores configuration space around the reset itself, but its
	// post-reset readiness wait is short (100 ms after an FLR, 1 s after a hot reset). A GB202
	// takes longer, so keep our own copy of the BARs and the command register and re-apply
	// them once the device really answers again.
	uint32_t bars[kTinyGPUBarCount];
	for (uint32_t i = 0; i < kTinyGPUBarCount; i++) ivars->pci->ConfigurationRead32(kIOPCIConfigurationOffsetBaseAddress0 + 4 * i, &bars[i]);
	uint16_t command = 0;
	ivars->pci->ConfigurationRead16(kIOPCIConfigurationOffsetCommand, &command);

	os_log(OS_LOG_DEFAULT, "tinygpu: reset-wait type=0x%x options=0x%x device=%08x timeout=%ums", type, options, vendorDevice, timeoutMs);
	kern_return_t ret = ivars->pci->Reset(type, options);
	os_log(OS_LOG_DEFAULT, "tinygpu: reset-wait Reset() -> 0x%08x", ret);
	if (ret != kIOReturnSuccess) {
		*status = kTinyGPUResetFailed | ((uint64_t)type << 8) | ((uint64_t)(uint32_t)ret << 40);
		return ret;
	}
	if (options & kIOPCIDeviceResetOptionTerminate) {
		// The nub is being terminated and re-probed; this service is going away with it.
		*status = kTinyGPUResetTerminated | ((uint64_t)type << 8);
		return kIOReturnSuccess;
	}

	uint32_t waited = 0, now = 0;
	for (;;) {
		now = 0;
		ivars->pci->ConfigurationRead32(kIOPCIConfigurationOffsetVendorID, &now);
		if (now != 0 && now != 0xffffffffu && now != 0xffff0001u) break;  // 0xffff0001: Configuration Request Retry Status
		if (waited >= timeoutMs) {
			os_log(OS_LOG_DEFAULT, "tinygpu: reset-wait timeout after %ums (last read %08x)", waited, now);
			*status = kTinyGPUResetTimeout | ((uint64_t)type << 8) | ((uint64_t)waited << 16);
			return kIOReturnTimeout;
		}
		IOSleep(kTinyGPUResetPollMs);
		waited += kTinyGPUResetPollMs;
	}
	if (now != vendorDevice) {
		os_log(OS_LOG_DEFAULT, "tinygpu: reset-wait device changed %08x -> %08x", vendorDevice, now);
		*status = kTinyGPUResetVendorMismatch | ((uint64_t)type << 8) | ((uint64_t)waited << 16);
		return kIOReturnNoDevice;
	}

	uint64_t flags = 0;
	uint32_t bar0 = 0;
	ivars->pci->ConfigurationRead32(kIOPCIConfigurationOffsetBaseAddress0, &bar0);
	if ((bar0 & ~0xfu) == 0 && (bars[0] & ~0xfu) != 0) {
		for (uint32_t i = 0; i < kTinyGPUBarCount; i++) ivars->pci->ConfigurationWrite32(kIOPCIConfigurationOffsetBaseAddress0 + 4 * i, bars[i]);
		flags |= TINYGPU_RESET_FLAG_BARS_RESTORED;
		os_log(OS_LOG_DEFAULT, "tinygpu: reset-wait BARs were cleared by the reset; rewrote the saved values");
	}
	uint16_t commandNow = 0;
	ivars->pci->ConfigurationRead16(kIOPCIConfigurationOffsetCommand, &commandNow);
	commandNow |= (command & (kIOPCICommandIOSpace | kIOPCICommandBusMaster | kIOPCICommandMemorySpace));
	commandNow |= (kIOPCICommandIOSpace | kIOPCICommandBusMaster | kIOPCICommandMemorySpace);
	ivars->pci->ConfigurationWrite16(kIOPCIConfigurationOffsetCommand, commandNow);

	os_log(OS_LOG_DEFAULT, "tinygpu: reset-wait ready after %ums (bars %s)", waited, (flags & TINYGPU_RESET_FLAG_BARS_RESTORED) ? "restored" : "kept");
	*status = kTinyGPUResetReady | ((uint64_t)type << 8) | ((uint64_t)waited << 16) | flags;
	return kIOReturnSuccess;
}

IOPCIDevice* TinyGPUDriver::GetPCI()
{
	return ivars->pci;
}
