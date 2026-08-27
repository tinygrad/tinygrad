#include "TinyGPUDriverUserClient.h"
#include "TinyGPUDriver.h"
#include <DriverKit/DriverKit.h>
#include <DriverKit/OSSharedPtr.h>
#include <PCIDriverKit/PCIDriverKit.h>

struct TinyGPUDriverUserClient_IVars
{
	OSSharedPtr<TinyGPUDriver> provider = nullptr;

	TinyGPUCreateDMAResp *dmas = nullptr;
	size_t dmaCount = 0;
	size_t dmaCap = 0;

	int ensureDMACap(size_t need)
	{
		// not thread-safe
		if (need <= dmaCap) return 0;

		size_t newCap = dmaCap ? dmaCap * 2 : 16;
		while (newCap < need) newCap *= 2;

		auto *newArr = IONewZero(TinyGPUCreateDMAResp, newCap);
		if (!newArr) return -kIOReturnNoMemory;

		if (dmas && dmaCount) {
			memcpy(newArr, dmas, dmaCount * sizeof(TinyGPUCreateDMAResp));
		}

		IOSafeDeleteNULL(dmas, TinyGPUCreateDMAResp, dmaCap);
		dmas = newArr;
		dmaCap = newCap;
		return 0;
	}
};

bool TinyGPUDriverUserClient::init()
{
	auto ok = super::init();
	if (!ok) return false;

	ivars = IONewZero(TinyGPUDriverUserClient_IVars, 1);
	if (!ivars) return false;
	return true;
}

void TinyGPUDriverUserClient::free()
{
	if (ivars) {
		IOSafeDeleteNULL(ivars, TinyGPUDriverUserClient_IVars, 1);
	}
	super::free();
}

kern_return_t TinyGPUDriverUserClient::Start_Impl(IOService* in_provider)
{
	kern_return_t err = kIOReturnSuccess;
	if (!in_provider) {
		os_log(OS_LOG_DEFAULT, "tinygpu: provider is null");
		err = kIOReturnBadArgument;
		goto error;
	}

	err = Start(in_provider, SUPERDISPATCH);
	if (err) {
		os_log(OS_LOG_DEFAULT, "tinygpu: failed to start super (%d)", err);
		goto error;
	}

	ivars->provider = OSSharedPtr(OSDynamicCast(TinyGPUDriver, in_provider), OSRetain);
	return 0;

error:
	ivars->provider.reset();
	return err;
}

kern_return_t TinyGPUDriverUserClient::Stop_Impl(IOService* in_provider)
{
	// release all DMA allocations for this client
	if (ivars) {
		for (size_t i = 0; i < ivars->dmaCount; i++) {
			auto &d = ivars->dmas[i];
			if (d.dmaCmd) {
				d.dmaCmd->CompleteDMA(kIODMACommandCompleteDMANoOptions);
				d.dmaCmd->release();
				d.dmaCmd = nullptr;
			}
			if (d.sharedBuf) {
				d.sharedBuf->release();
				d.sharedBuf = nullptr;
			}
		}
		ivars->dmaCount = 0;
		IOSafeDeleteNULL(ivars->dmas, TinyGPUCreateDMAResp, ivars->dmaCap);
		ivars->dmas = nullptr;
		ivars->provider.reset();
	}

	return Stop(in_provider, SUPERDISPATCH);
}

kern_return_t TinyGPUDriverUserClient::ExternalMethod(uint64_t selector, IOUserClientMethodArguments* args, const IOUserClientMethodDispatch* in_dispatch, OSObject* in_target, void* in_reference)
{
	kern_return_t err = 0;

	os_log(OS_LOG_DEFAULT, "tinygpu: rpc (%llu) in:%d, out:%d", selector, args->scalarInputCount, args->scalarOutputCount);

	if (selector == TinyGPURPC::ReadCfg) {
		if (args->scalarInputCount != 2 or args->scalarOutputCount < 1) return kIOReturnBadArgument;

		uint32_t off = uint32_t(args->scalarInput[0]);
		uint32_t size = uint32_t(args->scalarInput[1]);

		uint32_t val = 0;
		err = ivars->provider->CfgRead(off, size, &val);
		os_log(OS_LOG_DEFAULT, "tinygpu: read cfg off:%x sz:%d, val:%x", off, size, val);

		if (!err) {
			args->scalarOutput[0] = val;
			args->scalarOutputCount = 1;
		}
		return err;
	} else if (selector == TinyGPURPC::WriteCfg) {
		if (args->scalarInputCount != 3) return kIOReturnBadArgument;

		uint32_t off = uint32_t(args->scalarInput[0]);
		uint32_t size = uint32_t(args->scalarInput[1]);
		uint32_t val = uint32_t(args->scalarInput[2]);

		os_log(OS_LOG_DEFAULT, "tinygpu: wr cfg off:%x sz:%d, val:%x", off, size, val);
		return ivars->provider->CfgWrite(off, size, val);
	} else if (selector == TinyGPURPC::Reset) {
		os_log(OS_LOG_DEFAULT, "tinygpu: reset");
		return ivars->provider->ResetDevice();
	} else if (selector == TinyGPURPC::PrepareDMA) {
		// The DMA buffer is the caller's (out-of-line, >= 4097 bytes) OUTPUT struct; the inband INPUT struct carries flags.
		//
		// The kernel wraps an ool input struct in a kIODirectionOut descriptor and an ool output struct in a kIODirectionIn one. A
		// kIODirectionOut descriptor is prepared read-only for DMA, and IOMMUs that honour the access bits (Intel AppleVTD) silently
		// drop every device write into it, which is how GSP-RM ended up unable to post its RPC queue on an Intel Mac Pro. Wrapping the
		// output descriptor in an InOut multi-descriptor makes IODMACommand map it read+write (IOMemoryDescriptor::dmaMap).
		if (!args->structureOutputDescriptor) {
			os_log(OS_LOG_DEFAULT, "tinygpu: PrepareDMA requires an ool output buffer >= 4097 bytes");
			return kIOReturnBadArgument;
		}
		if (ivars->ensureDMACap(ivars->dmaCount + 1)) return kIOReturnNoMemory;

		uint64_t size = 0;
		args->structureOutputDescriptor->GetLength(&size);

		IOMemoryDescriptor* dmaMem = nullptr;
		IOMemoryDescriptor* sources[1] = { args->structureOutputDescriptor };
		err = IOMemoryDescriptor::CreateWithMemoryDescriptors(kIOMemoryDirectionInOut, 1, sources, &dmaMem);
		if (err || !dmaMem) { os_log(OS_LOG_DEFAULT, "tinygpu: PrepareDMA wrapper failed err=%d", err); return err ?: kIOReturnError; }

		IODMACommand* dmaCmd = nullptr;
		IOAddressSegment segments[32];
		uint32_t segCount = 32;
		uint64_t dmaFlags = 0;
		err = ivars->provider->SetupDMA(dmaMem, size, &dmaCmd, segments, &segCount, &dmaFlags);
		if (err) { dmaMem->release(); return err; }
		if ((dmaFlags & kIOMemoryDirectionInOut) != kIOMemoryDirectionInOut) {
			os_log(OS_LOG_DEFAULT, "tinygpu: PrepareDMA mapping is not read+write (flags=0x%llx), refusing", dmaFlags);
			dmaCmd->CompleteDMA(kIODMACommandCompleteDMANoOptions); dmaCmd->release(); dmaMem->release();
			return kIOReturnNotWritable;
		}

		// The [addr, len, ..., 0, 0] table goes to the head of the buffer itself (the app reads it from the shared mapping).
		IOMemoryMap* outMap = nullptr;
		err = args->structureOutputDescriptor->CreateMapping(0, 0, 0, 0, 0, &outMap);
		if (err || !outMap) {
			os_log(OS_LOG_DEFAULT, "tinygpu: output map failed err=%d", err);
			dmaCmd->CompleteDMA(kIODMACommandCompleteDMANoOptions); dmaCmd->release(); dmaMem->release();
			return err ?: kIOReturnError;
		}
		uint64_t* out = (uint64_t*)outMap->GetAddress();
		for (uint32_t i = 0; i < segCount; i++) { out[i * 2] = segments[i].address; out[i * 2 + 1] = segments[i].length; }
		out[segCount * 2] = 0; out[segCount * 2 + 1] = 0;
		outMap->release();

		os_log(OS_LOG_DEFAULT, "tinygpu: PrepareDMA size=%llu segs=%u flags=0x%llx", size, segCount, dmaFlags);
		ivars->dmas[ivars->dmaCount++] = {dmaMem, dmaCmd}; // keep the wrapper alive as long as the DMA command that references it
		return kIOReturnSuccess;
	}

	return kIOReturnUnsupported;
}

kern_return_t IMPL(TinyGPUDriverUserClient, CopyClientMemoryForType)
{
	if (!memory) return kIOReturnBadArgument;
	if (!ivars->provider.get()) return kIOReturnNotAttached;

	// bar handling, type is bar num
	if (type < 6) {
		uint32_t bar = (uint32_t)type;
		return ivars->provider->MapBar(bar, memory);
	}

	// dma handling, type is size
	if (ivars->ensureDMACap(ivars->dmaCount + 1)) {
		os_log(OS_LOG_DEFAULT, "tinygpu: cannot grow dma array");
		return kIOReturnNoMemory;
	}

	TinyGPUCreateDMAResp buf{};
	kern_return_t err = ivars->provider->CreateDMA(type, &buf);
	if (err) return err;

	ivars->dmas[ivars->dmaCount++] = buf;
	*memory = buf.sharedBuf;
	return 0;
}
