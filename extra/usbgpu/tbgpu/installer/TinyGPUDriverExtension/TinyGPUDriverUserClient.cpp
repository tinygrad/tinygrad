#include "TinyGPUDriverUserClient.h"
#include "TinyGPUDriver.h"
#include <DriverKit/DriverKit.h>
#include <DriverKit/OSSharedPtr.h>
#include <PCIDriverKit/PCIDriverKit.h>

struct TinyGPUDriverUserClient_IVars
{
	OSSharedPtr<TinyGPUDriver> provider = nullptr;
};

bool TinyGPUDriverUserClient::init()
{
	auto theAnswer = super::init();
	if (!theAnswer) {
		return false;
	}

	ivars = IONewZero(TinyGPUDriverUserClient_IVars, 1);
	if (ivars == nullptr) {
		return false;
	}

	return true;
}

void TinyGPUDriverUserClient::free()
{
	if (ivars != nullptr) {
		ivars->provider.reset();
	}

	IOSafeDeleteNULL(ivars, TinyGPUDriverUserClient_IVars, 1);
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
	}

	return kIOReturnUnsupported;
}

kern_return_t IMPL(TinyGPUDriverUserClient, CopyClientMemoryForType)
{
	if (!memory) {
		return kIOReturnBadArgument;
	}

	if (ivars->provider.get() == nullptr) {
		return kIOReturnNotAttached;
	}

	if (type < 6) {
		uint32_t bar = (uint32_t)type;
		return ivars->provider->MapBar(bar, memory);
	}

	// dma page buffer
	TinyGPUCreateDMAResp buf;
	kern_return_t err = ivars->provider->CreateDMA(type, &buf);
	if (err) {
		return err;
	}

	*memory = buf.sharedBuf;
	return 0;
}
