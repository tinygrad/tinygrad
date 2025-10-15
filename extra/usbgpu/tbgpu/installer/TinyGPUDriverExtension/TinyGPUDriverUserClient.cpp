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

kern_return_t TinyGPUDriverUserClient::ExternalMethod(uint64_t in_selector, IOUserClientMethodArguments* in_arguments, const IOUserClientMethodDispatch* in_dispatch, OSObject* in_target, void* in_reference)
{
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
