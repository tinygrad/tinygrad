#include "TinyGPUDriverUserClient.h"
#include "TinyGPUDriver.h"
#include <DriverKit/DriverKit.h>
#include <DriverKit/OSSharedPtr.h>
#include <PCIDriverKit/PCIDriverKit.h>

struct TinyGPUDriverUserClient_IVars
{
	OSSharedPtr<TinyGPUDriver> m_provider = nullptr;
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
		ivars->m_provider.reset();
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

	ivars->m_provider = OSSharedPtr(OSDynamicCast(TinyGPUDriver, in_provider), OSRetain);
	return 0;

error:
	ivars->m_provider.reset();
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

	if (ivars->m_provider.get() == nullptr) {
		return kIOReturnNotAttached;
	}

	if (type < 6) {
		uint32_t bar = (uint32_t)type;
		return ivars->m_provider->MapBar(bar, memory);
	}

	// dma page buffer
	return ivars->m_provider->CreateDMA(type, memory);
}
