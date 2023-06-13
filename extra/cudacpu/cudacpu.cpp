#include <sstream>

#include <ocelot/ir/Module.h>
#include <ocelot/executive/EmulatedKernel.h>
#include <ocelot/executive/NVIDIAGPUDevice.h>
#include <ocelot/executive/RuntimeException.h>
#include <ocelot/executive/CooperativeThreadArray.h>

extern "C"
{

void* ptx_kernel_create(const char* source)
{
	std::stringstream ss;
	ss << source;
	ir::Module mod((void*)source, ss);
	if (!mod.loaded())
		return NULL;

	ir::PTXKernel* rawk = mod.kernels().begin()->second; // assume 1 kernel / module
	if (!rawk)
		return NULL

	return new executive::EmulatedKernel(rawk, NULL);
}

void ptx_kernel_destroy(void* kernel) { delete (executive::EmulatedKernel*)kernel; }

void ptx_call(void* kernel, int n_args, void* args[],
             int blck_x, int blck_y, int blck_z,
             int grid_x, int grid_y, int grid_z)
{
	executive::EmulatedKernel* emuk = (executive::EmulatedKernel*)kernel;
	for (int i=0; i<n_args; i++) {
		ir::Parameter* param = emuk->getParameter(proto.arguments[i].name);
		param->arrayValues.resize(1);
		param->arrayValues[0].val_u64 = (ir::PTXU64)args[i];
	}
	emuk->updateArgumentMemory();

	emuk->setKernelShape(blck_x, blck_y, blck_z);
	emuk->launchGrid(grid_x, grid_y, grid_z);
}

}	
