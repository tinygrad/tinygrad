#include <sstream>

#include <ocelot/ir/Module.h>
#include <ocelot/executive/EmulatedKernel.h>
#include <ocelot/executive/NVIDIAGPUDevice.h>
#include <ocelot/executive/RuntimeException.h>
#include <ocelot/executive/CooperativeThreadArray.h>

extern "C"
{

enum PTX_ERR
{
	PTX_ERR_SUCCESS = 0,
	PTX_ERR_LOAD_FAILED,
	PTX_ERR_KERNEL_NOT_FOUND,
	PTX_ERR_ARGS_MISMATCH,
};

enum PTX_ERR run_ptx(const char* source, int n_args, void* args[],
                     int blck_x, int blck_y, int blck_z,
                     int grid_x, int grid_y, int grid_z,
					 int log_level)
{
	std::stringstream ss;
	ss << source;
	ir::Module mod((void*)source, ss);
	if (!mod.loaded())
		return PTX_ERR_LOAD_FAILED;

	ir::PTXKernel* rawk = mod.kernels().begin()->second; // assume 1 kernel / module
	if (!rawk)
		return PTX_ERR_KERNEL_NOT_FOUND;

	const ir::PTXKernel::Prototype proto = rawk->getPrototype();
	if (proto.arguments.size() != n_args)
		return PTX_ERR_ARGS_MISMATCH;

	executive::EmulatedKernel emuk(rawk, NULL);
	
	for (int i=0; i<n_args; i++) {
		ir::Parameter* param = emuk.getParameter(proto.arguments[i].name);
		param->arrayValues.resize(1);
		param->arrayValues[0].val_u64 = (ir::PTXU64)args[i];
	}
	emuk.updateArgumentMemory();

	emuk.setKernelShape(blck_x, blck_y, blck_z);
	emuk.launchGrid(grid_x, grid_y, grid_z);

	return PTX_ERR_SUCCESS;
}

}	
