#include <ocelot/ir/Module.h>
#include <ocelot/executive/EmulatedKernel.h>
struct state { ir::Module *mod; executive::EmulatedKernel *emuk; std::stringstream *ss; };
extern "C" void* ptx_kernel_create(const char* source) { return  new state {new ir::Module((void*)source, *new std::stringstream(source)), new executive::EmulatedKernel((*S->mod).kernels().begin()->second, NULL)}; }
extern "C" void ptx_kernel_destroy(void* hstate) {
	delete static_cast<state*>(hstate)->emuk;
	delete static_cast<state*>(hstate)->mod;
	delete static_cast<state*>(hstate)->ss;
	delete static_cast<state*>(hstate); }
extern "C" void ptx_call(void* hstate, int n_args, void* args[], int blck_x, int blck_y, int blck_z, int grid_x, int grid_y, int grid_z) {
	const ir::PTXKernel::Prototype proto = static_cast<state*>(hstate)->mod->kernels().begin()->second->getPrototype();
	for (int i=0; i<n_args; i++) {
		S->emuk->getParameter(proto.arguments[i].name)->arrayValues.resize(1);
		S->emuk->getParameter(proto.arguments[i].name)->arrayValues[0].val_u64 = (ir::PTXU64)args[i]; }
	static_cast<state*>(hstate)->emuk->updateArgumentMemory(), static_cast<state*>(hstate)->emuk->setKernelShape(blck_x, blck_y, blck_z), static_cast<state*>(hstate)->emuk->launchGrid(grid_x, grid_y, grid_z); }