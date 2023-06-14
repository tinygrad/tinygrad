#include <ocelot/ir/Module.h>
#include <ocelot/executive/EmulatedKernel.h>
#include <ocelot/executive/CooperativeThreadArray.h>

struct state {
	ir::Module *mod;
	executive::EmulatedKernel *emuk;
	std::stringstream *ss;
};

extern "C" {

void* ptx_kernel_create(const char* source) {
	state* S = new state;
	S->ss = new std::stringstream;
	(*S->ss) << source;
	S->mod = new ir::Module((void*)source, *S->ss);
	if (!S->mod->loaded())
		return NULL;
	ir::PTXKernel* rawk = S->mod->kernels().begin()->second;
	if (!rawk)
		return NULL;
	S->emuk = new executive::EmulatedKernel(rawk, NULL);
	return S;
}

void ptx_kernel_destroy(void* hstate) {
	state* S = (state*)hstate;
	delete S->emuk;
	delete S->mod;
	delete S->ss;
	delete S;
}

void ptx_call(void* hstate, int n_args, void* args[], int blck_x, int blck_y, int blck_z, int grid_x, int grid_y, int grid_z) {
	state* S = (state*)hstate;
	ir::PTXKernel* rawk = S->mod->kernels().begin()->second;
	const ir::PTXKernel::Prototype proto = rawk->getPrototype();
	for (int i=0; i<n_args; i++) {
		rawk->arguments[i];
		ir::Parameter* param = S->emuk->getParameter(proto.arguments[i].name);
		param->arrayValues.resize(1);
		param->arrayValues[0].val_u64 = (ir::PTXU64)args[i];
	}
	S->emuk->updateArgumentMemory();
	S->emuk->setKernelShape(blck_x, blck_y, blck_z);
	S->emuk->launchGrid(grid_x, grid_y, grid_z);
}

}	
