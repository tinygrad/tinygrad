#include <sstream>

#include <ocelot/ir/Module.h>
#include <ocelot/executive/EmulatedKernel.h>
#include <ocelot/executive/NVIDIAGPUDevice.h>
#include <ocelot/executive/RuntimeException.h>
#include <ocelot/executive/CooperativeThreadArray.h>


struct state {
	ir::Module *mod;
	executive::EmulatedKernel *emuk;
	std::stringstream *ss;
};

extern "C"
{

void* ptx_kernel_create(const char* source)
{
	state* S = new state;
	S->ss = new std::stringstream;
	(*S->ss) << source;
	S->mod = new ir::Module((void*)source, *S->ss);
	if (!S->mod->loaded())
		return NULL;

	ir::PTXKernel* rawk = S->mod->kernels().begin()->second; // assume 1 kernel / module
	// printf("kernelp %p\n", rawk);
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

void ptx_call(void* hstate, int n_args, void* args[],
             int blck_x, int blck_y, int blck_z,
             int grid_x, int grid_y, int grid_z)
{
	state* S = (state*)hstate;
	ir::PTXKernel* rawk = S->mod->kernels().begin()->second;
	const ir::PTXKernel::Prototype proto = rawk->getPrototype();
	for (int i=0; i<n_args; i++) {
		rawk->arguments[i];
		ir::Parameter* param = S->emuk->getParameter(proto.arguments[i].name);
		// printf("updating arg %d to %p\n", i, args[i]);
		param->arrayValues.resize(1);
		param->arrayValues[0].val_u64 = (ir::PTXU64)args[i];
	}
	S->emuk->updateArgumentMemory();

	S->emuk->setKernelShape(blck_x, blck_y, blck_z);
	S->emuk->launchGrid(grid_x, grid_y, grid_z);
}

}	

// void tinytest()
// {
// 	auto printarr = [](float* a, int n) {
// 		printf("%p = [", a);
// 		if (n) {
// 			for (int i=0; i<n-1; i++) {
// 				printf("%f, ", a[i]);
// 			}
// 			printf("%f", a[n-1]);
// 		}
// 		printf("]\n");
// 	};

// 	auto allocb = [](int n) {
// 		float* buff =  new alignas(32) float[n];
// 		printf("[+] allocating float[%d] buffer at %p\n", n, buff);
// 		return buff;
// 	};
// 	char* kernels[] = {
// 		"../../kernels/1.ptx",
// 		"../../kernels/2.ptx"
// 	};

// 	const int N = 16;
// 	float* buff0 = allocb(N);
// 	{
// 		ir::Module mod;
// 		mod.load(std::string(kernels[0]));
// 		ir::PTXKernel* rawk = mod.kernels().begin()->second; // assume 1 kernel / module
// 		assert(rawk);
// 		printf("[+] loaded kernel %s from '%s'\n", rawk->getPrototype().identifier.c_str(), kernels[0]);

// 		executive::NVIDIAGPUDevice exec;
// 		executive::EmulatedKernel emuk(rawk, &exec);


// 		printf("[+] binding param_0 to %p\n", buff0);
// 		ir::Parameter* param0 = emuk.getParameter("_Z4E_16Pf_param_0");
// 		param0->arrayValues.resize(1);
// 		param0->arrayValues[0].val_u64 = (ir::PTXU64)buff0;
// 		printf("[+] commiting argument memory\n");
// 		emuk.updateArgumentMemory();

// 		printf("[+] launching kernel (%d, %d, %d), [%d, %d, %d]\n", 1, 1, 1, N, 1, 1);
// 		emuk.setKernelShape(N,1,1);
// 		emuk.launchGrid(1,1,1);
// 		printarr(buff0, N);
// 	}
// 	delete[] buff0;
// }

// }

// int main() {
// 	printf("aaa\n");
// }
