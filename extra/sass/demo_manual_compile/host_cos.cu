#include "helpers.h"

using namespace std;

// Variables
CUdevice cuDevice;
CUcontext cuContext;
CUmodule cuModule;
CUfunction sgemm_kernel;
float *h_A;
CUdeviceptr d_A;

int main(int argc, char **argv)
{
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <kernel_name> <module_path>\n";
        return 1;
    }
    std::string kernel_name = argv[1];
    std::string module_path = argv[2];
    int devID = 0;

    checkCudaErrors(cuInit(0));
    checkCudaErrors(cuDeviceGet(&cuDevice, devID));
    checkCudaErrors(cuCtxCreate(&cuContext, 0, cuDevice));
    checkCudaErrors(cuModuleLoad(&cuModule, module_path.c_str()));
    checkCudaErrors(cuModuleGetFunction(&sgemm_kernel, cuModule, kernel_name.c_str()));

    int M = 3;
    int sizeA = M * sizeof(float);
    h_A = (float *)malloc(sizeA);
    checkCudaErrors(cuMemAlloc(&d_A, sizeA));

    void *args[] = { &d_A };
    checkCudaErrors(
        cuLaunchKernel(sgemm_kernel, 
            1, 1, 1, // blockIdx x, y, z
            3, 1, 1, // threadIdx x, y, z
            0, // Shared mem bytes
            NULL, // hStream
            args, // Kernel params
            NULL // extra
        )
    );
    checkCudaErrors(cuMemcpyDtoH(h_A, d_A, sizeA));
    for (int i=0; i < M; i++) {
        printf("%f\n", h_A[i]);
    }
}
