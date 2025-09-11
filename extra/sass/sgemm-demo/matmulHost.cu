#include "helpers.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <cmath>

using namespace std;

#define DTYPE float
//#define M 3 
//#define N 2
#define M 512
#define N 512
#define K 512

// Error checking macro for CUDA Runtime API
#define cudaCheck(error) \
    do { \
        cudaError_t err = (error); \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)


// Variables
CUdevice cuDevice;
CUcontext cuContext;
CUmodule cuModule;
CUfunction sgemm_kernel;
DTYPE *h_A, *h_B, *h_C, *h_C_ref;
CUdeviceptr d_A, d_B, d_C, d_C_ref;
CUdeviceptr d_A_col, d_B_col;
cublasHandle_t handle;

// Function to run cuBLAS kernel
void run_cublas_kernel(float alpha, CUdeviceptr dA, CUdeviceptr dB, float beta, CUdeviceptr dC) {
    float* dA_ptr = reinterpret_cast<float*>(dA);
    float* dB_ptr = reinterpret_cast<float*>(dB);
    float* dC_ptr = reinterpret_cast<float*>(dC);
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, dB_ptr, CUDA_R_32F,
               N, dA_ptr, CUDA_R_32F, K, &beta, dC_ptr, CUDA_R_32F, N, CUBLAS_COMPUTE_32F,
               CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

// Function to verify matrix results
bool verify_matrix(DTYPE* ref, DTYPE* test, int size, float epsilon = 1e-2) {
    for (int i = 0; i < size; i++) {
        if (fabs(ref[i] - test[i]) > epsilon) {
            std::cout << "Mismatch at index " << i << ": ref=" << ref[i] << ", test=" << test[i] << std::endl;
            return false;
        }
    }
    return true;
}

// Function to print matrix
void print_matrix(DTYPE* matrix, int rows, int cols, std::ostream& os = std::cout) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            os << matrix[i * cols + j] << " ";
        }
        os << std::endl;
    }
}

int main(int argc, char **argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <kernel_name> <module_path>\n";
        return 1;
    }
    std::string kernel_name = argv[1];
    std::string module_path = argv[2];
    int devID = 0;

    // Initialize cuBLAS
    if (cublasCreate(&handle)) {
        std::cerr << "Create cublas handle error." << std::endl;
        exit(EXIT_FAILURE);
    };

    // Create events for timing

    // Initialize CUDA
    checkCudaErrors(cuInit(0));
    checkCudaErrors(cuDeviceGet(&cuDevice, devID));
    checkCudaErrors(cuCtxCreate(&cuContext, 0, cuDevice));
    checkCudaErrors(cuModuleLoad(&cuModule, module_path.c_str()));
    checkCudaErrors(cuModuleGetFunction(&sgemm_kernel, cuModule, kernel_name.c_str()));

    // Matrix dimensions and sizes
    int sizeA = M * K * sizeof(DTYPE);
    int sizeB = K * N * sizeof(DTYPE);
    int sizeC = M * N * sizeof(DTYPE);
    
    // Allocate host memory
    h_A = (DTYPE *)malloc(sizeA);
    h_B = (DTYPE *)malloc(sizeB);
    h_C = (DTYPE *)malloc(sizeC);
    h_C_ref = (DTYPE *)malloc(sizeC);

    // Initialize matrices
    initMatrix(h_A, M*K, InitMode::RANDOM);
    initMatrix(h_B, K*N, InitMode::RANDOM);


    // Allocate device memory
    checkCudaErrors(cuMemAlloc(&d_A, sizeA));
    checkCudaErrors(cuMemAlloc(&d_B, sizeB));
    checkCudaErrors(cuMemAlloc(&d_C, sizeC));
    checkCudaErrors(cuMemAlloc(&d_C_ref, sizeC));
    
    // Copy data to device
    checkCudaErrors(cuMemcpyHtoD(d_A, h_A, sizeA));
    checkCudaErrors(cuMemcpyHtoD(d_B, h_B, sizeB));
    
    // Initialize output matrices to zero
    checkCudaErrors(cuMemsetD8(d_C, 0, sizeC));
    checkCudaErrors(cuMemsetD8(d_C_ref, 0, sizeC));

    // Parameters for GEMM
    float alpha = 1.0f;
    float beta = 0.0f;

    // Run cuBLAS kernel for reference
    run_cublas_kernel(alpha, d_A, d_B, beta, d_C_ref);
    cudaCheck(cudaDeviceSynchronize());
    
    // Run custom kernel
    void *args[] = { &d_C, &d_A, &d_B };
    checkCudaErrors(
        cuLaunchKernel(sgemm_kernel, 
            //2, 3, 1,    // gridDim x, y, z
            512, 512, 1,    // gridDim x, y, z
            1, 1, 1,    // blockDim x, y, z
            0,           // sharedMemBytes
            NULL,        // hStream
            args,        // kernel params
            NULL         // extra
        )
    );
    cudaCheck(cudaDeviceSynchronize());
    cudaCheck(cudaGetLastError());

    // Copy results back to host
    checkCudaErrors(cuMemcpyDtoH(h_C, d_C, sizeC));
    checkCudaErrors(cuMemcpyDtoH(h_C_ref, d_C_ref, sizeC));

    // Verify results
    printf("M %d N %d K %d\n", M, N, K);
    if (!verify_matrix(h_C_ref, h_C, M * N)) {
        std::cout << "Failed to pass the correctness verification against NVIDIA cuBLAS." << std::endl;
        
        // For small matrices, print the details
        if (M <= 128) {
            std::cout << "A:" << std::endl;
            print_matrix(h_A, M, K);
            std::cout << "B:" << std::endl;
            print_matrix(h_B, K, N);
            std::cout << "Custom kernel result:" << std::endl;
            print_matrix(h_C, M, N);
            std::cout << "cuBLAS result:" << std::endl;
            print_matrix(h_C_ref, M, N);
        }
        exit(EXIT_FAILURE);
    } else {
        std::cout << "Results match cuBLAS reference!" << std::endl;
    }

    // Time custom kernel
    float elapsed_time_custom;
    cudaEvent_t beg, end;
    cudaEventCreate(&beg);
    cudaEventCreate(&end);
    int repeat_times = 50;
    cudaEventRecord(beg);
    for (int j = 0; j < repeat_times; j++) {
        checkCudaErrors(
            cuLaunchKernel(sgemm_kernel, 
                N, M, 1,    // gridDim x, y, z
                1, 1, 1,    // blockDim x, y, z
                0,           // sharedMemBytes
                NULL,        // hStream
                args,        // kernel params
                NULL         // extra
            )
        );
    }
    cudaEventRecord(end);
    cudaEventSynchronize(beg);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&elapsed_time_custom, beg, end);
    elapsed_time_custom /= 1000.0f; // Convert to seconds

    // Time cuBLAS kernel
    cudaEventRecord(beg);
    for (int j = 0; j < repeat_times; j++) {
        run_cublas_kernel(alpha, d_A, d_B, beta, d_C_ref);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(beg);
    cudaEventSynchronize(end);
    
    float elapsed_time_cublas;
    cudaEventElapsedTime(&elapsed_time_cublas, beg, end);
    elapsed_time_cublas /= 1000.0f; // Convert to seconds

    // Calculate performance
    long flops = 2 * M * N * K;
    printf("Custom kernel - Average elapsed time: (%7.6f) s, performance: (%7.1f) GFLOPS\n",
           elapsed_time_custom / repeat_times,
           (repeat_times * flops * 1e-9) / elapsed_time_custom);
    printf("cuBLAS kernel - Average elapsed time: (%7.6f) s, performance: (%7.1f) GFLOPS\n",
           elapsed_time_cublas / repeat_times,
           (repeat_times * flops * 1e-9) / elapsed_time_cublas);

    // Cleanup
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);
    checkCudaErrors(cuMemFree(d_A));
    checkCudaErrors(cuMemFree(d_B));
    checkCudaErrors(cuMemFree(d_C));
    checkCudaErrors(cuMemFree(d_C_ref));
    cublasDestroy(handle);
    cudaEventDestroy(beg);
    cudaEventDestroy(end);

    return 0;
}
