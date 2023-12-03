// http://s3.amazonaws.com/files.rai-project.com/userdata/build-6545ac89d42bea0a4d9556ae.tar.gz

#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#include <cassert>
#define SHOWTIME

#define TILE_WIDTH 32
#define CHECK_ERR { \
    cudaError_t error = cudaGetLastError(); \
    if(error != cudaSuccess) { \
        std::cerr<<"CUDA error: "<<__FILE__<<":"<<__LINE__<<": "<<cudaGetErrorString(error)<<std::endl; \
        exit(-1); \
    } \
}
#define CONSTANT_SIZE 12544


__global__ void matrixMultiply(const float *A, const float *B, float *C, int numARows,
    int numAColumns, int numBRows,
    int numBColumns, int numCRows,
    int numCColumns, int BB) {

    #define INDEX(M, b, r, c) ((b)*(num##M##Columns)*(num##M##Rows) + num##M##Columns * (r) + (c))
    #define Matrix(M, b, r, c) (M[INDEX(M, b, r, c)])

    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x, by = blockIdx.y, tx = threadIdx.x, ty = threadIdx.y;
    int row = TILE_WIDTH * by + ty;
    int col = TILE_WIDTH * bx + tx;

    for(int b = 0; b < BB; b++) {
        float Cvalue = 0.;

        for(int q = 0; q < (numAColumns - 1) / TILE_WIDTH + 1; q++) {    // 0 <= q < ceil(numAColumns / TILEWIDTH)

            int orgRow = by * TILE_WIDTH + ty, orgCol = q * TILE_WIDTH + tx;
            tileA[ty][tx] = (orgRow < numARows && orgCol < numAColumns) ? Matrix(A, 0, orgRow, orgCol) : 0.;

            orgRow = q * TILE_WIDTH + ty, orgCol = bx * TILE_WIDTH + tx;
            tileB[ty][tx] = (orgRow < numBRows && orgCol < numBColumns) ? Matrix(B, b, orgRow, orgCol) : 0.;

            __syncthreads();    // q-th tile loaded

            for(int i = 0; i < TILE_WIDTH; i++)
                Cvalue += tileA[ty][i] * tileB[i][tx];

            // __syncthreads();
        }
        if(row < numCRows && col < numCColumns) {
            Matrix(C, b, row, col) = Cvalue;            
        }
        // __syncthreads();
    }
    #undef Matrix
}

__constant__ float mask[CONSTANT_SIZE];

#define H_out ((H - K)/S + 1)
#define W_out ((W - K)/S + 1)
#define MAT_W (H_out * W_out)
#define MAT_H (K * K * C)
#define MAT_TOT (MAT_H * MAT_W)

__global__ void transform_matrix(float *output, const float *input, const int B, const int M, const int C, const int H, const int W, const int K,const int S)
{
    /*
    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    B - batch_size (number of images in x)
    M - number of output feature maps
    C - number of input feature maps
    H - input height dimension
    W - input width dimension
    K - kernel height and width (K x K)
    S - stride step length
    */
    
    #define in_4d(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    const unsigned int mat_w = MAT_W;
    const unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
    const unsigned int thread_tot = blockDim.x * gridDim.x;
    const unsigned int mat_tot = MAT_TOT;
    const unsigned int kern_size = K * K;

    for(unsigned int i = id; i < mat_tot * B; i += thread_tot) {
        const unsigned int b = i / mat_tot;
        const unsigned int h = (i % mat_tot) / mat_w, w = (i % mat_tot) % mat_w;
        const unsigned int c = h / kern_size, id_in_kern = h % kern_size;
        const unsigned int kern_no_w = w % W_out, kern_no_h = w / W_out;
        const unsigned int w_in_kern = id_in_kern % K, h_in_kern = id_in_kern / K;
        output[i] = in_4d(b, c, kern_no_h * S + h_in_kern, kern_no_w * S + w_in_kern);
    }

    #undef mat
    #undef in_4d
}

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    unsigned int size_in = B * W * H * C * sizeof(float);
    unsigned int size_out = B * MAT_W * M * sizeof(float);
    unsigned int size_mask = K * K * M * C * sizeof(float);

    cudaMalloc(device_input_ptr, size_in);
    cudaMalloc(device_output_ptr, size_out);
    cudaMalloc(device_mask_ptr, size_mask);

    cudaMemcpy(*device_input_ptr, host_input, size_in, cudaMemcpyHostToDevice);
    cudaMemcpy(*device_output_ptr, host_output, size_out, cudaMemcpyHostToDevice);

    cudaMemcpy(*device_mask_ptr, host_mask, size_mask, cudaMemcpyHostToDevice);
    // if(size_mask > CONSTANT_SIZE) std::cerr << "mask size exceeded: " << size_mask << std::endl;
    // else cudaMemcpyToSymbol(mask, host_mask, size_mask);
    
    CHECK_ERR;
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    float *device_mat;
    cudaMalloc(&device_mat, B * MAT_TOT * sizeof(float));

#ifdef SHOWTIME
cudaEvent_t start, stop;
float milliseconds;
cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord(start);
#endif

    dim3 block(1024);
    dim3 grid(min(1024, B * MAT_TOT / 1024));
    transform_matrix<<<grid, block>>>(device_mat, device_input, B, M, C, H, W, K, S); CHECK_ERR;

#ifdef SHOWTIME
cudaEventRecord(stop);
cudaEventSynchronize(stop);
milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start, stop);
std::cerr << "Function 1 execution time: " << milliseconds << " ms" << std::endl;
cudaEventDestroy(start);
cudaEventDestroy(stop);
#endif

#ifdef SHOWTIME
cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord(start);
#endif

    grid = dim3((MAT_W - 1) / TILE_WIDTH + 1, (M - 1) / TILE_WIDTH + 1);
    block = dim3(TILE_WIDTH, TILE_WIDTH); // output tile for untiled code
    matrixMultiply<<<grid, block>>>(device_mask, device_mat, device_output, M, MAT_H, MAT_H, MAT_W, M, MAT_W, B); CHECK_ERR;

#ifdef SHOWTIME
cudaEventRecord(stop);
cudaEventSynchronize(stop);
milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start, stop);
std::cerr << "Function 2 execution time: " << milliseconds << " ms" << std::endl;
cudaEventDestroy(start);
cudaEventDestroy(stop);
#endif

    cudaFree(device_mat); CHECK_ERR;
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Copy the output back to host
    unsigned int size_out = B * H_out * W_out * M * sizeof(float);
    cudaMemcpy(host_output, device_output, size_out, cudaMemcpyDeviceToHost);
   
    // Free device memory 
    cudaFree(device_input);
    cudaFree(device_output);
    cudaFree(device_mask);

    CHECK_ERR;
}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}
