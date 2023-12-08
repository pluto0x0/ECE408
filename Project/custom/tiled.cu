#ifdef TILED_CU

#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define CHECK_ERR { \
    cudaError_t error = cudaGetLastError(); \
    if(error != cudaSuccess) { \
        std::cerr<<"CUDA error: "<<__FILE__<<":"<<__LINE__<<": "<<cudaGetErrorString(error)<<std::endl; \
        exit(-1); \
    } \
}

#define BLOCK_SIZE 16
#define TILE_WIDTH ((BLOCK_SIZE - 1) * S + K)

__global__ void conv_forward_kernel(float *output, const float *input, const float  *mask, const int B, const int M, const int C, const int H, const int W, const int K,const int S)
{
    /*
    mask - convolution kernel
    B - batch_size (number of images in x)
    M - number of output feature maps
    C - number of input feature maps
    H - input height dimension
    W - input width dimension
    K - kernel height and width (K x K)
    S - stride step length
    */

    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;

    #define out_4d(i3, i2, i1, i0) output[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    #define tile(i2, i1, i0) tile_input[(i2) * tw * tw + tw * (i1) + (i0)]
    
    extern __shared__ float tile_input[];

    const unsigned int b = blockIdx.x;
    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;
    unsigned int W_blksize = (W_out - 1) / BLOCK_SIZE + 1; // number of horizontal tiles per output map
    unsigned int m = blockIdx.y; // output channel
    unsigned int h_s = (blockIdx.z / W_blksize) * BLOCK_SIZE + threadIdx.y;
    unsigned int h = h_s * S;
    unsigned int w_s = (blockIdx.z % W_blksize) * BLOCK_SIZE + threadIdx.x;
    unsigned int w = w_s * S;
    const unsigned int tw = TILE_WIDTH;

    // if (w + K - 1 >= W || h + K - 1 >= H) return;

    float acc = 0.;

        for (int c = 0; c < C; c++) {
            int margin = min(S, K);
            for(int y = 0; y < margin && ty * S + y < tw; y++)
                for(int x = 0; x < margin && tx * S + x < tw; x++)
                    tile(c, ty * S + y, tx * S + x) = in_4d(b, c, h + y, w + x);
        }
    __syncthreads();
    if(tx < BLOCK_SIZE && ty < BLOCK_SIZE)
    for (int c = 0; c < C; c++) {
        // if(w + K - 1 < W && h + K - 1 < H)
        for (int y = 0; y < K; y++)
            for (int x = 0; x < K; x++)
                acc += tile(c, ty * S + y, tx * S + x) * mask_4d(m, c, y, x);
    }
    // if (w + K - 1 < W && h + K - 1 < H)
    if(tx < BLOCK_SIZE && ty < BLOCK_SIZE)
        if(h_s < H_out && w_s < W_out)
            out_4d(b, m, h_s, w_s) = acc;

    #undef tile
    #undef out_4d
    #undef in_4d
    #undef mask_4d
}

	
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    unsigned int H_out = (H - K)/S + 1;
    unsigned int W_out = (W - K)/S + 1;
    unsigned int size_in = B * W * H * C * sizeof(float);
    unsigned int size_out = B * H_out * W_out * M * sizeof(float);
    unsigned int sizse_mask = K * K * M * C * sizeof(float); 

    cudaMalloc(device_input_ptr, size_in);
    cudaMalloc(device_output_ptr, size_out);
    cudaMalloc(device_mask_ptr, sizse_mask);

    cudaMemcpy(*device_input_ptr, host_input, size_in, cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr, host_mask, sizse_mask, cudaMemcpyHostToDevice);
    
    CHECK_ERR;
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    std::cerr << "Running in " << __FILE__ << std::endl;

    unsigned int H_out = (H - K)/S + 1;
    unsigned int W_out = (W - K)/S + 1;
    unsigned int W_blksize = (W_out - 1) / BLOCK_SIZE + 1; // number of horizontal tiles per output map
    unsigned int H_blksize = (H_out - 1) / BLOCK_SIZE + 1; // number of vertical tiles per output map
    unsigned int Y = H_blksize * W_blksize; // total number of tiles per map

    dim3 grid(B, M, Y);
    int margin = K > S ? (K - S - 1) / S + 1 : 0;
    dim3 block(BLOCK_SIZE + margin, BLOCK_SIZE + margin); // output tile for untiled code
    unsigned int shared_size = C * TILE_WIDTH * TILE_WIDTH *  sizeof(float);
    conv_forward_kernel<<<grid, block, shared_size>>>(device_output, device_input, device_mask, B, M, C, H, W, K, S);

    CHECK_ERR;
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Copy the output back to host
    unsigned int H_out = (H - K)/S + 1;
    unsigned int W_out = (W - K)/S + 1;
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

#endif