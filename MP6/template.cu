// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 1024 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void scan(float *input, float *output, int len, float *blockSum) {
  // Brent-Kung
  __shared__ float arr[BLOCK_SIZE * 2];
  const int tx = threadIdx.x, bx = blockDim.x * blockIdx.x * 2;
  arr[tx * 2] = bx + tx * 2 < len ? input[bx + tx * 2] : 0.;
  arr[tx * 2 + 1] = bx + tx * 2 + 1 < len ? input[bx + tx * 2 + 1] : 0.;
  __syncthreads();
  int stride;

  for(stride = 1; stride <= BLOCK_SIZE; stride <<= 1) {
    __syncthreads();
    if(!((tx + 1) & (stride - 1))) {
      arr[tx * 2 + 1] += arr[tx * 2 + 1 - stride];
    }
  }
  for(stride >>= 1; stride; stride >>= 1) {
    __syncthreads();
    if(tx < BLOCK_SIZE - 1 && !((tx + 1) & (stride - 1))) {
      arr[tx * 2 + 1 + stride] += arr[tx * 2 + 1];
    }
  }
  __syncthreads();
  if(bx + tx * 2 < len) output[bx + tx * 2] = arr[tx * 2];
  if(bx + tx * 2 + 1 < len) output[bx + tx * 2 + 1] = arr[tx * 2 + 1];
  if(blockSum != nullptr && tx == BLOCK_SIZE - 1) blockSum[blockIdx.x] = arr[tx * 2 + 1];
}


__global__ void add(float *input, float *output, int len) {
  // Brent-Kung
  const int tx = threadIdx.x, bx = blockDim.x * blockIdx.x * 2;
  float sum = blockIdx.x ? input[blockIdx.x - 1] : 0.;
  output[bx + tx * 2] += sum;
  output[bx + tx * 2 + 1] += sum;
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  int gridDim = ceil(double(numElements) / (BLOCK_SIZE * 2));
  float *deviceBlockSum;
  cudaMalloc(&deviceBlockSum, gridDim * sizeof(float));

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce

  scan<<<gridDim, BLOCK_SIZE>>>(deviceInput, deviceOutput, numElements, deviceBlockSum);
  scan<<<1, BLOCK_SIZE>>>(deviceBlockSum, deviceBlockSum, gridDim, nullptr);  // only when gridDim <= BLOCKSIZE * 2
  add<<<gridDim, BLOCK_SIZE>>>(deviceBlockSum, deviceOutput, numElements);
  
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}
