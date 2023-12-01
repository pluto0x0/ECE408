#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
#define KERNEL_WIDTH 3
#define BLOCK_WIDTH 8

//@@ Define constant memory for device kernel here
__constant__ float deviceKernel[27];

__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here

  const int offset = (KERNEL_WIDTH - 1) / 2;
  const int x_tile_size = BLOCK_WIDTH + offset * 2,
            y_tile_size = BLOCK_WIDTH + offset * 2,
            z_tile_size = BLOCK_WIDTH + offset * 2;
  __shared__ float tile[x_tile_size][y_tile_size][z_tile_size];
  const int tx = threadIdx.x,
            ty = threadIdx.y,
            tz = threadIdx.z;
  const int bx = tx + blockIdx.x * BLOCK_WIDTH,
            by = ty + blockIdx.y * BLOCK_WIDTH,
            bz = tz + blockIdx.z * BLOCK_WIDTH;
  #define DATA(var, x, y, z)\
  var[(z) * (x_size * y_size) + (y) * x_size + (x)]
  #define IN_RANGE(x, y, z) ((x)>=0&&(x)<x_size\
                           &&(y)>=0&&(y)<y_size\
                           &&(z)>=0&&(z)<z_size)
  #define GET(x, y, z) (IN_RANGE(x, y, z) ? DATA(input, x, y, z) : (0.))
  #define KER(x, y, z) deviceKernel[(z) * (KERNEL_WIDTH * KERNEL_WIDTH) + (y) * KERNEL_WIDTH + (x)]
  #define TILE(x, y, z) tile[(x) + offset][(y) + offset][(z) + offset]
  TILE(tx, ty, tz) = GET(bx, by, bz);
  __syncthreads();
  if(tx == 0 || ty == 0 || tz == 0 || tx == BLOCK_WIDTH - 1 || ty == BLOCK_WIDTH - 1 || tz == BLOCK_WIDTH - 1) 
    for(int dx = -offset; dx <= offset; dx++)
      for(int dy = -offset; dy <= offset; dy++)
        for(int dz = -offset; dz <= offset; dz++)        
          TILE(tx + dx, ty + dy, tz + dz) = GET(bx + dx, by + dy, bz + dz);
  __syncthreads();

  float answer = 0.;
  for(int dx = -offset; dx <= offset; dx++)
    for(int dy = -offset; dy <= offset; dy++)
      for(int dz = -offset; dz <= offset; dz++)
        answer += TILE(tx + dx, ty + dy, tz + dz) * KER(offset + dx, offset + dy, offset + dz);
  __syncthreads();

  if(IN_RANGE(bx, by, bz)) DATA(output, bx, by, bz) = answer;
  
  #undef TILE
  #undef KER
  #undef GET
  #undef IN_RANGE
  #undef DATA
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  cudaMalloc(&deviceInput, x_size * y_size * z_size * sizeof(float));
  cudaMalloc(&deviceOutput, x_size * y_size * z_size * sizeof(float));
  assert(kernelLength == KERNEL_WIDTH * KERNEL_WIDTH * KERNEL_WIDTH);

  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu
  cudaMemcpy(deviceInput, hostInput + 3, x_size * y_size * z_size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(deviceKernel, hostKernel, kernelLength * sizeof(float));

  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ Initialize grid and block dimensions here
  dim3 blockDim(BLOCK_WIDTH, BLOCK_WIDTH, BLOCK_WIDTH);
  dim3 gridDim(ceil(x_size * 1. / BLOCK_WIDTH),
               ceil(y_size * 1. / BLOCK_WIDTH),
               ceil(z_size * 1. / BLOCK_WIDTH)
              );
  //@@ Launch the GPU kernel here
  conv3d<<<gridDim, blockDim>>>(deviceInput, deviceOutput, z_size, y_size, x_size);
  
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  cudaMemcpy(hostOutput + 3, deviceOutput, x_size * y_size * z_size * sizeof(float), cudaMemcpyDeviceToHost);

  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  // Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}
