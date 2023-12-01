
#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

#define TILE_WIDTH 32

// Compute C = A * B
__global__ void matrixMultiply(float *A, float *B, float *C, int numARows,
                               int numAColumns, int numBRows,
                               int numBColumns, int numCRows,
                               int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
  #define Matrix(M, r, c) (M[num##M##Columns * (r) + (c)])

  __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
  __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];
  int bx = blockIdx.x, by = blockIdx.y, tx = threadIdx.x, ty = threadIdx.y;
  int row = TILE_WIDTH * by + ty;
  int col = TILE_WIDTH * bx + tx;
  Matrix(C, row, col) = 0.;
  float Cvalue = 0.;
  for(int q = 0; q < (numAColumns + TILE_WIDTH - 1) / TILE_WIDTH; q++) {    // 0 <= q < ceil(numAColumns / TILEWIDTH)
    int orgRow = by * TILE_WIDTH + ty, orgCol = q * TILE_WIDTH + tx;
    tileA[ty][tx] = (orgRow < numARows && orgCol < numAColumns) ? Matrix(A, orgRow, orgCol) : 0.;
    orgRow = q * TILE_WIDTH + ty, orgCol = bx * TILE_WIDTH + tx;
    tileB[ty][tx] = (orgRow < numBRows && orgCol < numBColumns) ? Matrix(B, orgRow, orgCol) : 0.;
    __syncthreads();    // q-th tile loaded
    for(int i = 0; i < TILE_WIDTH; i++)
    Cvalue += tileA[ty][i] * tileB[i][tx];
    __syncthreads();
  }
  if(row < numCRows && col < numCColumns)
    Matrix(C, row, col) = Cvalue;
  #undef Matrix
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA;
  float *deviceB;
  float *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
  //@@ Set numCRows and numCColumns
  numCRows = numARows;
  numCColumns = numBColumns;

  //@@ Allocate the hostC matrix
  hostC = (float *)malloc(numCRows * numCColumns * sizeof(float));

  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  cudaMalloc(&deviceA, numARows * numAColumns * sizeof(float));
  cudaMalloc(&deviceB, numBRows * numBColumns * sizeof(float));
  cudaMalloc(&deviceC, numCRows * numCColumns * sizeof(float));

  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  cudaMemcpy(deviceA, hostA, numARows * numAColumns * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, numBRows * numBColumns * sizeof(float), cudaMemcpyHostToDevice);

  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 DimGrid(ceil(numCColumns * 1. / TILE_WIDTH), ceil(numCRows * 1. / TILE_WIDTH), 1);
  dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, 1);       // 32 * 32 = 1024

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  matrixMultiply<<<DimGrid, DimBlock>>>(deviceA, deviceB, deviceC,
                                        numARows, numAColumns,
                                        numBRows, numBColumns,
                                        numCRows, numCColumns);

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostC, deviceC, numCRows * numCColumns * sizeof(float), cudaMemcpyDeviceToHost);

  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);

  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}
