// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256

#define BLOCK_SIZE 1024
#define MAX_GRID 1024

//@@ insert code here
__global__ void castUchar (float *input, unsigned char *output, int len) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  while(i < len) {
    output[i] = (unsigned char) (255. * input[i]);
    i += stride;
  }
}

__global__ void convertGray (unsigned char *input, unsigned char *output, int len) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  while(i < len) {
		unsigned char r = input[3*i];
		unsigned char g = input[3*i + 1];
		unsigned char b = input[3*i + 2];
    output[i] = (unsigned char) (0.21*r + 0.71*g + 0.07*b);
    i += stride;
  }
__global__ void histogram (unsigned char *buffer, int *histo, int size) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  // stride is total number of threads
  int stride = blockDim.x * gridDim.x;
  while (i < size) {
    atomicAdd( &(histo[buffer[i]]), 1);
    i += stride;
  }
}

__global__ void scan(int *input, float *output, int len, float scale) {
  // Brent-Kung
  __shared__ float arr[BLOCK_SIZE * 2];
  const int tx = threadIdx.x, bx = blockDim.x * blockIdx.x * 2;
  arr[tx * 2] = bx + tx * 2 < len ? scale * input[bx + tx * 2] : 0.;
  arr[tx * 2 + 1] = bx + tx * 2 + 1 < len ? scale * input[bx + tx * 2 + 1] : 0.;
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
}

__device__ unsigned char clamp(unsigned char x, unsigned char start, unsigned char end) {
	return min(max(x, start), end);
}

__device__ unsigned char correct_color(int val, float *cdf) {
	return clamp((unsigned char)(255.*(cdf[val] - cdf[0])/(1.0 - cdf[0])), 0, 255);
}


__global__ void correctColor (unsigned char *input, unsigned char *output, int len, float *cdf) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  while(i < len) {
    output[i] = correct_color(input[i], cdf);
    i += stride;
  }
}

__global__ void castFloat (unsigned char *input, float *output, int len) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  while(i < len) {
    output[i] = (float)input[i] / 255.0;
    i += stride;
  }
}

int gridSize(int num) {
  int numBlock = (num - 1) / BLOCK_SIZE + 1; // ceil
  return min(numBlock, MAX_GRID);
}

int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  //@@ Insert more code here
  float *deviceFloatImage;
  unsigned char *deviceUcharImage;
  unsigned char *deviceGrayImage;
  int *deviceHist;
  float *deviceCDF;


  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  //@@ insert code here
  const int numElements = imageWidth * imageHeight;
  const int numColor =  numElements * imageChannels;

  cudaMalloc(&deviceFloatImage, numColor * sizeof(float));
  cudaMalloc(&deviceUcharImage, numColor * sizeof(unsigned char));
  cudaMalloc(&deviceGrayImage, numElements * sizeof(unsigned char));
  cudaMalloc(&deviceHist, HISTOGRAM_LENGTH * sizeof(int));
  cudaMemset(deviceHist, 0, HISTOGRAM_LENGTH * sizeof(int));
  cudaMalloc(&deviceCDF, HISTOGRAM_LENGTH * sizeof(float));
  cudaMemset(deviceCDF, 0, HISTOGRAM_LENGTH * sizeof(float));

  cudaMemcpy(deviceFloatImage, hostInputImageData, numColor * sizeof(float), cudaMemcpyHostToDevice);  
  castUchar<<<gridSize(numColor), BLOCK_SIZE>>>(deviceFloatImage, deviceUcharImage, numColor);
  convertGray<<<gridSize(numElements), BLOCK_SIZE>>>(deviceUcharImage, deviceGrayImage, numElements);
  histogram<<<gridSize(numElements), BLOCK_SIZE>>>(deviceGrayImage, deviceHist, numElements);
  scan<<<1, HISTOGRAM_LENGTH / 2>>>(deviceHist, deviceCDF, numElements, 1. / numElements);
  correctColor<<<gridSize(numColor), BLOCK_SIZE>>>(deviceUcharImage, deviceUcharImage, numColor, deviceCDF);
  castFloat<<<gridSize(numColor), BLOCK_SIZE>>>(deviceUcharImage, deviceFloatImage, numColor);

  cudaMemcpy(hostOutputImageData, deviceFloatImage, numColor * sizeof(float), cudaMemcpyDeviceToHost);

  wbSolution(args, outputImage);

  //@@ insert code here
  cudaFree(deviceFloatImage);
  cudaFree(deviceUcharImage);
  cudaFree(deviceGrayImage);
  cudaFree(deviceHist);
  cudaFree(deviceCDF);

  return 0;
}
