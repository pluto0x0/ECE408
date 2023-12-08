// #define MATRIX_CU
// #define BASELINE_CU
// #define UNROLL_CU
// #define CONSTANT_CU
// #define FP16_CU
// #define BEST_CU
#define TILED_CU

#include "baseline.cu"
#include "unroll.cu"
#include "matrix.cu"
#include "constant.cu"
#include "tiled.cu"
#include "fp16.cu"
#include "best.cu"