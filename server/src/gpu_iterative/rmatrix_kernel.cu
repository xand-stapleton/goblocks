#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

#define MAX_CACHE 4096  // max number of cached matrices

// Simple GPU cache (fixed-size)
static double* d_matrices[MAX_CACHE] = {0};
static double rkeys[MAX_CACHE] = {0};
static int n_cached = 0;

// Device functions
__device__ double comb(int n, int k) {
    if (n < 0 || k < 0 || k > n) return NAN;
    if (k > n - k) k = n - k;
    if (k == 0) return 1.0;
    double num = 1.0, den = 1.0;
    for (int i = 1; i <= k; i++) {
        num *= (n - (k - i));
        den *= i;
    }
    return num / den;
}

__device__ double ff(double x, int n) {
    double res = 1.0;
    for (int i = 0; i < n; i++) res *= (x - i);
    return res;
}

__device__ double powInt(double a, int n) {
    double res = 1.0;
    if (n < 0) {
        for (int i = 0; i < -n; i++) res *= a;
        return 1.0/res;
    } else if (n == 0) return 1.0;
    else {
        for (int i = 0; i < n; i++) res *= a;
        return res;
    }
}

// Kernel: one thread per element
__global__ void buildRMatrixKernel(double* matrix, const int* orders, double rPower, double rStar, int num) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num*num) return;

    int i = idx / num;
    int j = idx % num;

    int mOut = orders[2*i];
    int nOut = orders[2*i+1];
    int s    = orders[2*j];
    int nIn  = orders[2*j+1];

    double val = 0.0;
    if (nOut == nIn && mOut >= s) {
        val = comb(mOut, s) * ff(rPower, mOut - s) * powInt(rStar, (int)(rPower - mOut + s));
    }

    matrix[i*num + j] = val;
}

// Pure C wrapper: compute or retrieve GPU matrix
extern "C" double* buildRMatrixGPUDev(int* orders, double rPower, double rStar, int num) {
    // Search cache
    for (int i = 0; i < n_cached; i++) {
        if (rkeys[i] == rPower) return d_matrices[i];
    }

    // Not found: allocate new matrix
    double* d_matrix;
    int* d_orders;
    cudaMalloc(&d_orders, sizeof(int)*2*num);
    cudaMalloc(&d_matrix, sizeof(double)*num*num);

    cudaMemcpy(d_orders, orders, sizeof(int)*2*num, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (num*num + threads - 1)/threads;
    buildRMatrixKernel<<<blocks, threads>>>(d_matrix, d_orders, rPower, rStar, num);
    cudaDeviceSynchronize();

    cudaFree(d_orders);

    // Store in cache (if there’s space)
    if (n_cached < MAX_CACHE) {
        rkeys[n_cached] = rPower;
        d_matrices[n_cached] = d_matrix;
        n_cached++;
    } else {
        printf("Warning: R matrix cache full, not caching rPower=%f\n", rPower);
    }

    return d_matrix;
}

// Free a single cached matrix
extern "C" void freeRMatrixGPU(double rPower) {
    for (int i = 0; i < n_cached; i++) {
        if (rkeys[i] == rPower) {
            cudaFree(d_matrices[i]);
            // shift remaining entries
            for (int j = i; j < n_cached-1; j++) {
                d_matrices[j] = d_matrices[j+1];
                rkeys[j] = rkeys[j+1];
            }
            n_cached--;
            break;
        }
    }
}

// Free all cached matrices
extern "C" void freeAllRMatricesGPU() {
    for (int i = 0; i < n_cached; i++) {
        cudaFree(d_matrices[i]);
    }
    n_cached = 0;
}

