#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

typedef struct { int N; double Delta; int Ell; double C; int Idx; } PolesData;
typedef struct { int Ell; double Delta; } KeyData;

__global__ void kernel_recurse_order_p(
    int p,
    int n,
    int numEtaDerivs,
    int strideP,
    const KeyData *d_keys,
    const PolesData *d_poles,
    const int *d_polesOffset,
    const double *d_htildeCoeffs,
    double *d_hCoeffs
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * numEtaDerivs;

    for (int idx = tid; idx < total; idx += blockDim.x * gridDim.x) {
        int j = idx / numEtaDerivs;
        int q = idx % numEtaDerivs;
        int base = ((j * numEtaDerivs) + q) * strideP;

        double val = d_htildeCoeffs[base + p];
        double delta = d_keys[j].Delta;

        for (int poleIdx = d_polesOffset[j]; poleIdx < d_polesOffset[j + 1]; poleIdx++) {
            PolesData pole = d_poles[poleIdx];
            if (pole.N > p) {
                continue;
            }

            double denom = delta - pole.Delta;
            if (denom == 0.0) {
                continue;
            }

            int childBase = ((pole.Idx * numEtaDerivs) + q) * strideP;
            val += pole.C * d_hCoeffs[childBase + (p - pole.N)] / denom;
        }

        d_hCoeffs[base + p] = val;
    }
}

extern "C"
int recurse_hcoeffs_gpu(
    int n,
    int numEtaDerivs,
    int rOrder,
    const KeyData* h_keys,
    const PolesData* h_poles,
    const int* h_polesOffset,
    const double* h_htildeCoeffs,
    double* h_out_hcoeffs
) {
    if (n <= 0 || numEtaDerivs <= 0 || rOrder < 0 || h_keys == NULL || h_polesOffset == NULL || h_htildeCoeffs == NULL || h_out_hcoeffs == NULL) {
        return 10;
    }

    int strideP = rOrder + 1;
    int totalPoles = h_polesOffset[n];
    size_t keysSize = (size_t)n * sizeof(KeyData);
    size_t polesOffsetSize = (size_t)(n + 1) * sizeof(int);
    size_t hCoeffsSize = (size_t)n * (size_t)numEtaDerivs * (size_t)strideP * sizeof(double);
    size_t polesSize = (size_t)totalPoles * sizeof(PolesData);

    KeyData *d_keys = NULL;
    PolesData *d_poles = NULL;
    int *d_polesOffset = NULL;
    double *d_htildeCoeffs = NULL;
    double *d_hCoeffs = NULL;

    cudaError_t err;

    err = cudaMalloc(&d_keys, keysSize);
    if (err != cudaSuccess) {
        return 11;
    }
    err = cudaMemcpy(d_keys, h_keys, keysSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_keys);
        return 12;
    }

    err = cudaMalloc(&d_polesOffset, polesOffsetSize);
    if (err != cudaSuccess) {
        cudaFree(d_keys);
        return 13;
    }
    err = cudaMemcpy(d_polesOffset, h_polesOffset, polesOffsetSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_keys);
        cudaFree(d_polesOffset);
        return 14;
    }

    if (totalPoles > 0) {
        if (h_poles == NULL) {
            cudaFree(d_keys);
            cudaFree(d_polesOffset);
            return 15;
        }
        err = cudaMalloc(&d_poles, polesSize);
        if (err != cudaSuccess) {
            cudaFree(d_keys);
            cudaFree(d_polesOffset);
            return 16;
        }
        err = cudaMemcpy(d_poles, h_poles, polesSize, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            cudaFree(d_keys);
            cudaFree(d_polesOffset);
            cudaFree(d_poles);
            return 17;
        }
    }

    err = cudaMalloc(&d_htildeCoeffs, hCoeffsSize);
    if (err != cudaSuccess) {
        cudaFree(d_keys);
        cudaFree(d_polesOffset);
        if (d_poles) cudaFree(d_poles);
        return 18;
    }
    err = cudaMemcpy(d_htildeCoeffs, h_htildeCoeffs, hCoeffsSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_keys);
        cudaFree(d_polesOffset);
        if (d_poles) cudaFree(d_poles);
        cudaFree(d_htildeCoeffs);
        return 19;
    }

    err = cudaMalloc(&d_hCoeffs, hCoeffsSize);
    if (err != cudaSuccess) {
        cudaFree(d_keys);
        cudaFree(d_polesOffset);
        if (d_poles) cudaFree(d_poles);
        cudaFree(d_htildeCoeffs);
        return 20;
    }
    err = cudaMemset(d_hCoeffs, 0, hCoeffsSize);
    if (err != cudaSuccess) {
        cudaFree(d_keys);
        cudaFree(d_polesOffset);
        if (d_poles) cudaFree(d_poles);
        cudaFree(d_htildeCoeffs);
        cudaFree(d_hCoeffs);
        return 21;
    }

    int threads = 256;
    int blocks = (n * numEtaDerivs + threads - 1) / threads;
    if (blocks > 1024) {
        blocks = 1024;
    }
    if (blocks < 1) {
        blocks = 1;
    }

    for (int p = 0; p <= rOrder; p++) {
        kernel_recurse_order_p<<<blocks, threads>>>(
            p,
            n,
            numEtaDerivs,
            strideP,
            d_keys,
            d_poles,
            d_polesOffset,
            d_htildeCoeffs,
            d_hCoeffs
        );
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            cudaFree(d_keys);
            cudaFree(d_polesOffset);
            if (d_poles) cudaFree(d_poles);
            cudaFree(d_htildeCoeffs);
            cudaFree(d_hCoeffs);
            return 22;
        }
    }

    err = cudaMemcpy(h_out_hcoeffs, d_hCoeffs, hCoeffsSize, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cudaFree(d_keys);
        cudaFree(d_polesOffset);
        if (d_poles) cudaFree(d_poles);
        cudaFree(d_htildeCoeffs);
        cudaFree(d_hCoeffs);
        return 23;
    }

    cudaFree(d_keys);
    cudaFree(d_polesOffset);
    if (d_poles) cudaFree(d_poles);
    cudaFree(d_htildeCoeffs);
    cudaFree(d_hCoeffs);
    return 0;
}