// iterative_update_timed_pinned_granular.cu
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CHECK_CUDA(call) do { cudaError_t err = (call); \
    if (err != cudaSuccess) { fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(err)); return 1; } } while(0)
#define CHECK_CUBLAS(call) do { cublasStatus_t st = (call); \
    if (st != CUBLAS_STATUS_SUCCESS) { fprintf(stderr,"cuBLAS error %s:%d: %d\n",__FILE__,__LINE__,st); return 2; } } while(0)

typedef struct { int N; double Delta; int Ell; double C; int Idx; } PolesData;
typedef struct { int Ell; double Delta; } KeyData;

// ---------------- KERNELS ----------------
__global__ void kernel_add_columns(double *dgNew,const double *colUpdates,int m,int n){
    int tid=blockIdx.x*blockDim.x+threadIdx.x;
    int total=m*n;
    for(int idx=tid;idx<total;idx+=blockDim.x*gridDim.x) dgNew[idx]+=colUpdates[idx];
}

__global__ void kernel_rel_diff_max(const double *dgNew,const double *dgOld,double *blockMax,int m,int n,double eps_replace_zero){
    extern __shared__ double sdata[];
    int tid=threadIdx.x; int idx=blockIdx.x*blockDim.x+tid; int total=m*n;
    double local_max=0.0;
    for(int i=idx;i<total;i+=blockDim.x*gridDim.x){
        double oldv=dgOld[i], newv=dgNew[i];
        double denom=fabs(oldv)>0?fabs(oldv):eps_replace_zero;
        double rel=fabs(newv-oldv)/denom;
        if(rel>local_max) local_max=rel;
    }
    sdata[tid]=local_max; __syncthreads();
    for(unsigned int s=blockDim.x/2;s>0;s>>=1){ if(tid<s) if(sdata[tid+s]>sdata[tid]) sdata[tid]=sdata[tid+s]; __syncthreads(); }
    if(tid==0) blockMax[blockIdx.x]=sdata[0];
}

static double host_reduce_blockmax(double *d_blockMax,int blocks){
    double *h=(double*)malloc(sizeof(double)*blocks);
    if(!h) return -1.0;
    cudaMemcpy(h,d_blockMax,sizeof(double)*blocks,cudaMemcpyDeviceToHost);
    double mval=0.0; for(int i=0;i<blocks;i++) if(h[i]>mval)mval=h[i]; free(h); return mval;
}

__global__ void kernel_scale_accumulate(double *d_colUpdates,const double *d_tempVecAll,const double *d_scales,const PolesData *d_poles,const int *d_polesOffset,int m,int n,int totalPoles){
    int idx=blockIdx.x*blockDim.x+threadIdx.x; if(idx>=totalPoles*m)return;
    int poleIdx=idx/m,row=idx%m;
    double temp=d_tempVecAll[poleIdx*m+row]*d_scales[poleIdx];
    int col=0; for(int j=0;j<n;j++){ if(poleIdx>=d_polesOffset[j] && poleIdx<d_polesOffset[j+1]){ col=j; break; } }
    atomicAdd(&d_colUpdates[row+col*m],temp);
}

// ---------------- MAIN FUNCTION ----------------
extern "C"
int iterative_update_gpu(
    int m, int n,
    const double* h_dg,
    const double* h_dgTilde,
    int maxIterations,
    double tol,
    const KeyData* h_keys,
    const PolesData* h_poles,
    const int* polesOffset,
    const double** h_R_ptrs,
    double* h_out_dg,
    int* converged
) {
    int status = 0;
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // ---------------- TIMING ACCUMULATORS ----------------
    float totalTimeCopy_dg = 0;
    float totalTimeCopy_dgTilde = 0;
    float totalTimeCopy_R = 0;
    float totalTimeCopy_GEMV_prep = 0;
    float totalTimeReset = 0, totalTimeGEMV = 0, totalTimeScale = 0, totalTimeAdd = 0, totalTimeDiff = 0;

    // ---------------- DEVICE BUFFERS ----------------
    double *d_dgOld=NULL,*d_dgNew=NULL,*d_colUpdates=NULL,*d_tempVec=NULL,*d_blockMax=NULL;
    size_t mat_size = (size_t)m*(size_t)n*sizeof(double);
    size_t vec_m = (size_t)m*sizeof(double);
    CHECK_CUDA(cudaMalloc(&d_dgOld, mat_size));
    CHECK_CUDA(cudaMalloc(&d_dgNew, mat_size));
    CHECK_CUDA(cudaMalloc(&d_colUpdates, mat_size));
    CHECK_CUDA(cudaMalloc(&d_tempVec, vec_m));

    // ---------------- COPY DG ----------------
    CHECK_CUDA(cudaEventRecord(start));
    CHECK_CUDA(cudaMemcpy(d_dgOld, h_dg, mat_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms_copy_dg = 0; CHECK_CUDA(cudaEventElapsedTime(&ms_copy_dg, start, stop));
    totalTimeCopy_dg += ms_copy_dg;

    // ---------------- COPY DG TILDE ----------------
    double *d_dgTilde_dev=NULL;
    CHECK_CUDA(cudaMalloc(&d_dgTilde_dev, mat_size));
    CHECK_CUDA(cudaEventRecord(start));
    CHECK_CUDA(cudaMemcpy(d_dgTilde_dev, h_dgTilde, mat_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms_copy_dgTilde = 0; CHECK_CUDA(cudaEventElapsedTime(&ms_copy_dgTilde, start, stop));
    totalTimeCopy_dgTilde += ms_copy_dgTilde;

    // ---------------- COPY R MATRICES ----------------
    int totalPoles = polesOffset[n];
    double **d_R_ptrs = NULL;
    CHECK_CUDA(cudaMallocHost((void**)&d_R_ptrs, sizeof(double*)*totalPoles));
    for(int p=0;p<totalPoles;p++){
        double *dR=NULL;
        CHECK_CUDA(cudaMalloc(&dR, (size_t)m*(size_t)m*sizeof(double)));
        CHECK_CUDA(cudaEventRecord(start));
        CHECK_CUDA(cudaMemcpy(dR, h_R_ptrs[p], (size_t)m*(size_t)m*sizeof(double), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float ms_copy_R=0; CHECK_CUDA(cudaEventElapsedTime(&ms_copy_R,start,stop));
        totalTimeCopy_R += ms_copy_R;
        d_R_ptrs[p]=dR;
    }

    double **d_R_ptrs_dev=NULL;
    CHECK_CUDA(cudaMalloc(&d_R_ptrs_dev,sizeof(double*)*totalPoles));
    CHECK_CUDA(cudaMemcpy(d_R_ptrs_dev,d_R_ptrs,sizeof(double*)*totalPoles,cudaMemcpyHostToDevice));

    // ---------------- BLOCKS AND THREADS ----------------
    int threads=256;
    int blocks=(m*n+threads-1)/threads; if(blocks>1024) blocks=1024;
    CHECK_CUDA(cudaMalloc(&d_blockMax,sizeof(double)*blocks));

    // ---------------- PER-ITERATION DEVICE BUFFERS ----------------
    double *d_tempVecAll=NULL;
    CHECK_CUDA(cudaMalloc(&d_tempVecAll,sizeof(double)*m*totalPoles));
    const double **dR_ptrs_dev_iter=NULL, **dX_ptrs_dev=NULL;
    double **dY_ptrs_dev=NULL;
    CHECK_CUDA(cudaMalloc(&dR_ptrs_dev_iter,totalPoles*sizeof(double*)));
    CHECK_CUDA(cudaMalloc(&dX_ptrs_dev,totalPoles*sizeof(double*)));
    CHECK_CUDA(cudaMalloc(&dY_ptrs_dev,totalPoles*sizeof(double*)));
    double *d_scales=NULL;
    PolesData *d_poles=NULL;
    int *d_polesOffset=NULL;
    CHECK_CUDA(cudaMalloc(&d_scales,totalPoles*sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_poles,totalPoles*sizeof(PolesData)));
    CHECK_CUDA(cudaMalloc(&d_polesOffset,(n+1)*sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_poles,h_poles,totalPoles*sizeof(PolesData),cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_polesOffset,polesOffset,(n+1)*sizeof(int),cudaMemcpyHostToDevice));

    // ---------------- ITERATIONS ----------------
    int converged_flag = 0;
    for(int iter=0;iter<maxIterations;iter++){
        // ---------------- RESET ----------------
        CHECK_CUDA(cudaEventRecord(start));
        CHECK_CUDA(cudaMemset(d_colUpdates,0,mat_size));
        CHECK_CUDA(cudaMemcpy(d_dgNew,d_dgTilde_dev,mat_size,cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float ms_reset=0; CHECK_CUDA(cudaEventElapsedTime(&ms_reset,start,stop));
        totalTimeReset += ms_reset;

        // ---------------- PREP GEMV ----------------
        const double **h_dR_array=(const double**)malloc(totalPoles*sizeof(double*));
        const double **h_dX_array=(const double**)malloc(totalPoles*sizeof(double*));
        double **h_dY_array=(double**)malloc(totalPoles*sizeof(double*));
        double *h_scales_host=(double*)malloc(totalPoles*sizeof(double));

        for(int j=0;j<n;j++){
            int start_idx=polesOffset[j], end_idx=polesOffset[j+1];
            double keyDelta=h_keys[j].Delta;
            for(int k=start_idx;k<end_idx;k++){
                const PolesData *pole=&h_poles[k];
                h_dR_array[k]=d_R_ptrs[k];
                h_dX_array[k]=d_dgOld+(size_t)pole->Idx*(size_t)m;
                h_dY_array[k]=d_tempVecAll+k*m;
                double deltaDiff=keyDelta-pole->Delta;
                h_scales_host[k]=(deltaDiff!=0.0)?pole->C/deltaDiff:0.0;
            }
        }

        CHECK_CUDA(cudaEventRecord(start));
        CHECK_CUDA(cudaMemcpy(dR_ptrs_dev_iter,h_dR_array,totalPoles*sizeof(double*),cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(dX_ptrs_dev,h_dX_array,totalPoles*sizeof(double*),cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(dY_ptrs_dev,h_dY_array,totalPoles*sizeof(double*),cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_scales,h_scales_host,totalPoles*sizeof(double),cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float ms_gemv_prep=0; CHECK_CUDA(cudaEventElapsedTime(&ms_gemv_prep,start,stop));
        totalTimeCopy_GEMV_prep += ms_gemv_prep;

        free(h_dR_array); free(h_dX_array); free(h_dY_array); free(h_scales_host);

        // ---------------- GEMV ----------------
        CHECK_CUDA(cudaEventRecord(start));
        double alpha=1.0,beta=0.0;
        CHECK_CUBLAS(cublasDgemvBatched(handle,CUBLAS_OP_N,m,m,&alpha,
                                        dR_ptrs_dev_iter,m,dX_ptrs_dev,1,&beta,
                                        dY_ptrs_dev,1,totalPoles));
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float ms_gemv=0; CHECK_CUDA(cudaEventElapsedTime(&ms_gemv,start,stop));
        totalTimeGEMV += ms_gemv;

        // ---------------- SCALE + ACCUMULATE ----------------
        CHECK_CUDA(cudaEventRecord(start));
        int blocks_acc=(totalPoles*m+threads-1)/threads;
        kernel_scale_accumulate<<<blocks_acc,threads>>>(d_colUpdates,d_tempVecAll,d_scales,d_poles,d_polesOffset,m,n,totalPoles);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float ms_scale=0; CHECK_CUDA(cudaEventElapsedTime(&ms_scale,start,stop));
        totalTimeScale += ms_scale;

        // ---------------- ADD COLUMNS ----------------
        CHECK_CUDA(cudaEventRecord(start));
        kernel_add_columns<<<blocks,threads>>>(d_dgNew,d_colUpdates,m,n);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float ms_add=0; CHECK_CUDA(cudaEventElapsedTime(&ms_add,start,stop));
        totalTimeAdd += ms_add;

        // ---------------- RELATIVE DIFF ----------------
        CHECK_CUDA(cudaEventRecord(start));
        kernel_rel_diff_max<<<blocks,threads,threads*sizeof(double)>>>(d_dgNew,d_dgOld,d_blockMax,m,n,1e-12);
        CHECK_CUDA(cudaGetLastError());
        double maxDiff=host_reduce_blockmax(d_blockMax,blocks);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float ms_diff=0; CHECK_CUDA(cudaEventElapsedTime(&ms_diff,start,stop));
        totalTimeDiff += ms_diff;

        if(maxDiff<tol){
            converged_flag = 1;
            CHECK_CUDA(cudaMemcpy(h_out_dg,d_dgNew,mat_size,cudaMemcpyDeviceToHost));
            break;
        }

        // Swap
        double *tmp=d_dgOld; d_dgOld=d_dgNew; d_dgNew=tmp;
    }

    // ---------------- FINAL COPY ----------------
    if (!converged_flag) {
        CHECK_CUDA(cudaMemcpy(h_out_dg,d_dgOld,mat_size,cudaMemcpyDeviceToHost));
    }
    
    // Set convergence output
    if (converged) {
        *converged = converged_flag;
    }

    // ---------------- PRINT TIMINGS ----------------
    printf("CPU->GPU copy timings (ms):\n");
    printf("  dg:        %.3f\n", totalTimeCopy_dg);
    printf("  dgTilde:   %.3f\n", totalTimeCopy_dgTilde);
    printf("  R matrices:%.3f\n", totalTimeCopy_R);
    printf("  GEMV prep: %.3f\n", totalTimeCopy_GEMV_prep);
    printf("  Total CPU->GPU: %.3f\n",
           totalTimeCopy_dg+totalTimeCopy_dgTilde+totalTimeCopy_R+totalTimeCopy_GEMV_prep);
    printf("Kernel timings total / per-iteration average (ms):\n");
    printf("  Reset + dgTilde copy: %.3f / %.3f\n", totalTimeReset, totalTimeReset/maxIterations);
    printf("  Batched GEMV: %.3f / %.3f\n", totalTimeGEMV, totalTimeGEMV/maxIterations);
    printf("  Scale & accumulate: %.3f / %.3f\n", totalTimeScale, totalTimeScale/maxIterations);
    printf("  Add columns: %.3f / %.3f\n", totalTimeAdd, totalTimeAdd/maxIterations);
    printf("  Relative diff: %.3f / %.3f\n", totalTimeDiff, totalTimeDiff/maxIterations);

    // ---------------- CLEANUP ----------------
cleanup:
    for(int p=0;p<totalPoles;p++) if(d_R_ptrs[p]) cudaFree(d_R_ptrs[p]);
    if(d_R_ptrs) cudaFreeHost(d_R_ptrs);
    if(d_R_ptrs_dev) cudaFree(d_R_ptrs_dev);
    if(d_dgOld) cudaFree(d_dgOld); if(d_dgNew) cudaFree(d_dgNew);
    if(d_colUpdates) cudaFree(d_colUpdates); if(d_tempVec) cudaFree(d_tempVec);
    if(d_blockMax) cudaFree(d_blockMax);
    if(d_tempVecAll) cudaFree(d_tempVecAll);
    if(dR_ptrs_dev_iter) cudaFree(dR_ptrs_dev_iter);
    if(dX_ptrs_dev) cudaFree(dX_ptrs_dev);
    if(dY_ptrs_dev) cudaFree(dY_ptrs_dev);
    if(d_scales) cudaFree(d_scales);
    if(d_poles) cudaFree(d_poles);
    if(d_polesOffset) cudaFree(d_polesOffset);
    if(d_dgTilde_dev) cudaFree(d_dgTilde_dev);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    cublasDestroy(handle);

    return status;
}

