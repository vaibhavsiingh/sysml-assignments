%%writefile gemm.cu

#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>

int main(int argc, char *argv[]){
    int N = atoi(argv[1]), M = atoi(argv[2]), K = atoi(argv[3]);
    float* A, *B, *C;

    A = (float*)malloc(N*M*sizeof(float));
    B = (float*)malloc(K*M*sizeof(float));
    C = (float*)malloc(N*K*sizeof(float));

    for(int i=0; i<N; i++){
        for(int j=0; j<M; j++){
            A[i*M+j] = i+j;            
        }
    }

    for(int i=0; i<M; i++){
        for(int j=0; j<K; j++){
            B[i*K+j] = i+j;            
        }
    }

    float* da, *db, *dc;

    cudaMalloc((void **)&da, N*M*sizeof(float));
    cudaMalloc((void **)&db, K*M*sizeof(float));
    cudaMalloc((void **)&dc, N*K*sizeof(float));

    cudaMemcpy(da, A, N*M*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(db, B, K*M*sizeof(float), cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0;
    float beta = 1.0;

    cublasSgemm(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        K, N, M,
        &alpha,
        db, K,
        da, M,
        &beta,
        dc, K
    );

    cudaMemcpy(C, dc, N*K*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    free(A); free(B); free(C);
}

