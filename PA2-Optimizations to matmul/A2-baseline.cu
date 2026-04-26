%%writefile naive.cu

#include <iostream>

__global__ void matmul(const float* A, const float *B, float * C, int N, int M, int K){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i < N && j < K){
        float sm  = 0;        
        for(int ptr=0; ptr<M; ptr++){
            sm += A[i*M + ptr] * B[ptr*K + j];
        }
        C[i*K +j] = sm;
    }
}

int main(int argc, char *argv[]){
    if(argc <4){
        return 1;
    }
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

    
    dim3 dimBlock(32,32);
    dim3 dimGrid((N + dimBlock.x - 1)/dimBlock.x, (K + dimBlock.y - 1)/dimBlock.y);

    matmul<<<dimGrid, dimBlock>>> (da,db,dc,N,M,K);

    cudaMemcpy(C, dc, N*K*sizeof(float), cudaMemcpyDeviceToHost);


    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    free(A);free(B);free(C);
    
    
}

