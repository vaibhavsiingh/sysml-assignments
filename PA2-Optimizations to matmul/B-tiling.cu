%%writefile tiling.cu

#include<iostream>

#define TILE_S 16

__global__ void matmul(float *A, float *B, float *C, int N, int M, int K){
    __shared__ float tile_A[TILE_S][TILE_S];    
    __shared__ float tile_B[TILE_S][TILE_S];    

    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    float sm = 0;
    for(int idx=0; idx<(M+TILE_S-1)/TILE_S; idx++){
        int j = idx*TILE_S + threadIdx.x;
        if(i<N && j<M) tile_A[threadIdx.y][threadIdx.x] = A[M*i+j];
        else tile_A[threadIdx.y][threadIdx.x] = 0;
        j = idx*TILE_S + threadIdx.y;
        if(j<M && k<K) tile_B[threadIdx.y][threadIdx.x] = B[K*j+k];
        else tile_B[threadIdx.y][threadIdx.x] = 0;

        __syncthreads();

        for(int t=0; t<TILE_S; t++) sm+= tile_A[threadIdx.y][t]*tile_B[t][threadIdx.x];

        __syncthreads();
    }

    if(i < N && k < K) C[i*K+k] = sm;
            
}

int main(int argc, char *argv[]){
    int N = atoi(argv[1]), M = atoi(argv[2]), K = atoi(argv[3]);
    float *A,*B,*C;

    A = (float*)malloc(N*M*sizeof(float));
    B = (float*)malloc(M*K*sizeof(float));
    C = (float*)malloc(N*K*sizeof(float));


    float *dA, *dB, *dC;
    cudaMalloc((void**)&dA, N*M*sizeof(float));
    cudaMalloc((void**)&dB, M*K*sizeof(float));
    cudaMalloc((void**)&dC, N*K*sizeof(float));

    cudaMemcpy(dA, A, N*M*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, M*K*sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(TILE_S,TILE_S);
    dim3 blocks((K+threadsPerBlock.x-1)/threadsPerBlock.x, (N+threadsPerBlock.y-1)/threadsPerBlock.y);

    matmul<<<blocks,threadsPerBlock>>> (dA, dB, dC, N, M , K);

    cudaMemcpy(C, dC, N*K*sizeof(float), cudaMemcpyDeviceToHost);

        
}
