#include <iostream>
#include <random>
using namespace std;

#define TILE 16
#define RELU_BLOCK_S 256

__global__ void matmul_kernel(const float *A, const float *B, float *C, int N, int M, int K){
    __shared__ float A_T[TILE][TILE];
    __shared__ float B_T[TILE][TILE];

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    float sm = 0;
    for(int j=0; j<(M+TILE-1)/TILE; j++){
        int idx = j*TILE + threadIdx.x;
        if(row < N && idx < M) A_T[threadIdx.y][threadIdx.x] = A[row*M + idx];
        else A_T[threadIdx.y][threadIdx.x] = 0;

        idx = j*TILE + threadIdx.y;
        if(idx < M && col < K) B_T[threadIdx.y][threadIdx.x] = B[idx*K + col];
        else B_T[threadIdx.y][threadIdx.x] = 0;

        __syncthreads();

        for(int t=0; t<TILE; t++) sm += A_T[threadIdx.y][t]*B_T[t][threadIdx.x];        

        __syncthreads();
    }
    if(row < N && col < K) C[row*K + col] = sm;
}


void matmul(const float *dA, const float *dB, float *dC, int N, int M, int K){

    dim3 threadsPerBlock(TILE,TILE);
    dim3 grid((K+TILE-1)/TILE, (N+TILE-1)/TILE);

    matmul_kernel<<<grid, threadsPerBlock>>> (dA, dB, dC, N, M, K);
    cudaDeviceSynchronize();    
}

__global__ void relu_kernel(float* A, int N, int M){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;    
    if(idx < M*N){
         A[idx] = (A[idx] > 0) ? A[idx] : 0;
    }
}

void relu(float *dA, int N, int M){
            
    dim3 threadsPerBlock(RELU_BLOCK_S);
    dim3 blocks((N*M+RELU_BLOCK_S -1)/RELU_BLOCK_S);
    relu_kernel<<<blocks, threadsPerBlock>>> (dA, N, M);
    cudaDeviceSynchronize();        
}


void run_mlp(int N){
    
    int B = 32;

    random_device rd;  
    mt19937 gen(rd());
    uniform_real_distribution<float> dist(-1.0, 1.0); 

    float* W1, *W2;

    int size = N*N*sizeof(float);
    W1 = (float *)malloc(size);
    W2 = (float *)malloc(size);

    for(int i=0; i<N*N; i++){
        W1[i] = dist(gen);
        W2[i] = dist(gen);
    }
        
    
    float* inp;
    int BN_size = B*N*sizeof(float);
    inp = (float *) malloc(BN_size);
    for(int i=0; i<B*N; i++){
        inp[i] = dist(gen);
    }

    float *dW1, *dW2, *din;
    cudaMalloc((void **)&dW1, size);
    cudaMalloc((void **)&dW2, size);
    cudaMalloc((void **)&din, BN_size);

    cudaMemcpy(dW1, W1, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dW2, W2, size, cudaMemcpyHostToDevice);
    cudaMemcpy(din, inp, BN_size, cudaMemcpyHostToDevice);

    float *dX1, *dZ;
    cudaMalloc((void **)&dX1, BN_size);
    cudaMalloc((void **)&dZ, BN_size);

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matmul(din, dW1, dX1, B, N, N);    
    relu(dX1, B, N);            
    matmul(dX1, dW2, dZ, B, N, N);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);

    float total_time;
    cudaEventElapsedTime(&total_time, start, stop);

    float *Z;
    Z = (float *)malloc(BN_size);
    cudaMemcpy(Z, dZ, BN_size, cudaMemcpyDeviceToHost);

    cout << "Total time: " << total_time << endl;
}

int main(){
    for(int N=32; N<=33000; N*=2){
        run_mlp(N);
    }
}