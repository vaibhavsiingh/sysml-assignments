// %%writefile test.cu
#include<iostream>
#include<cuda_runtime.h>
using namespace std;

#define TILE 16

// does C[N,K] = A[N,M]*B[M,K] + bias[K]
__global__
void matmul_kernel(const float *A, const float *B, const float *bias, float *C, int N, int M, int K){
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
    if(row < N && col < K) C[row*K + col] = sm + bias[col];
}


__global__
void matmul_kernel_transpose_at_A_kernel(const float *A, const float *B, float *C, int N, int M, int K){
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    int col = threadIdx.x + blockDim.x * blockIdx.x;

    float sum = 0;
    if(row < M && col < K){
        for(int idx=0; idx < N; idx++){
            sum += A[idx * M + row] * B[idx * K + col]; 
        }
        C[row * K + col] = sum;
    }
}

// for transpose shit
// A is N * M
// B is N * K
// C will be M * K
void matmul_kernel_transpose_at_A(const float *A, const float *B, float *C, int N, int M, int K)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocks((K + 15)/ 16, (M + 15)/ 16);
    matmul_kernel_transpose_at_A_kernel<<<blocks,threadsPerBlock>>> (A, B, C, N, M, K);   
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Sync error: %s\n", cudaGetErrorString(err));
    }
}



__global__
void matmul_kernel_transpose_at_B_kernel(const float *A, const float *B, float *C, int N, int M, int K){
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    int col = threadIdx.x + blockDim.x * blockIdx.x;

    float sum = 0;
    if(row < N && col < K){
        for(int idx=0; idx < M; idx++){
            sum += A[row * M + idx] * B[idx * K + col];
        }
        C[row * K + col] = sum;
    }
}

// for transpose shit
// A is N * M
// B is K * M
// C will be N * K
void matmul_kernel_transpose_at_B(const float *A, const float *B, float *C, int N, int M, int K)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocks((K + 15)/ 16, (N + 15)/ 16);
    matmul_kernel_transpose_at_B_kernel<<<blocks,threadsPerBlock>>> (A, B, C, N, M, K);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Sync error: %s\n", cudaGetErrorString(err));
    }
}

// expect device pointers
void matmul_with_bias(float* X, float* W, float* b, float* Z, int N, int M, int K){
    dim3 threadsPerBlock(TILE, TILE);
    dim3 blocks((K + TILE - 1)/TILE, (N + TILE - 1)/TILE);

    matmul_kernel<<<blocks, threadsPerBlock>>> (X, W, b, Z, N, M, K);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Sync error: %s\n", cudaGetErrorString(err));
    }
}

__global__
void relu_kernel(const float *A, float *B, int N, int M){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N*M){
        B[idx] = (A[idx] > 0) ? A[idx] : 0;
    }
}



// expect device pointers
void relu(float *X, float* A, int N, int M){
    int threadsPerBlock = 256;
    int numBlocks = (N*M+threadsPerBlock-1)/threadsPerBlock;

    relu_kernel<<<numBlocks, threadsPerBlock>>> (X, A, N, M);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Sync error: %s\n", cudaGetErrorString(err));
    }
}

__global__
void filter_copy_kernel(const float *Z, const float* dA, float *dZ, int N, int M)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N * M && Z[idx] > 0)
    {
        dZ[idx] = dA[idx];
    }
    else
        dZ[idx] = 0;
}

// if Z > 0
// copy dA to dZ
// else 0
void filter_copy(const float *Z, const float* dA, float *dZ, int N, int M)
{
    int threadsPerBlock = 256;
    int numBlocks = (N*M+threadsPerBlock-1)/threadsPerBlock;

    filter_copy_kernel<<<numBlocks, threadsPerBlock>>> (Z, dA, dZ, N, M);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Sync error: %s\n", cudaGetErrorString(err));
    }
}

__global__
void bias_grad_kernel(float *dZ, float *db, int N, int M)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    if(col < M)
    {
        for(int i=0; i<N; i++) sum += dZ[i*M + col];
        db[col] = sum;
    }
}

void calculate_bias_grad(float *dZ, float *db, int N, int M)
{
    int threadsPerBlock = 256;
    int numBlocks = (M + threadsPerBlock -1)/ threadsPerBlock;
    bias_grad_kernel<<<numBlocks, threadsPerBlock>>> (dZ, db, N, M);
}

// call with blockSize.x <= 256
__global__
void loss_kernel(float *Z4, float *y, float *loss, float *dZ4, int N, int OUT){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float diff = 0;
    int total_ele = N * OUT;
    if(idx < total_ele){
        diff = Z4[idx] - y[idx];
        dZ4[idx] = (2 * diff) / total_ele;
        diff *= diff;
    }
    __shared__ float smem[256];
    float val = (idx < total_ele) ? diff : 0;
    smem[threadIdx.x] = val;
    __syncthreads();

    int size = 256;
    while(size > 1){
        size /= 2;
        if(threadIdx.x < size) smem[threadIdx.x] += smem[threadIdx.x + size];
        __syncthreads();
    }

    if(threadIdx.x == 0){
        atomicAdd(loss, smem[0]);
    }
}

// expects device pointers
float calculate_loss(float *y_pred, float *y_true, float *dZ4,int N, int OUT){
    float sum = 0;
    float *dSum;
    cudaMalloc((void **)&dSum, sizeof(float));
    cudaMemcpy(dSum, &sum, sizeof(float), cudaMemcpyHostToDevice);

    int totalEle = N * OUT;
    int threadsPerBlock = 256;
    int numBlocks = (totalEle + threadsPerBlock -1)/threadsPerBlock;

    
    loss_kernel<<<numBlocks, threadsPerBlock>>> (y_pred, y_true, dSum, dZ4, N, OUT);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Sync error: %s\n", cudaGetErrorString(err));
    }
    cudaMemcpy(&sum, dSum, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dSum);
    return sum / totalEle;
}


__global__
void gradient_descent_kernel(float *A, float *dA, int N, int M, float eta)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < N * M)
    {
        A[idx] = A[idx] - eta * dA[idx];
    }
}

void gradient_descent(float *A, float *dA, int N, int M, float eta)
{   
    int threadsPerBlock = 256;
    int numBlocks = (N*M+threadsPerBlock-1)/threadsPerBlock;

    gradient_descent_kernel<<<numBlocks, threadsPerBlock>>> (A, dA, N, M, eta);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Sync error: %s\n", cudaGetErrorString(err));
    }
}

void forward_pass(
    float *X, int N, int IN,
    float* W1, int H1, float *b1,
    float* W2, int H2, float *b2,
    float* W3, int H3, float *b3,
    float* W4, int OUT, float *b4,
    float* A2_checkpoint,
    float* Z4_out
)
{
    float *Z,  *A;
    int sz_A1 = N*H1*sizeof(float), sz_A3 = N*H3*sizeof(float);
    int sz_Z1 = sz_A1, sz_Z2 = N*H2*sizeof(float), sz_Z3 = sz_A3;
    int sz_A = max(sz_A1, sz_A3);
    int sz_Z = max(sz_Z1, max(sz_Z2, sz_Z3));

    cudaMalloc((void **)&A, sz_A);
    cudaMalloc((void **)&Z, sz_Z);    

    matmul_with_bias(X, W1, b1, Z, N, IN, H1);
    relu(Z, A, N, H1);
    matmul_with_bias(A, W2, b2, Z, N, H1, H2);
    relu(Z, A2_checkpoint, N, H2);
    matmul_with_bias(A2_checkpoint, W3, b3, Z, N, H2, H3);
    relu(Z, A, N, H3);
    matmul_with_bias(A, W4, b4, Z4_out, N, H3, OUT);
    

    cudaFree(Z);
    cudaFree(A); 
}

void backward_pass(
    float *X, float* Y, float *Z4, float *A2,
    float *db4,  float *db3, float *db2,  float *db1,
    float *W4, float *W3, float *W2, float *W1,
    float *dW4, float *dW3, float *dW2, float *dW1,
    float *b4, float *b3, float *b2, float *b1,
    int N, int IN, int H1, int H2, int H3, int OUT
)
{ 

    int sz_A1 = N*H1*sizeof(float), sz_A3 = N*H3*sizeof(float);
    int sz_Z1 = sz_A1, sz_Z2 = N*H2*sizeof(float), sz_Z3 = sz_A3;
    int sz_A = max(sz_A1, sz_A3);
    int sz_Z = max(sz_Z1, max(sz_Z2, sz_Z3));
    
    float *Z,  *A;
    cudaMalloc((void **)&A, sz_A);
    cudaMalloc((void **)&Z, sz_Z);   

    float *dZ, *dA;
    cudaMalloc((void **)&dA, sz_A);
    cudaMalloc((void **)&dZ, sz_Z);

    float loss = calculate_loss(Z4, Y, dZ, N, OUT);     // dZ <- dZ4       

    // layer 4
    // calculating Z3, A3
    matmul_with_bias(A2, W3, b3, Z, N, H2, H3); // Z <- Z3
    relu(Z, A, N, H3);  // calculating A3

    matmul_kernel_transpose_at_A(A, dZ, dW4, N, H3, OUT);
    calculate_bias_grad(dZ, db4, N, OUT);
    matmul_kernel_transpose_at_B(dZ, W4, dA, N, OUT, H3); // dA <- dA3

    // layer 3
    filter_copy(Z, dA, dZ, N, H3); // dZ <- dZ3
    matmul_kernel_transpose_at_A(A2, dZ, dW3, N, H2, H3);
    calculate_bias_grad(dZ, db3, N, H3);
    matmul_kernel_transpose_at_B(dZ, W3, dA, N, H3, H2); // dA <- dA2

    // layer 2
    // calculating  Z2, A2
    matmul_with_bias(X, W1, b1, Z, N, IN, H1); // Z <- Z1
    relu(Z, A, N, H1); // A <- A1
    matmul_with_bias(A, W2, b2, Z, N, H1, H2); // Z  <- Z2    

    filter_copy(Z, dA, dZ, N, H2);  // dZ <- dZ2
    matmul_kernel_transpose_at_A(A, dZ, dW2, N, H1, H2); 
    calculate_bias_grad(dZ, db2, N, H2);
    matmul_kernel_transpose_at_B(dZ, W2, dA, N, H2, H1); // dA <- dA1

    // layer 1
    // calculating Z1
    matmul_with_bias(X, W1, b1, Z, N, IN, H1); // Z <- Z1

    filter_copy(Z, dA, dZ, N, H1); // dZ <- dZ1
    matmul_kernel_transpose_at_A(X, dZ, dW1, N, IN, H1);
    calculate_bias_grad(dZ, db1, N, H1);

    cudaFree(A);
    cudaFree(Z);   
    cudaFree(dA);
    cudaFree(dZ);
     
}

// expects device pointers
void train(float *Xs, float *Ys, int N_samples, int IN,
          float* W1, int H1, float *b1,
          float* W2, int H2, float *b2,
          float* W3, int H3, float *b3,
          float* W4, int OUT, float *b4, int epochs = 10, float eta = 0.02)
{
    int N = 1000;
    float *A2, *Z4;
    
    
    
    cudaMalloc((void **)&A2, N*H2*sizeof(float));        
    cudaMalloc((void **)&Z4, N*OUT*sizeof(float));

    float *dW1, *db1, *dW2, *db2, *dW3, *db3, *dW4, *db4;
    cudaMalloc((void **)&dW1, IN*H1*sizeof(float)); 
    cudaMalloc((void **)&dW2, H1*H2*sizeof(float)); 
    cudaMalloc((void **)&dW3, H2*H3*sizeof(float)); 
    cudaMalloc((void **)&dW4, H3*OUT*sizeof(float)); 
    cudaMalloc((void **)&db1, H1*sizeof(float)); 
    cudaMalloc((void **)&db2, H2*sizeof(float)); 
    cudaMalloc((void **)&db3, H3*sizeof(float)); 
    cudaMalloc((void **)&db4, OUT*sizeof(float));
         

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float smForwardTimes = 0;
    float smBackwardTimes = 0;
    float frac = 0;

    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);

    size_t used_mem = total_mem - free_mem;

    cout << "Used GPU memory: " << used_mem / (1024.0 * 1024.0) << " MB\n";

    for(int e = 0; e<epochs; e++){
        if(e >= frac * (float) epochs){
            cout << "Epoch: " << e+1 << " / " << epochs << '\n';
            frac += 0.1;
        }
        for(int i=0; i < N_samples / N; i++){
            float ms = 0;
            float *X = Xs + i * N * IN;
            float *Y = Ys + i * N * OUT;
            // forward            

            cudaEventRecord(start);

            forward_pass(X, N, IN, W1, H1, b1, W2, H2, b2, W3, H3, b3, W4, OUT, b4, A2, Z4);
            
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&ms, start, stop);            
            smForwardTimes += ms;

            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("Kernel launch error: %s\n", cudaGetErrorString(err));
                break;
            }
            else{
                printf("cool cool cool");
            }

            // backward

            cudaEventRecord(start);
            
            backward_pass(X, Y, Z4, A2, db4, db3, db2, db1, W4, W3, W2, W1, dW4, dW3, dW2, dW1, b4, b3, b2, b1, N, IN, H1, H2, H3, OUT);
                                    
            gradient_descent(W1, dW1, IN, H1, eta);
            gradient_descent(W2, dW2, H1, H2, eta);
            gradient_descent(W3, dW3, H2, H3, eta);
            gradient_descent(W4, dW4, H3, OUT, eta);
            gradient_descent(b1, db1, 1, H1, eta);
            gradient_descent(b2, db2, 1, H2, eta);
            gradient_descent(b3, db3, 1, H3, eta);
            gradient_descent(b4, db4, 1, OUT, eta);
            
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            ms = 0;
            cudaEventElapsedTime(&ms, start, stop);
            smBackwardTimes += ms;

             err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("Kernel launch error: %s\n", cudaGetErrorString(err));
                break;
            }

        }
    }

    int iterPerEpoch =  N_samples / N;
    cout << "Avg. forward time = " << smForwardTimes / (epochs * iterPerEpoch) << '\n';
    cout << "Avg. backward time = " << smBackwardTimes / (epochs * iterPerEpoch) << '\n';
    
    cudaFree(Z4);    
    cudaFree(A2);
     
    
    cudaFree(dW1);
    cudaFree(dW2);
    cudaFree(dW3);
    cudaFree(dW4);
    cudaFree(db1);
    cudaFree(db2);
    cudaFree(db3); 
    cudaFree(db4);    

}


int main(){
    int N = 10000;
    int IN = 2000;
    int H1 = 4000;
    int H2 = 6000;
    int H3 = 3000;
    int OUT = 500;

    float *Xs_cpu = (float *)malloc(N*IN*sizeof(float));
    float *Ys_cpu = (float *)malloc(N*OUT*sizeof(float));
    
    float *W1_cpu = (float *)malloc(IN*H1*sizeof(float));
    float *W2_cpu = (float *)malloc(H1*H2*sizeof(float));
    float *W3_cpu = (float *)malloc(H2*H3*sizeof(float));
    float *W4_cpu = (float *)malloc(H3*OUT*sizeof(float));

    float *b1_cpu = (float *)malloc(H1*sizeof(float));
    float *b2_cpu = (float *)malloc(H2*sizeof(float));
    float *b3_cpu = (float *)malloc(H3*sizeof(float));
    float *b4_cpu = (float *)malloc(OUT*sizeof(float));

    float *Xs, *Ys, *W1, *b1, *W2, *b2, *W3, *b3, *W4, *b4;
    cudaMalloc((void **)&Xs, N*IN*sizeof(float)); 
    cudaMalloc((void **)&Ys, N*OUT*sizeof(float)); 

    cudaMalloc((void **)&W1, IN*H1*sizeof(float)); 
    cudaMalloc((void **)&W2, H1*H2*sizeof(float)); 
    cudaMalloc((void **)&W3, H2*H3*sizeof(float)); 
    cudaMalloc((void **)&W4, H3*OUT*sizeof(float)); 

    cudaMalloc((void **)&b1, H1*sizeof(float)); 
    cudaMalloc((void **)&b2, H2*sizeof(float)); 
    cudaMalloc((void **)&b3, H3*sizeof(float)); 
    cudaMalloc((void **)&b4, OUT*sizeof(float)); 

    
    cudaMemcpy(Xs, Xs_cpu, N*IN*sizeof(float), cudaMemcpyHostToDevice); 
    cudaMemcpy(Ys, Ys_cpu, N*OUT*sizeof(float), cudaMemcpyHostToDevice); 

    cudaMemcpy(W1, W1_cpu, IN*H1*sizeof(float), cudaMemcpyHostToDevice); 
    cudaMemcpy(W2, W2_cpu, H1*H2*sizeof(float), cudaMemcpyHostToDevice); 
    cudaMemcpy(W3, W3_cpu, H2*H3*sizeof(float), cudaMemcpyHostToDevice); 
    cudaMemcpy(W4, W4_cpu, H3*OUT*sizeof(float), cudaMemcpyHostToDevice); 

    cudaMemcpy(b1, b1_cpu, H1*sizeof(float), cudaMemcpyHostToDevice); 
    cudaMemcpy(b2, b2_cpu, H2*sizeof(float), cudaMemcpyHostToDevice); 
    cudaMemcpy(b3, b3_cpu, H3*sizeof(float), cudaMemcpyHostToDevice); 
    cudaMemcpy(b4, b4_cpu, OUT*sizeof(float), cudaMemcpyHostToDevice); 

    train(Xs, Ys, N, IN, W1, H1, b1, W2, H2, b2, W3, H3, b3, W4, OUT, b4, 2);

    cudaFree(Xs);
    cudaFree(Ys);
    cudaFree(W1);
    cudaFree(W2);
    cudaFree(W3);
    cudaFree(W4);
    cudaFree(b1);
    cudaFree(b2);
    cudaFree(b3);
    cudaFree(b4);

    free(Xs_cpu);
    free(Ys_cpu);
    free(W1_cpu);
    free(W2_cpu);
    free(W3_cpu);
    free(W4_cpu);
    free(b1_cpu);
    free(b2_cpu);
    free(b3_cpu);
    free(b4_cpu);
}