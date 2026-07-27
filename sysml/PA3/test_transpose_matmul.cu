#include <iostream>
#include<cuda_runtime.h>
using namespace std;

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
}

int main() {
    int N = 2, M = 3, K = 4;
    float *X = (float *)malloc(N * M * sizeof(float));
    float *Y = (float *)malloc(N * M *sizeof(float));
    float *Z = (float *)malloc(N * M * sizeof(float));
    float *dX, *dY, *dZ;
    cudaMalloc((void **)&dX, N * M * sizeof(float));
    cudaMalloc((void **)&dY, N * M   * sizeof(float));
    cudaMalloc((void **)&dZ, N * M  * sizeof(float));

    for(int i=0; i<N*M; i+=1){
        X[i] = i;
    }

    for(int i=0; i<N * M ; i+=1){
        Y[i] = 5-i;
    }

    for(int i=0; i<N; i++){
        for(int j=0; j<M; j++)
            cout << X[i*M + j] << ' ';
        cout << endl;
    }

    for(int i=0; i<N; i++){
        for(int j=0; j<M; j++)
            cout << Y[i* M + j] << ' ';
        cout << endl;
    }

    cudaMemcpy(dX, X, N * M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dY, Y, N * M  * sizeof(float), cudaMemcpyHostToDevice);

    gradient_descent(dX, dY, N, M, 0.02);

    cudaMemcpy(X, dX, N * M * sizeof(float), cudaMemcpyDeviceToHost);
    for(int i=0; i<N; i++){
        for(int j=0; j<M; j++)
            cout << X[i*M + j] << ' ';
        cout << endl;
    }

    cudaFree(dX);
    cudaFree(dY);
    cudaFree(dZ);

    free(X);
    free(Y);
    free(Z);
}