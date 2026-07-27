#include <iostream>
using namespace std;

__global__
void relu_kernel(const float *A, float *B, int N, int M){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N*M){
        B[idx] = (A[idx] > 0) ? A[idx] : 0;
    }
}
int main() {
    size_t size = 5*sizeof(float);
    float *X = (float *)malloc(size);
    float *Y = (float *)malloc(size);
    float *dX, *dY;
    cudaMalloc((void **)&dX, size);
    cudaMalloc((void **)&dY, size);

    X[0] = -134.2;
    X[1] = 1234.2;
    X[2] = -2;
    X[3] = 1.2;
    X[4] = 0;

    cudaMemcpy(dX, X, size, cudaMemcpyHostToDevice);
    relu_kernel<<<2,3>>> (dX, dY, 5,1);
    cudaMemcpy(Y, dY, size, cudaMemcpyDeviceToHost);
    for(int i=0; i<5; i++){
        cout << Y[i] << ' ';
    }
}