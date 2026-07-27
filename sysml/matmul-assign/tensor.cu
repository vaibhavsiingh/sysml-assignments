%%writefile tensor.cu

#include <iostream>   // FIXED
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <chrono>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
using namespace nvcuda;

#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
} while(0)

__global__
void matmul_tensorcore(const half* A,
                       const half* B,
                       float* C,
                       int N)
{
    int warpId = (threadIdx.y * blockDim.x + threadIdx.x) / 32;

    int warpRow = blockIdx.y * (blockDim.y / 1) + warpId;
    int warpCol = blockIdx.x;

    if (warpRow * 16 >= N || warpCol * 16 >= N) return;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    for (int k = 0; k < N; k += 16) {

        const float* A_tile = A + (warpRow * 16) * N + k;
        const float* B_tile = B + k * N + (warpCol * 16);

        wmma::load_matrix_sync(a_frag, A_tile, N);
        wmma::load_matrix_sync(b_frag, B_tile, N);

        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    float* C_tile = C + (warpRow * 16) * N + (warpCol * 16);
    wmma::store_matrix_sync(C_tile, c_frag, N, wmma::mem_row_major);
}


void matmul_gpu_block2d(int N,
                        int TILE,
                        const float* A_h,
                        const float* B_h,
                        float* C_h)
{
    size_t bytes = (size_t)N * N * sizeof(float);

    half *A_d, *B_d;
    float *C_d;
    
    cudaMalloc(&A_d, bytes / 2);
    cudaMalloc(&B_d, bytes / 2);
    cudaMalloc(&C_d, bytes);

    std::vector<half> A_half(N*N), B_half(N*N);

    for (int i = 0; i < N*N; i++) {
        A_half[i] = __float2half(A_h[i]);
        B_half[i] = __float2half(B_h[i]);
    }

    CUDA_CHECK(cudaMemcpy(A_d, A_half.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_d, B_half.data(), bytes, cudaMemcpyHostToDevice));

    dim3 block(32, 1);
    dim3 grid((N + 16 - 1) / 16,
          (N + 16 - 1) / 16);

    matmul_tensorcore<<<grid, block>>>(A_d, B_d, C_d, N);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(C_h, C_d, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(B_d));
    CUDA_CHECK(cudaFree(C_d));
}


/* ================= CPU + TEST ================= */

void matmul_cpu(int N, const float* A, const float* B, float* C)
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            float sum = 0;
            for (int k = 0; k < N; k++)
                sum += A[i*N + k] * B[k*N + j];
            C[i*N + j] = sum;
        }
}

bool verify(int N, const float* ref, const float* gpu)
{
    for (int i = 0; i < N*N; i++) {
        if (fabs(ref[i] - gpu[i]) > 1e-3f) {
            std::cout << "Mismatch at " << i
                      << " ref=" << ref[i]
                      << " gpu=" << gpu[i] << "\n";
            return false;
        }
    }
    return true;
}

int main()
{
    constexpr int TILE = 32;

    std::vector<int> test_sizes = {32, 64, 128};

    for (int N : test_sizes) {

        std::cout << "\n===== Testing N = " << N << " =====\n";

        std::vector<float> A(N*N), B(N*N), C_gpu(N*N), C_cpu(N*N);

        for (int i = 0; i < N*N; i++) {
            A[i] = float(i % 100);
            B[i] = float((i*7) % 100);
        }

        matmul_cpu(N, A.data(), B.data(), C_cpu.data());

        matmul_gpu_block2d(N, TILE,
                           A.data(),
                           B.data(),
                           C_gpu.data());

        bool ok = verify(N, C_cpu.data(), C_gpu.data());

        std::cout << "Verification: " << (ok ? "PASSED" : "FAILED") << "\n";
    }

    {
    const int N = 32768;
    const int TILE = 32;

    printf("\n=== GPU Test N = %d ===\n", N);

    size_t bytes = (size_t)N * N * sizeof(float);

    float *A_h, *B_h, *C_h;

    // Use malloc instead of vector (less overhead)
    A_h = (float*)malloc(bytes);
    B_h = (float*)malloc(bytes);
    C_h = (float*)malloc(bytes);

    if (!A_h || !B_h || !C_h) {
        printf("Host allocation failed (expected for 32k)\n");
        return 0;
    }

    // Initialize
    for (size_t i = 0; i < (size_t)N*N; i++) {
        A_h[i] = (float)(i % 97) / 97.f;
        B_h[i] = (float)((i * 7 + 3) % 97) / 97.f;
    }

    // Timing using CUDA events (kernel-only approx)
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    matmul_gpu_block2d(N, TILE, A_h, B_h, C_h);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    printf("Time (N=32768): %.3f ms\n", ms);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    free(A_h);
    free(B_h);
    free(C_h);
}

    return 0;
}