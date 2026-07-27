#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
} while(0)

#define R 2

__global__
void matmul_tiled_block2d_kernel(const float* __restrict__ A,
                                 const float* __restrict__ B,
                                 float* __restrict__ C,
                                 int N,
                                 int TILE)
{
    extern __shared__ float s[];
    float* sA = s;
    float* sB = s + TILE*TILE;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int rowBase = blockIdx.y * TILE + ty * R;
    int colBase = blockIdx.x * TILE + tx * R;

    float reg[R][R] = {0};

    int numTiles = (N + TILE - 1) / TILE;

    for (int t = 0; t < numTiles; t++) {

        // --- FIXED: full cooperative load ---
        for (int i = 0; i < R; i++) {
            int row = rowBase + i;
            for (int j = 0; j < R; j++) {
                int col = t * TILE + tx * R + j;

                sA[(ty * R + i) * TILE + (tx * R + j)] =
                    (row < N && col < N) ? A[row * N + col] : 0.0f;
            }
        }

        for (int i = 0; i < R; i++) {
            int row = t * TILE + ty * R + i;
            for (int j = 0; j < R; j++) {
                int col = colBase + j;

                sB[(ty * R + i) * TILE + (tx * R + j)] =
                    (row < N && col < N) ? B[row * N + col] : 0.0f;
            }
        }

        __syncthreads();

        // --- Compute ---
        #pragma unroll
        for (int k = 0; k < TILE; k++) {
            for (int i = 0; i < R; i++) {
                float a = sA[(ty * R + i) * TILE + k];
                for (int j = 0; j < R; j++) {
                    reg[i][j] += a * sB[k * TILE + (tx * R + j)];
                }
            }
        }

        __syncthreads();
    }

    // --- Store ---
    for (int i = 0; i < R; i++) {
        for (int j = 0; j < R; j++) {
            int row = rowBase + i;
            int col = colBase + j;

            if (row < N && col < N)
                C[row * N + col] = reg[i][j];
        }
    }
}


void matmul_gpu_block2d(int N,
                        int TILE,
                        const float* A_h,
                        const float* B_h,
                        float* C_h)
{
    size_t bytes = (size_t)N * N * sizeof(float);

    float *A_d, *B_d, *C_d;
    CUDA_CHECK(cudaMalloc(&A_d, bytes));
    CUDA_CHECK(cudaMalloc(&B_d, bytes));
    CUDA_CHECK(cudaMalloc(&C_d, bytes));

    CUDA_CHECK(cudaMemcpy(A_d, A_h, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_d, B_h, bytes, cudaMemcpyHostToDevice));

    dim3 block(TILE / R, TILE / R);
    dim3 grid((N + TILE - 1) / TILE,
              (N + TILE - 1) / TILE);

    size_t sharedBytes = 2 * TILE * TILE * sizeof(float);

    matmul_tiled_block2d_kernel<<<grid, block, sharedBytes>>>(A_d, B_d, C_d, N, TILE);
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

     // ── Performance test (large size, averaged runs) ───────────
    printf("=== Performance Test (Averaged) ===\n");
    {
        const int N = 32768;
        const int NUM_RUNS = 2;

        std::vector<float> A(N*N), B(N*N), C(N*N, 0.f);

        for (int i = 0; i < N*N; ++i) {
            A[i] = (float)(i % 97) / 97.f;
            B[i] = (float)((i * 7 + 3) % 97) / 97.f;
        }

        float total_time = 0.0f;

        for (int run = 0; run < NUM_RUNS; ++run) {
            printf("Run %d/%d:\n", run + 1, NUM_RUNS);

            auto start = std::chrono::high_resolution_clock::now();

            matmul_gpu(N, A.data(), B.data(), C.data());

            auto end = std::chrono::high_resolution_clock::now();
            float elapsed =
                std::chrono::duration<float, std::milli>(end - start).count();

            total_time += elapsed;
        }

        float avg_time = total_time / NUM_RUNS;
        printf("\nAverage total time over %d runs (N=%d): %.3f ms\n",
               NUM_RUNS, N, avg_time);
    }

    return EXIT_SUCCESS;

    return 0;
}