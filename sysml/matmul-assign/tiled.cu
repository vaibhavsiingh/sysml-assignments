%%writefile tiled.cu

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

// ─── Error-checking macro ────────────────────────────────────────────────────
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d  %s\n",                       \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)


/* ============================================================
   Edit ONLY this section.
   ============================================================ */

/**
 * TODO: Implement your CUDA kernel(s) here.
 *
 * You may define multiple __global__ and __device__ functions.
 * You may use templates, #define constants, and helper structs.
 */

// ─── Example: a bare naive kernel to get you started ─────────────────────────
#define TILE 16   // 16x16 tile → 2 * 16*16 * 4B = 2KB shared memory

__global__
void matmul_kernel_tiled(const float* __restrict__ A,
                         const float* __restrict__ B,
                               float* __restrict__ C,
                         int N)
{
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;

    // Loop over tiles
    for (int t = 0; t < N; t += TILE) {

        // Load A tile
        if (row < N && (t + threadIdx.x) < N)
            As[threadIdx.y][threadIdx.x] = A[row * N + (t + threadIdx.x)];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        // Load B tile
        if (col < N && (t + threadIdx.y) < N)
            Bs[threadIdx.y][threadIdx.x] = B[(t + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Compute partial product
        #pragma unroll
        for (int k = 0; k < TILE; ++k)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads();
    }

    if (row < N && col < N)
        C[row * N + col] = sum;
}
/**
 * @brief Launch wrapper — allocate device memory, copy data,
 *        run your kernel(s), copy result back. You aren't allowed to change this function signature.
 *
 * @param N    Matrix dimension (N x N).  Always a power of 2.
 * @param A_h  Host pointer to matrix A (row-major, N*N floats).
 * @param B_h  Host pointer to matrix B (row-major, N*N floats).
 * @param C_h  Host pointer to output C (row-major, N*N floats).
 *             You must write the result here before returning.
 */
void matmul_gpu(int N,
                const float* A_h,
                const float* B_h,
                      float* C_h)
{
    size_t bytes = (size_t)N * N * sizeof(float);

    // ── Allocate device buffers ───────────────────────────────
    float *A_d, *B_d, *C_d;
    CUDA_CHECK(cudaMalloc(&A_d, bytes));
    CUDA_CHECK(cudaMalloc(&B_d, bytes));
    CUDA_CHECK(cudaMalloc(&C_d, bytes));

    // ── Transfer inputs to device ─────────────────────────────
    CUDA_CHECK(cudaMemcpy(A_d, A_h, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_d, B_h, bytes, cudaMemcpyHostToDevice));

    // ── Create CUDA events ─────────────────────────────
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // ── Record start ───────────────────────────────────
    CUDA_CHECK(cudaEventRecord(start));
    
    {
        dim3 block(16, 16);
        dim3 grid((N + block.x - 1) / block.x,
                  (N + block.y - 1) / block.y);
    
        matmul_kernel_tiled<<<grid, block>>>(A_d, B_d, C_d, N);
        CUDA_CHECK(cudaGetLastError());
    }
    
    // ── Record end ─────────────────────────────────────
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    // ── Compute elapsed time ───────────────────────────
    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    
    printf("Kernel time (N=%d): %.3f ms\n", N, milliseconds);

    // ── Cleanup ────────────────────────────────────────
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaDeviceSynchronize());

    // ── Copy result back to host ──────────────────────────────
    CUDA_CHECK(cudaMemcpy(C_h, C_d, bytes, cudaMemcpyDeviceToHost));

    // ── Free device memory ────────────────────────────────────
    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(B_d));
    CUDA_CHECK(cudaFree(C_d));
}

/* ============================================================
   END OF STUDENT CODE — do not modify below this line
   ============================================================ */


// ─── CPU reference ────────────────────────────────────────────────────────────
static void matmul_cpu(int N,
                       const float* A,
                       const float* B,
                             float* C)
{
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            float s = 0.0f;
            for (int k = 0; k < N; ++k)
                s += A[i*N+k] * B[k*N+j];
            C[i*N+j] = s;
        }
}

// ─── Element-wise verification ────────────────────────────────────────────────
static bool verify(int N, const float* ref, const float* gpu,
                   float tol = 1e-2f)
{
    for (int i = 0; i < N*N; ++i) {
        float diff = fabsf(ref[i] - gpu[i]);
        if (diff > tol) {
            int row = i / N, col = i % N;
            fprintf(stderr,
                    "MISMATCH at (%d,%d): ref=%.6f  gpu=%.6f  |diff|=%.2e\n",
                    row, col, ref[i], gpu[i], diff);
            return false;
        }
    }
    return true;
}

int main()
{
    // ── Correctness tests (small sizes, CPU reference) ────────
    printf("=== Correctness Tests ===\n");
    {
        const std::vector<int> small_sizes = {64, 128, 256, 512};
        bool all_ok = true;

        for (int N : small_sizes) {

            std::vector<float> A(N*N), B(N*N),
                               C_cpu(N*N, 0.f),
                               C_gpu(N*N, 0.f);

            for (int i = 0; i < N*N; ++i) {
                A[i] = (float)(i % 97) / 97.f;
                B[i] = (float)((i * 7 + 3) % 97) / 97.f;
            }

            matmul_cpu(N, A.data(), B.data(), C_cpu.data());
            matmul_gpu(N, A.data(), B.data(), C_gpu.data());

            bool ok = verify(N, C_cpu.data(), C_gpu.data());
            printf("  N = %4d : %s\n", N, ok ? "PASSED" : "FAILED");
            all_ok &= ok;
        }

        if (!all_ok) {
            fprintf(stderr,
                    "\nCorrectness FAILED — fix your kernel before optimising.\n");
            return EXIT_FAILURE;
        }
        printf("All correctness tests PASSED.\n\n");
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
}