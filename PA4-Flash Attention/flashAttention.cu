#include<vector>
#include<iostream>
#include<cmath>
#include<cuda_runtime.h>
#include <fstream>


using namespace std;


constexpr int Br = 32;
constexpr int Bc = 32;
constexpr int Hs = 64;
constexpr int smem_size = Br * Hs + 2 * Bc * Hs + Br * Bc;
constexpr float float_min = -999999.99;

__global__
void flash_attention(float *Q, float *K, float *V, float *m, float *l, float *O, int Nh, int N){
/*
    Q, K, V = B, Nh, N, Hs
    m,l = B, Nh, N
    O = B, Nh, N, Hs
*/
    
    
    int b = blockIdx.x;
    int h = blockIdx.y;

    int tidx = threadIdx.x;
    int qkv_base = (b*Nh + h)*N * Hs;
    int lm_base = (b*Nh + h)*N;

    int Tc = (N + Bc -1)/ Bc;
    int Tr = (N + Br -1)/ Br;

    float scale = rsqrtf((float)Hs);

    // smem will hold Q, K, V, and final values of this tile
    // Qi = Br * Hs, K/V = Bc * Hs, final values = Br * Bc

    __shared__ float smem[smem_size];
    float *Qi = smem;
    float *Kj = Qi + Br * Hs;
    float *Vj = Kj + Bc * Hs;
    float *S  = Vj + Bc * Hs;

    for(int j = 0; j<Tc; j++){
        for(int x=0; x<Hs; x++){
            int kv_idx = qkv_base + (j*Bc+tidx)*Hs + x;
            Kj[tidx * Hs + x] = K[kv_idx];
            Vj[tidx * Hs + x] = V[kv_idx];
        }
        __syncthreads();
        

        for(int i=0; i<Tr; i++){
            // step 1
            for(int x=0; x<Hs; x++){
                Qi[tidx*Hs + x] = Q[qkv_base+ (i*Br + tidx)*Hs + x];                
            }
            
            // step 2
            int lm_idx = lm_base + i*Br + tidx;
            float m_prev = m[lm_idx];
            float l_prev = l[lm_idx];

            // step 3
            float m_tilda = float_min;
            for(int col=0; col<Bc; col++){
                float sm = 0;
                for(int x=0; x<Hs; x++){
                    sm += Qi[tidx*Hs + x] * Kj[col*Hs + x];
                }
                m_tilda = max(sm, m_tilda);
                S[tidx * Bc + col] = sm * scale;
            }

            // step 4
            float l_tilda = 0;            
            for(int x=0; x<Bc; x++){
                float temp = expf(S[tidx * Bc + x] - m_tilda);
                S[tidx * Bc + x] = temp; // S holds the value of P now
                l_tilda += temp;
            }

            // step 5
            float m_new = max(m_prev, m_tilda);
            float l_new = expf(m_prev - m_new) * l_prev + expf(m_tilda - m_new) * l_tilda;

            float PV[Hs];
            for(int col=0; col<Hs; col++){
                float sum = 0;
                for(int x=0; x<Bc; x++){
                    sum += S[tidx * Bc + x] * Vj[x*Hs + col];
                }
                PV[col] = sum;
            }

            for(int x=0; x<Hs; x++){
                O[((b*Nh + h)*N + i*Br + tidx)*Hs + x] = (expf(m_prev - m_new) * l_prev * O[((b*Nh + h)*N + i*Br + tidx)*Hs + x] + expf(m_tilda - m_new) * PV[x])/ l_new;
            }

            m[lm_idx] = m_new;
            l[lm_idx] = l_new;            
            
        }
        __syncthreads();

    }


}

// initialise m and l to -inf and 0

int main() {
    int B = 1;
    int Nh = 1;
    int N = 64;

    size_t qkv_size = B * Nh * N * Hs;
    size_t ml_size  = B * Nh * N;

    std::vector<float> h_Q(qkv_size), h_K(qkv_size), h_V(qkv_size);
    std::vector<float> h_O(qkv_size, 0.0f);
    std::vector<float> h_m(ml_size, float_min);
    std::vector<float> h_l(ml_size, 0.0f);

    // init inputs
    for (int i = 0; i < qkv_size; i++) {
        h_Q[i] = 0.01f * (i % 100);
        h_K[i] = 0.02f * (i % 100);
        h_V[i] = 0.03f * (i % 100);
    }

    float *d_Q, *d_K, *d_V, *d_O, *d_m, *d_l;

    cudaMalloc(&d_Q, qkv_size * sizeof(float));
    cudaMalloc(&d_K, qkv_size * sizeof(float));
    cudaMalloc(&d_V, qkv_size * sizeof(float));
    cudaMalloc(&d_O, qkv_size * sizeof(float));
    cudaMalloc(&d_m, ml_size * sizeof(float));
    cudaMalloc(&d_l, ml_size * sizeof(float));

    cudaMemcpy(d_Q, h_Q.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_m, h_m.data(), ml_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_l, h_l.data(), ml_size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 grid(B, Nh);    // one block per Q-tile
    dim3 block(Br);                  // one thread per row

    flash_attention<<<grid, block>>>(d_Q, d_K, d_V, d_m, d_l, d_O, Nh, N);

    cudaDeviceSynchronize();

    cudaMemcpy(h_O.data(), d_O, qkv_size * sizeof(float), cudaMemcpyDeviceToHost);

    // print few values
    for (int i = 0; i < 10; i++) {
        std::cout << h_O[i] << " ";
    }
    std::cout << std::endl;

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
    cudaFree(d_m);
    cudaFree(d_l);

    std::ofstream fout("cuda_out.txt");
    for (int i = 0; i < qkv_size; i++) {
        fout << h_O[i] << "\n";
    }
    fout.close();

    std::ofstream fq("Q.txt"), fk("K.txt"), fv("V.txt");

    for (int i = 0; i < qkv_size; i++) fq << h_Q[i] << "\n";
    for (int i = 0; i < qkv_size; i++) fk << h_K[i] << "\n";
    for (int i = 0; i < qkv_size; i++) fv << h_V[i] << "\n";

    return 0;
}