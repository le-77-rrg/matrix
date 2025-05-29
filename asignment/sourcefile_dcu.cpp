#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>

// 编译
// hipcc sourcefile_dcu.cpp -o outputfile_dcu
// 执行
// ./outputfile_dcu
//hipprof ./outputfile_dcu

#define N 1024
#define M 2024
#define P 512

// 主要修改函数
__global__ void matmul_kernel(const double* A, const double* B, double* C, int n, int m, int p) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < p) {
        double sum = 0.0;
        for (int k = 0; k < m; ++k)
            sum += A[row * m + k] * B[k * p + col];
        C[row * p + col] = sum;
    }
}

#define TILE_SIZE 16

__global__ void matmul_kernel_shared(const double* A, const double* B, double* C, int n, int m, int p) {
    __shared__ double tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ double tile_B[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    double sum = 0.0;

    for (int t = 0; t < (m + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // 从全局内存加载子块 A 和 B 到共享内存（边界检查）
        if (row < n && t * TILE_SIZE + threadIdx.x < m)
            tile_A[threadIdx.y][threadIdx.x] = A[row * m + t * TILE_SIZE + threadIdx.x];
        else
            tile_A[threadIdx.y][threadIdx.x] = 0.0;

        if (t * TILE_SIZE + threadIdx.y < m && col < p)
            tile_B[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * p + col];
        else
            tile_B[threadIdx.y][threadIdx.x] = 0.0;

        __syncthreads();  // 同步线程块内所有线程

        for (int k = 0; k < TILE_SIZE; ++k)
            sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];

        __syncthreads();  // 准备下一轮 tile 加载
    }

    if (row < n && col < p)
        C[row * p + col] = sum;
}


double matmul_dcu(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C) {
    double *d_A, *d_B, *d_C;
    hipMalloc(&d_A, sizeof(double) * N * M);
    hipMalloc(&d_B, sizeof(double) * M * P);
    hipMalloc(&d_C, sizeof(double) * N * P);

    hipMemcpy(d_A, A.data(), sizeof(double) * N * M, hipMemcpyHostToDevice);
    hipMemcpy(d_B, B.data(), sizeof(double) * M * P, hipMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((P + blockDim.x - 1) / blockDim.x,
                 (N + blockDim.y - 1) / blockDim.y);

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    hipEventRecord(start);

    matmul_kernel_shared<<<gridDim, blockDim>>>(d_A, d_B, d_C, N, M, P);

    hipEventRecord(stop);
    hipEventSynchronize(stop);
    float milliseconds = 0;
    hipEventElapsedTime(&milliseconds, start, stop);

    hipMemcpy(C.data(), d_C, sizeof(double) * N * P, hipMemcpyDeviceToHost);

    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);

    return static_cast<double>(milliseconds);  // 毫秒
}





void init_matrix(std::vector<double>& mat) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (auto& x : mat)
        x = dist(gen);
    return;
}

double matmul_cpu(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C) {
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < N; ++i)
        for (int j = 0; j < P; ++j) {
            double sum = 0.0;
            for (int k = 0; k < M; ++k)
                sum += A[i * M + k] * B[k * P + j];
            C[i * P + j] = sum;
        }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    return duration.count();  // 毫秒
}

bool validate(const std::vector<double>& ref, const std::vector<double>& test) {
    for (size_t i = 0; i < ref.size(); ++i)
        if (std::abs(ref[i] - test[i]) > 1e-6)
            return false;
    return true;
}

int main() {
    std::vector<double> A(N * M), B(M * P), C_cpu(N * P), C_dcu(N * P);
    init_matrix(A);
    init_matrix(B);

    double time_cpu = matmul_cpu(A, B, C_cpu);
    double time_dcu = matmul_dcu(A, B, C_dcu);

    bool correct = validate(C_cpu, C_dcu);

    std::cout << "[CPU] Time: " << time_cpu << " ms\n";
    std::cout << "[DCU] Time: " << time_dcu << " ms\n";
    std::cout << "[HIP] Valid: " << (correct ? "1" : "0") << std::endl;

    if (time_dcu > 0)
        std::cout << "[Speedup] CPU/DCU: " << (time_cpu / time_dcu) << "x\n";

    return 0;
}
