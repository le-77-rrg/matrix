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

#include <hipblas.h>  // 新增 hipblas 头文件

double matmul_dcu(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C) {
    double *d_A, *d_B, *d_C;
    hipMalloc(&d_A, sizeof(double) * N * M);
    hipMalloc(&d_B, sizeof(double) * M * P);
    hipMalloc(&d_C, sizeof(double) * N * P);

    hipMemcpy(d_A, A.data(), sizeof(double) * N * M, hipMemcpyHostToDevice);
    hipMemcpy(d_B, B.data(), sizeof(double) * M * P, hipMemcpyHostToDevice);

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    hipEventRecord(start);

    const double alpha = 1.0;
    const double beta = 0.0;

    // 注意 hipblasDgemm 是列主矩阵，且参数顺序是 C = alpha*op(A)*op(B) + beta*C
    // 这里我们矩阵按行主存储，因此传入时把A和B顺序对调并设置转置，以等价于行主乘法
    // 即：C = A * B
    // 传给hipblas: op(B) = N, op(A) = N, 矩阵尺寸参数对应
    hipblasStatus_t status = hipblasDgemm(handle,
                                         HIPBLAS_OP_N, HIPBLAS_OP_N,
                                         P, N, M,       // C维度是N×P，但hipblas按列主，所以传入P,N
                                         &alpha,
                                         d_B, P,        // B矩阵 (M×P)，列主存储时 leading dimension = P
                                         d_A, M,        // A矩阵 (N×M)，列主存储时 leading dimension = M
                                         &beta,
                                         d_C, P);       // C矩阵 (N×P)，列主存储时 leading dimension = P

    if (status != HIPBLAS_STATUS_SUCCESS) {
        std::cerr << "hipblasDgemm failed\n";
    }

    hipEventRecord(stop);
    hipEventSynchronize(stop);
    float milliseconds = 0;
    hipEventElapsedTime(&milliseconds, start, stop);

    hipMemcpy(C.data(), d_C, sizeof(double) * N * P, hipMemcpyDeviceToHost);

    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);

    hipblasDestroy(handle);

    return static_cast<double>(milliseconds);  // 单位 ms
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
