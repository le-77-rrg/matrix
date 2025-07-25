#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <omp.h>
// #include <mpi.h>

// 编译执行方式参考：
// 编译， 也可以使用g++，但使用MPI时需使用mpic
// hipcc -O2 -fopenmp -o test_dcu test_dcu.cpp

// 运行 baseline
// ./test_dcu baseline

// 运行 OpenMP
// ./test_dcu openmp

// 运行 子块并行优化
// ./test_dcu block


// 运行 MPI（假设 4 个进程）
// ./test other


// 初始化矩阵（以一维数组形式表示），用于随机填充浮点数
void init_matrix(std::vector<double>& mat, int rows, int cols) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(-100.0, 100.0);
    for (int i = 0; i < rows * cols; ++i)
        mat[i] = dist(gen);
}

// 验证计算优化后的矩阵计算和baseline实现是否结果一致，可以设计其他验证方法，来验证计算的正确性和性能
bool validate(const std::vector<double>& A, const std::vector<double>& B, int rows, int cols, double tol = 1e-6) {
    for (int i = 0; i < rows * cols; ++i)
        if (std::abs(A[i] - B[i]) > tol) return false;
    return true;
}

// 基础的矩阵乘法baseline实现（使用一维数组）
void matmul_baseline(const std::vector<double>& A,
                     const std::vector<double>& B,
                     std::vector<double>& C, int N, int M, int P) {
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < P; ++j) {
            C[i * P + j] = 0;
            for (int k = 0; k < M; ++k)
                C[i * P + j] += A[i * M + k] * B[k * P + j];
        }
}

// 方式1: 利用OpenMP进行多线程并发的编程 （主要修改函数）
void matmul_openmp(const std::vector<double>& A,
                   const std::vector<double>& B,
                   std::vector<double>& C, int N, int M, int P) {
    std::cout << "matmul_openmp methods..." << std::endl;
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < P; ++j) {
            double sum = 0;
            for (int k = 0; k < M; ++k)
                sum += A[i * M + k] * B[k * P + j];
            C[i * P + j] = sum;
        }
}


// 方式2: 利用子块并行思想，进行缓存友好型的并行优化方法 （主要修改函数)
void matmul_block_tiling(const std::vector<double>& A,
                         const std::vector<double>& B,
                         std::vector<double>& C, int N, int M, int P, int block_size=64) {
    std::cout << "matmul_block_tiling methods..." << std::endl;

    #pragma omp parallel for collapse(2)
    for (int ii = 0; ii < N; ii += block_size)
        for (int jj = 0; jj < P; jj += block_size)
            for (int kk = 0; kk < M; kk += block_size)
                for (int i = ii; i < std::min(ii + block_size, N); ++i)
                    for (int j = jj; j < std::min(jj + block_size, P); ++j) {
                        double sum = 0;
                        for (int k = kk; k < std::min(kk + block_size, M); ++k)
                            sum += A[i * M + k] * B[k * P + j];
                        #pragma omp atomic
                        C[i * P + j] += sum;
                    }
}


// // 方式3: 利用MPI消息传递，实现多进程并行优化 （主要修改函数）
// void matmul_mpi(int N, int M, int P) {
//     int rank, size;
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//     MPI_Comm_size(MPI_COMM_WORLD, &size);

//     int rows_per_proc = N / size;
//     std::vector<double> A_local(rows_per_proc * M);
//     std::vector<double> B(M * P);
//     std::vector<double> C_local(rows_per_proc * P, 0);
//     std::vector<double> A, C;

//     if (rank == 0) {
//         A.resize(N * M);
//         C.resize(N * P);
//         init_matrix(A, N, M);
//         init_matrix(B, M, P);
//     }

//     MPI_Bcast(B.data(), M * P, MPI_DOUBLE, 0, MPI_COMM_WORLD);
//     MPI_Scatter(A.data(), rows_per_proc * M, MPI_DOUBLE, A_local.data(), rows_per_proc * M, MPI_DOUBLE, 0, MPI_COMM_WORLD);

//     for (int i = 0; i < rows_per_proc; ++i)
//         for (int j = 0; j < P; ++j) {
//             for (int k = 0; k < M; ++k)
//                 C_local[i * P + j] += A_local[i * M + k] * B[k * P + j];
//         }

//     MPI_Gather(C_local.data(), rows_per_proc * P, MPI_DOUBLE, C.data(), rows_per_proc * P, MPI_DOUBLE, 0, MPI_COMM_WORLD);

//     if (rank == 0) {
//         std::vector<double> C_ref(N * P);
//         matmul_baseline(A, B, C_ref, N, M, P);
//         std::cout << "[MPI] Valid: " << validate(C, C_ref, N, P) << std::endl;
//     }
// }


// 方式4: 其他方式 （主要修改函数）
void matmul_other(const std::vector<double>& A,
                  const std::vector<double>& B,
                  std::vector<double>& C, int N, int M, int P) {
    std::cout << "Other methods..." << std::endl;

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < P; ++j) {
            double sum = 0.0;
            for (int k = 0; k < M; ++k)
                sum += A[i * M + k] * B[k * P + j];  // B[k][j]
            C[i * P + j] = sum;
        }
}

#include <hip/hip_runtime.h>

// GPU kernel: naive 实现版本
__global__ void matmul_kernel(const double* A, const double* B, double* C, int N, int M, int P) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // i
    int col = blockIdx.x * blockDim.x + threadIdx.x; // j

    if (row < N && col < P) {
        double sum = 0.0;
        for (int k = 0; k < M; ++k) {
            sum += A[row * M + k] * B[k * P + col];
        }
        C[row * P + col] = sum;
    }
}

void matmul_dcu(const std::vector<double>& A,
                const std::vector<double>& B,
                std::vector<double>& C, int N, int M, int P) {
    std::cout << "[DCU GPU] Launching HIP kernel..." << std::endl;

    double *d_A, *d_B, *d_C;
    size_t size_A = N * M * sizeof(double);
    size_t size_B = M * P * sizeof(double);
    size_t size_C = N * P * sizeof(double);

    // 分配 GPU 内存
    hipMalloc(&d_A, size_A);
    hipMalloc(&d_B, size_B);
    hipMalloc(&d_C, size_C);

    // 拷贝数据到 GPU
    hipMemcpy(d_A, A.data(), size_A, hipMemcpyHostToDevice);
    hipMemcpy(d_B, B.data(), size_B, hipMemcpyHostToDevice);

    // 定义 block 和 grid 维度
    dim3 blockDim(16, 16);
    dim3 gridDim((P + 15) / 16, (N + 15) / 16);

    // 启动 kernel
    hipLaunchKernelGGL(matmul_kernel, gridDim, blockDim, 0, 0, d_A, d_B, d_C, N, M, P);

    // 拷贝结果回 CPU
    hipMemcpy(C.data(), d_C, size_C, hipMemcpyDeviceToHost);

    // 释放 GPU 内存
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
}



int main(int argc, char** argv) {
    const int N = 1024, M = 2048, P = 512;
    std::string mode = argc >= 2 ? argv[1] : "baseline";

    // if (mode == "mpi") {
    //     MPI_Init(&argc, &argv);
    //     matmul_mpi(N, M, P);
    //     MPI_Finalize();
    //     return 0;
    // }

    std::vector<double> A(N * M);
    std::vector<double> B(M * P);
    std::vector<double> C(N * P, 0);
    std::vector<double> C_ref(N * P, 0);

    init_matrix(A, N, M);
    init_matrix(B, M, P);
    double start_time, end_time;
    start_time = omp_get_wtime();
    matmul_baseline(A, B, C_ref, N, M, P);
    end_time = omp_get_wtime();
    std::cout << "[Baseline] " << "Time: " << (end_time - start_time) << " seconds\n";

    if (mode == "baseline") {
        std::cout << "[Baseline] Done.\n";
    } else if (mode == "openmp") {
        start_time = omp_get_wtime();
        matmul_openmp(A, B, C, N, M, P);
        end_time = omp_get_wtime();
        std::cout << "[OpenMP] Valid: " << validate(C, C_ref, N, P)
                  << ", Time: " << (end_time - start_time) << " seconds\n";
    } else if (mode == "block") {
        start_time = omp_get_wtime();
        matmul_block_tiling(A, B, C, N, M, P);
        end_time = omp_get_wtime();
        std::cout << "[Block Parallel] Valid: " << validate(C, C_ref, N, P)
                  << ", Time: " << (end_time - start_time) << " seconds\n";
    } else if (mode == "other") {
        start_time = omp_get_wtime();
        matmul_other(A, B, C, N, M, P);
        end_time = omp_get_wtime();
        std::cout << "[Other] Valid: " << validate(C, C_ref, N, P)
                  << ", Time: " << (end_time - start_time) << " seconds\n";
    } else if (mode == "dcu") {
        start_time = omp_get_wtime();
        matmul_dcu(A, B, C, N, M, P);
        end_time = omp_get_wtime();
        std::cout << "[DCU GPU] Valid: " << validate(C, C_ref, N, P)
                  << ", Time: " << (end_time - start_time) << " seconds\n";
    }
    else {
        std::cerr << "Usage: ./main [baseline|openmp|block|mpi]" << std::endl;
    }
        // 需额外增加性能评测代码或其他工具进行评测
    return 0;
}