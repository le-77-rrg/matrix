#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <mpi.h>

// 编译执行方式参考：
// 编译， 也可以使用g++，但使用MPI时需使用mpic
// mpic++ -fopenmp -o test test.cpp

// 运行 baseline
// ./test baseline

// 运行 OpenMP
// ./test openmp

// 运行 子块并行优化
// ./test block

// 运行 MPI（假设 4 个进程）
// mpirun --allow-run-as-root -np 4 ./test mpi

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


void matmul_openmp_optimized(const std::vector<double>& A,
                             const std::vector<double>& B,
                             std::vector<double>& C,
                             int N, int M, int P) {
    std::cout << "matmul_openmp_optimized methods..." << std::endl;

    // Step 1: Transpose B => B_T[P][M]
    std::vector<double> B_T(P * M);
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < P; ++j)
            B_T[j * M + i] = B[i * P + j];

    // Step 2: Matrix multiplication using transposed B
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < P; ++j) {
            double sum = 0;
            for (int k = 0; k < M; ++k) {
                sum += A[i * M + k] * B_T[j * M + k];  // Both accesses are row-major (cache-friendly)
            }
            C[i * P + j] = sum;
        }
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

void matmul_block_tiling_optimized(const std::vector<double>& A,
                                        const std::vector<double>& B,
                                        std::vector<double>& C,
                                        int N, int M, int P,
                                        int block_size = 64) {
    std::cout << "matmul_block_tiling_optimized methods..." << std::endl;

    // 先转置矩阵B，变成 B_t[P][M]
    std::vector<double> B_t(P * M);
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < P; ++j)
            B_t[j * M + i] = B[i * P + j];

    #pragma omp parallel for collapse(2)
    for (int ii = 0; ii < N; ii += block_size)
        for (int jj = 0; jj < P; jj += block_size) {
            for (int i = ii; i < std::min(ii + block_size, N); ++i) {
                for (int j = jj; j < std::min(jj + block_size, P); ++j) {
                    double sum = 0;
                    for (int k = 0; k < M; ++k) {
                        sum += A[i * M + k] * B_t[j * M + k];  // B访问变成连续访问
                    }
                    C[i * P + j] += sum;
                }
            }
        }
}








// 方式3: 利用MPI消息传递，实现多进程并行优化 （主要修改函数）
void matmul_mpi(int N, int M, int P) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows_per_proc = N / size;
    std::vector<double> A_local(rows_per_proc * M);
    std::vector<double> B(M * P);
    std::vector<double> C_local(rows_per_proc * P, 0);
    std::vector<double> A, C;

    if (rank == 0) {
        A.resize(N * M);
        C.resize(N * P);
        init_matrix(A, N, M);
        init_matrix(B, M, P);
    }

    MPI_Bcast(B.data(), M * P, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(A.data(), rows_per_proc * M, MPI_DOUBLE, A_local.data(), rows_per_proc * M, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // 计时开始
    double start_time = MPI_Wtime();


    for (int i = 0; i < rows_per_proc; ++i)
        for (int j = 0; j < P; ++j) {
            for (int k = 0; k < M; ++k)
                C_local[i * P + j] += A_local[i * M + k] * B[k * P + j];
        }

    double local_elapsed = MPI_Wtime() - start_time;

    MPI_Gather(C_local.data(), rows_per_proc * P, MPI_DOUBLE, C.data(), rows_per_proc * P, MPI_DOUBLE, 0, MPI_COMM_WORLD);


    // 获取所有进程中最大的运行时间作为整个MPI计算的耗时
    double max_elapsed;
    MPI_Reduce(&local_elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::vector<double> C_ref(N * P);
        matmul_baseline(A, B, C_ref, N, M, P);
        std::cout << "[MPI] Valid: " << validate(C, C_ref, N, P)
                  << ", Time: " << max_elapsed << " seconds\n";
    }
}

void matmul_mpi_optimized(int N, int M, int P) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows_per_proc = N / size;
    std::vector<double> A_local(rows_per_proc * M);
    std::vector<double> B(M * P);           // 原始B矩阵，广播用
    std::vector<double> B_T(P * M);         // 转置后的B矩阵，用于计算
    std::vector<double> C_local(rows_per_proc * P, 0);
    std::vector<double> A, C;

    if (rank == 0) {
        A.resize(N * M);
        C.resize(N * P);
        init_matrix(A, N, M);
        init_matrix(B, M, P);

        // 对B进行转置，方便缓存访问
        for (int m = 0; m < M; ++m) {
            for (int p = 0; p < P; ++p) {
                B_T[p * M + m] = B[m * P + p];
            }
        }
    }

    // 广播原始的B矩阵，方便后面验证（验证时用）
    MPI_Bcast(B.data(), M * P, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // 广播转置后的B矩阵，计算时用
    MPI_Bcast(B_T.data(), M * P, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Scatter(A.data(), rows_per_proc * M, MPI_DOUBLE, A_local.data(), rows_per_proc * M, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double start_time = MPI_Wtime();

    // 计算 C_local = A_local * B_T^T = A_local * B
    for (int i = 0; i < rows_per_proc; ++i) {
        for (int j = 0; j < P; ++j) {
            double sum = 0;
            for (int k = 0; k < M; ++k) {
                sum += A_local[i * M + k] * B_T[j * M + k];  // 访问转置后的B，连续访问缓存友好
            }
            C_local[i * P + j] = sum;
        }
    }

    double local_elapsed = MPI_Wtime() - start_time;

    MPI_Gather(C_local.data(), rows_per_proc * P, MPI_DOUBLE, C.data(), rows_per_proc * P, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double max_elapsed;
    MPI_Reduce(&local_elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::vector<double> C_ref(N * P);
        matmul_baseline(A, B, C_ref, N, M, P);  // 这里用原始B做验证
        std::cout << "[MPI] Valid: " << validate(C, C_ref, N, P)
                  << ", Time: " << max_elapsed << " seconds\n";
    }
}


void matmul_other(const std::vector<double>& A,
                  const std::vector<double>& B,
                  std::vector<double>& C, int N, int M, int P) {
    std::cout << "Other methods (Hybrid Blocked + OpenMP Optimized)..." << std::endl;

    const int block_size = 64;  // 调整以适配缓存
    #pragma omp parallel
    {
        std::vector<double> C_local(N * P, 0.0);  // 每线程私有缓存，避免写冲突

        #pragma omp for collapse(2) nowait
        for (int ii = 0; ii < N; ii += block_size) {
            for (int jj = 0; jj < P; jj += block_size) {
                for (int kk = 0; kk < M; kk += block_size) {
                    for (int i = ii; i < std::min(ii + block_size, N); ++i) {
                        for (int k = kk; k < std::min(kk + block_size, M); ++k) {
                            double a_val = A[i * M + k];
                            for (int j = jj; j < std::min(jj + block_size, P); ++j) {
                                C_local[i * P + j] += a_val * B[k * P + j];
                            }
                        }
                    }
                }
            }
        }

        // 归约写入主结果
        #pragma omp critical
        {
            for (int i = 0; i < N * P; ++i)
                C[i] += C_local[i];
        }
    }
}




int main(int argc, char** argv) {
    const int N = 1024, M = 2048, P = 512;
    std::string mode = argc >= 2 ? argv[1] : "baseline";

    if (mode == "mpi") {
        MPI_Init(&argc, &argv);
        matmul_mpi_optimized(N, M, P);
        MPI_Finalize();
        return 0;
    }

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
        matmul_openmp_transpose(A, B, C, N, M, P);
        end_time = omp_get_wtime();
        std::cout << "[OpenMP] Valid: " << validate(C, C_ref, N, P)
                  << ", Time: " << (end_time - start_time) << " seconds\n";
    } else if (mode == "block") {
        start_time = omp_get_wtime();
        matmul_block_tiling_optimized(A, B, C, N, M, P);
        end_time = omp_get_wtime();
        std::cout << "[Block Parallel] Valid: " << validate(C, C_ref, N, P)
                  << ", Time: " << (end_time - start_time) << " seconds\n";
    } else if (mode == "other") {
        start_time = omp_get_wtime();
        matmul_other(A, B, C, N, M, P);
        end_time = omp_get_wtime();
        std::cout << "[Other] Valid: " << validate(C, C_ref, N, P)
                  << ", Time: " << (end_time - start_time) << " seconds\n";
    } else {
        std::cerr << "Usage: ./main [baseline|openmp|block|mpi]" << std::endl;
    }
        // 需额外增加性能评测代码或其他工具进行评测
    return 0;
}