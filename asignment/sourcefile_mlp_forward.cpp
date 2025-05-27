#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <chrono>

// 编译文件
// hipcc sourcefile_mlp_forward.cpp -o mlp_forward
// 执行文件
// ./mlp_forward 或者 hipprof ./mlp_forward

#define BATCH 1024
#define I 10
#define H 20
#define O 5

__global__ void matmul_kernel(const double* A, const double* B, double* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // M
    int col = blockIdx.x * blockDim.x + threadIdx.x; // K
    if (row < M && col < K) {
        double sum = 0.0;
        for (int i = 0; i < N; ++i)
            sum += A[row * N + i] * B[i * K + col];
        C[row * K + col] = sum;
    }
}

__global__ void relu_kernel(double* A, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        A[idx] = fmax(0.0, A[idx]);
}

__global__ void add_bias_kernel(double* A, const double* bias, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        int col = idx % cols;
        A[idx] += bias[col];
    }
}

void random_init(std::vector<double>& mat) {
    for (auto& val : mat)
        val = static_cast<double>(rand()) / RAND_MAX * 2 - 1;
}

// CPU baseline for comparison
void mlp_forward_cpu(const std::vector<double>& X,
                     const std::vector<double>& W1,
                     const std::vector<double>& B1,
                     const std::vector<double>& W2,
                     const std::vector<double>& B2,
                     std::vector<double>& H_layer,
                     std::vector<double>& Y)
{
    for (int b = 0; b < BATCH; ++b) {
        for (int h = 0; h < H; ++h) {
            double sum = B1[h];
            for (int i = 0; i < I; ++i) {
                sum += X[b * I + i] * W1[i * H + h];
            }
            H_layer[b * H + h] = fmax(0.0, sum); // ReLU
        }

        for (int o = 0; o < O; ++o) {
            double sum = B2[o];
            for (int h = 0; h < H; ++h) {
                sum += H_layer[b * H + h] * W2[h * O + o];
            }
            Y[b * O + o] = sum;
        }
    }
}

int main() {
    std::vector<double> h_X(BATCH * I), h_W1(I * H), h_B1(H), h_W2(H * O), h_B2(O);
    std::vector<double> h_H(BATCH * H), h_Y(BATCH * O), h_Y_cpu(BATCH * O);
    random_init(h_X);
    random_init(h_W1);
    random_init(h_B1);
    random_init(h_W2);
    random_init(h_B2);

    // CPU baseline
    auto t0 = std::chrono::high_resolution_clock::now();
    mlp_forward_cpu(h_X, h_W1, h_B1, h_W2, h_B2, h_H, h_Y_cpu);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "[CPU Time] " << cpu_time << " ms\n";

    // Device memory
    double *d_X, *d_W1, *d_B1, *d_H, *d_W2, *d_B2, *d_Y;
    hipMalloc(&d_X, BATCH * I * sizeof(double));
    hipMalloc(&d_W1, I * H * sizeof(double));
    hipMalloc(&d_B1, H * sizeof(double));
    hipMalloc(&d_H, BATCH * H * sizeof(double));
    hipMalloc(&d_W2, H * O * sizeof(double));
    hipMalloc(&d_B2, O * sizeof(double));
    hipMalloc(&d_Y, BATCH * O * sizeof(double));

    hipMemcpy(d_X, h_X.data(), BATCH * I * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_W1, h_W1.data(), I * H * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_B1, h_B1.data(), H * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_W2, h_W2.data(), H * O * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_B2, h_B2.data(), O * sizeof(double), hipMemcpyHostToDevice);


    dim3 block(16, 16);
    dim3 grid1((H + block.x - 1) / block.x, (BATCH + block.y - 1) / block.y);
    dim3 grid2((O + block.x - 1) / block.x, (BATCH + block.y - 1) / block.y);

    hipDeviceSynchronize();  

    auto t2 = std::chrono::high_resolution_clock::now();

    // H = X * W1
    hipLaunchKernelGGL(matmul_kernel, grid1, block, 0, 0, d_X, d_W1, d_H, BATCH, I, H);

    // Add B1 and ReLU
    hipLaunchKernelGGL(add_bias_kernel, dim3((BATCH * H + 255) / 256), dim3(256), 0, 0, d_H, d_B1, BATCH, H);
    hipLaunchKernelGGL(relu_kernel, dim3((BATCH * H + 255) / 256), dim3(256), 0, 0, d_H, BATCH * H);

    // Y = H * W2
    hipLaunchKernelGGL(matmul_kernel, grid2, block, 0, 0, d_H, d_W2, d_Y, BATCH, H, O);

    // Add B2
    hipLaunchKernelGGL(add_bias_kernel, dim3((BATCH * O + 255) / 256), dim3(256), 0, 0, d_Y, d_B2, BATCH, O);

    
    hipDeviceSynchronize();
    auto t3 = std::chrono::high_resolution_clock::now();
    double gpu_time = std::chrono::duration<double, std::milli>(t3 - t2).count();
    std::cout << "[GPU Time] " << gpu_time << " ms\n";

    hipMemcpy(h_Y.data(), d_Y, BATCH * O * sizeof(double), hipMemcpyDeviceToHost);

    // Compare and print acceleration
    bool valid = true;
    for (int i = 0; i < BATCH * O; ++i) {
        if (std::abs(h_Y[i] - h_Y_cpu[i]) > 1e-6) {
            valid = false;
            break;
        }
    }
    std::cout << "[Validation] " << (valid ? "PASS" : "FAIL") << std::endl;
    std::cout << "[Speedup] " << cpu_time / gpu_time << "x\n";

    hipFree(d_X); hipFree(d_W1); hipFree(d_B1); hipFree(d_H);
    hipFree(d_W2); hipFree(d_B2); hipFree(d_Y);
    return 0;
}
