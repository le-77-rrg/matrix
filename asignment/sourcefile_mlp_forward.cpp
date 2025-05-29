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

// 增大计算规模以更好体现GPU优势
#define BATCH 1024  // 增大batch size
#define I 10      // 增大输入维度
#define H 20       // 增大隐藏层维度
#define O 5        // 增大输出维度

// 修正的矩阵乘法核函数
__global__ void matmul_kernel(const double* A, const double* B, double* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {  // 修正：col < N
        double sum = 0.0;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];  // 修正索引
        }
        C[row * N + col] = sum;  // 修正输出索引
    }
}

// 优化的矩阵乘法核函数（使用共享内存）
__global__ void matmul_kernel_optimized(const double* A, const double* B, double* C, int M, int N, int K) {
    __shared__ double As[16][16];
    __shared__ double Bs[16][16];
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    double sum = 0.0;
    
    for (int tile = 0; tile < (K + blockDim.x - 1) / blockDim.x; ++tile) {
        // 加载数据到共享内存
        int k = tile * blockDim.x + threadIdx.x;
        if (row < M && k < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + k];
        else
            As[threadIdx.y][threadIdx.x] = 0.0;
            
        k = tile * blockDim.y + threadIdx.y;
        if (k < K && col < N)
            Bs[threadIdx.y][threadIdx.x] = B[k * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0;
            
        __syncthreads();
        
        // 计算部分乘积
        for (int k = 0; k < blockDim.x; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
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

// CPU baseline
void mlp_forward_cpu(const std::vector<double>& X, const std::vector<double>& W1,
                     const std::vector<double>& B1, const std::vector<double>& W2,
                     const std::vector<double>& B2, std::vector<double>& H_layer,
                     std::vector<double>& Y) {
    // 隐藏层计算
    for (int b = 0; b < BATCH; ++b) {
        for (int h = 0; h < H; ++h) {
            double sum = B1[h];
            for (int i = 0; i < I; ++i) {
                sum += X[b * I + i] * W1[i * H + h];
            }
            H_layer[b * H + h] = fmax(0.0, sum); // ReLU
        }
    }
    
    // 输出层计算
    for (int b = 0; b < BATCH; ++b) {
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

    std::cout << "计算规模: BATCH=" << BATCH << ", I=" << I << ", H=" << H << ", O=" << O << std::endl;

    // CPU baseline - 多次运行取平均
    double cpu_time_total = 0.0;
    const int cpu_runs = 5;
    for (int run = 0; run < cpu_runs; ++run) {
        auto t0 = std::chrono::high_resolution_clock::now();
        mlp_forward_cpu(h_X, h_W1, h_B1, h_W2, h_B2, h_H, h_Y_cpu);
        auto t1 = std::chrono::high_resolution_clock::now();
        cpu_time_total += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    double cpu_time = cpu_time_total / cpu_runs;
    std::cout << "[CPU时间] " << cpu_time << " ms (" << cpu_runs << "次平均)" << std::endl;

    // GPU计算
    double *d_X, *d_W1, *d_B1, *d_H, *d_W2, *d_B2, *d_Y;
    
    // 分配设备内存
    hipMalloc(&d_X, BATCH * I * sizeof(double));
    hipMalloc(&d_W1, I * H * sizeof(double));
    hipMalloc(&d_B1, H * sizeof(double));
    hipMalloc(&d_H, BATCH * H * sizeof(double));
    hipMalloc(&d_W2, H * O * sizeof(double));
    hipMalloc(&d_B2, O * sizeof(double));
    hipMalloc(&d_Y, BATCH * O * sizeof(double));

    // 数据传输到GPU
    hipMemcpy(d_X, h_X.data(), BATCH * I * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_W1, h_W1.data(), I * H * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_B1, h_B1.data(), H * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_W2, h_W2.data(), H * O * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_B2, h_B2.data(), O * sizeof(double), hipMemcpyHostToDevice);

    // 优化的线程配置
    dim3 block(16, 16);
    dim3 grid1((H + block.x - 1) / block.x, (BATCH + block.y - 1) / block.y);
    dim3 grid2((O + block.x - 1) / block.x, (BATCH + block.y - 1) / block.y);
    
    int threads_per_block = 256;
    int blocks_h = (BATCH * H + threads_per_block - 1) / threads_per_block;
    int blocks_o = (BATCH * O + threads_per_block - 1) / threads_per_block;

    // GPU预热
    matmul_kernel_optimized<<<grid1, block>>>(d_X, d_W1, d_H, BATCH, H, I);
    hipDeviceSynchronize();

    // 精确计时GPU计算（不包含数据传输）
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    
    const int gpu_runs = 10;
    float gpu_time_total = 0.0;
    
    for (int run = 0; run < gpu_runs; ++run) {
        hipEventRecord(start);
        
        // 隐藏层: H = X * W1 + B1, ReLU
        matmul_kernel_optimized<<<grid1, block>>>(d_X, d_W1, d_H, BATCH, H, I);
        add_bias_kernel<<<blocks_h, threads_per_block>>>(d_H, d_B1, BATCH, H);
        relu_kernel<<<blocks_h, threads_per_block>>>(d_H, BATCH * H);
        
        // 输出层: Y = H * W2 + B2
        matmul_kernel_optimized<<<grid2, block>>>(d_H, d_W2, d_Y, BATCH, O, H);
        add_bias_kernel<<<blocks_o, threads_per_block>>>(d_Y, d_B2, BATCH, O);
        
        hipEventRecord(stop);
        hipEventSynchronize(stop);
        
        float run_time;
        hipEventElapsedTime(&run_time, start, stop);
        gpu_time_total += run_time;
    }
    
    double gpu_time = gpu_time_total / gpu_runs;
    std::cout << "[GPU时间] " << gpu_time << " ms (" << gpu_runs << "次平均，纯计算)" << std::endl;

    // 数据传输回CPU
    hipMemcpy(h_Y.data(), d_Y, BATCH * O * sizeof(double), hipMemcpyDeviceToHost);

    // 验证结果
    bool valid = true;
    double max_diff = 0.0;
    for (int i = 0; i < BATCH * O; ++i) {
        double diff = std::abs(h_Y[i] - h_Y_cpu[i]);
        max_diff = std::max(max_diff, diff);
        if (diff > 1e-6) {
            valid = false;
        }
    }
    
    std::cout << "[验证结果] " << (valid ? "通过" : "失败") << std::endl;
    std::cout << "[最大误差] " << max_diff << std::endl;
    std::cout << "[加速比] " << cpu_time / gpu_time << "x" << std::endl;
    
    // 计算GFLOPS
    double flops = 2.0 * BATCH * (I * H + H * O); // 矩阵乘法的浮点运算数
    std::cout << "[CPU GFLOPS] " << flops / (cpu_time * 1e6) << std::endl;
    std::cout << "[GPU GFLOPS] " << flops / (gpu_time * 1e6) << std::endl;

    // 清理资源
    hipEventDestroy(start);
    hipEventDestroy(stop);
    hipFree(d_X);
    hipFree(d_W1);
    hipFree(d_B1);
    hipFree(d_H);
    hipFree(d_W2);
    hipFree(d_B2);
    hipFree(d_Y);

    return 0;
}