#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>
#include <cmath>
#include <iomanip>
#include <cuda_runtime.h>

namespace fs = std::filesystem;

#define BLOCK_DIM_X 16
#define BLOCK_DIM_Y 16

// Constant Sobel filter masks
__constant__ float c_sobelX[9];
__constant__ float c_sobelY[9];

// Kernel 1: Convert raw RGB buffer to Grayscale
__global__ void rgbToGrayscaleKernel(const unsigned char *d_rgb, unsigned char *d_gray, int width, int height, int channels)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int rgbIdx = (y * width + x) * channels;
        int grayIdx = y * width + x;

        unsigned char r = d_rgb[rgbIdx];
        unsigned char g = d_rgb[rgbIdx + 1];
        unsigned char b = d_rgb[rgbIdx + 2];

        // Standard luminance calculation
        d_gray[grayIdx] = static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);
    }
}

// Kernel 2: Sobel Edge Detection Kernel with shared memory tile
__global__ void sobelSharedKernel(const unsigned char *d_input, unsigned char *d_output, int width, int height)
{
    __shared__ float s_data[BLOCK_DIM_Y + 2][BLOCK_DIM_X + 2];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;

    // Load center pixel into shared memory
    int clampedX = max(0, min(x, width - 1));
    int clampedY = max(0, min(y, height - 1));
    s_data[ty + 1][tx + 1] = static_cast<float>(d_input[clampedY * width + clampedX]);

    // Load Halos/Aprons
    if (tx == 0)
    {
        int leftX = max(0, min(x - 1, width - 1));
        s_data[ty + 1][0] = static_cast<float>(d_input[clampedY * width + leftX]);
    }
    if (tx == BLOCK_DIM_X - 1 || x == width - 1)
    {
        int rightX = max(0, min(x + 1, width - 1));
        s_data[ty + 1][tx + 2] = static_cast<float>(d_input[clampedY * width + rightX]);
    }
    if (ty == 0)
    {
        int topY = max(0, min(y - 1, height - 1));
        s_data[0][tx + 1] = static_cast<float>(d_input[topY * width + clampedX]);
    }
    if (ty == BLOCK_DIM_Y - 1 || y == height - 1)
    {
        int bottomY = max(0, min(y + 1, height - 1));
        s_data[ty + 2][tx + 1] = static_cast<float>(d_input[bottomY * width + clampedX]);
    }

    // Load 4 Corners
    if (tx == 0 && ty == 0)
    {
        int leftX = max(0, min(x - 1, width - 1));
        int topY = max(0, min(y - 1, height - 1));
        s_data[0][0] = static_cast<float>(d_input[topY * width + leftX]);
    }
    if (tx == BLOCK_DIM_X - 1 && ty == 0)
    {
        int rightX = max(0, min(x + 1, width - 1));
        int topY = max(0, min(y - 1, height - 1));
        s_data[0][tx + 2] = static_cast<float>(d_input[topY * width + rightX]);
    }
    if (tx == 0 && ty == BLOCK_DIM_Y - 1)
    {
        int leftX = max(0, min(x - 1, width - 1));
        int bottomY = max(0, min(y + 1, height - 1));
        s_data[ty + 2][0] = static_cast<float>(d_input[bottomY * width + leftX]);
    }
    if (tx == BLOCK_DIM_X - 1 && ty == BLOCK_DIM_Y - 1)
    {
        int rightX = max(0, min(x + 1, width - 1));
        int bottomY = max(0, min(y + 1, height - 1));
        s_data[ty + 2][tx + 2] = static_cast<float>(d_input[bottomY * width + rightX]);
    }

    __syncthreads();

    if (x < width && y < height)
    {
        float sumX = 0.0f;
        float sumY = 0.0f;

#pragma unroll
        for (int ki = -1; ki <= 1; ++ki)
        {
#pragma unroll
            for (int kj = -1; kj <= 1; ++kj)
            {
                float px = s_data[ty + 1 + ki][tx + 1 + kj];
                int maskIdx = (ki + 1) * 3 + (kj + 1);
                sumX += px * c_sobelX[maskIdx];
                sumY += px * c_sobelY[maskIdx];
            }
        }

        float mag = sqrtf(sumX * sumX + sumY * sumY);
        d_output[y * width + x] = static_cast<unsigned char>(min(255.0f, max(0.0f, mag)));
    }
}

void initConstantMemory()
{
    float h_sobelX[9] = {-1.0f, 0.0f, 1.0f, -2.0f, 0.0f, 2.0f, -1.0f, 0.0f, 1.0f};
    float h_sobelY[9] = {-1.0f, -2.0f, -1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 2.0f, 1.0f};

    cudaMemcpyToSymbol(c_sobelX, h_sobelX, 9 * sizeof(float));
    cudaMemcpyToSymbol(c_sobelY, h_sobelY, 9 * sizeof(float));
}

int main(int argc, char **argv)
{
    std::string inputDir = (argc > 1) ? argv[1] : "./data/input/misc";
    std::string outputDir = (argc > 2) ? argv[2] : "./data/output";

    fs::create_directories(outputDir);
    fs::create_directories("./docs");
    initConstantMemory();

    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0)
    {
        std::cerr << "Error: No CUDA capable devices found!\n";
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    std::cout << "========================================================\n";
    std::cout << "Enterprise CUDA Image Processing Engine (Fixed)\n";
    std::cout << "Target Device      : " << prop.name << "\n";
    std::cout << "Compute Capability : " << prop.major << "." << prop.minor << "\n";
    std::cout << "SM Count           : " << prop.multiProcessorCount << "\n";
    std::cout << "Global Memory (MB) : " << (prop.totalGlobalMem / (1024 * 1024)) << "\n";
    std::cout << "========================================================\n";

    std::vector<std::string> files;
    for (const auto &entry : fs::directory_iterator(inputDir))
    {
        std::string ext = entry.path().extension().string();
        for (auto &c : ext)
            c = tolower(c);
        if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp")
        {
            files.push_back(entry.path().string());
        }
    }

    if (files.empty())
    {
        std::cerr << "Error: No supported images (.png, .jpg, .bmp) found in " << inputDir << "\n";
        return 1;
    }

    std::cout << "Discovered " << files.size() << " files in " << inputDir << ". Beginning execution...\n";

    std::ofstream csv("./docs/benchmark_telemetry.csv");
    csv << "image_index,filename,width,height,gpu_time_ms\n";

    cudaEvent_t startTotal, stopTotal;
    cudaEventCreate(&startTotal);
    cudaEventCreate(&stopTotal);
    cudaEventRecord(startTotal, 0);

    float cumulativeGpuTime = 0.0f;
    size_t processedCount = 0;

    for (size_t i = 0; i < files.size(); ++i)
    {
        int w = 0, h = 0, channels = 0;
        unsigned char *imgData = stbi_load(files[i].c_str(), &w, &h, &channels, 3);
        if (!imgData)
        {
            std::cerr << "Warning: Failed to load " << files[i] << ", skipping.\n";
            continue;
        }

        size_t rgbBytes = w * h * 3 * sizeof(unsigned char);
        size_t grayBytes = w * h * sizeof(unsigned char);

        unsigned char *d_rgb = nullptr;
        unsigned char *d_gray = nullptr;
        unsigned char *d_out = nullptr;

        cudaMalloc(&d_rgb, rgbBytes);
        cudaMalloc(&d_gray, grayBytes);
        cudaMalloc(&d_out, grayBytes);

        cudaEvent_t imgStart, imgStop;
        cudaEventCreate(&imgStart);
        cudaEventCreate(&imgStop);

        cudaEventRecord(imgStart);
        // Synchronous copy ensures data is fully transferred before kernel execution
        cudaMemcpy(d_rgb, imgData, rgbBytes, cudaMemcpyHostToDevice);

        dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
        dim3 grid((w + BLOCK_DIM_X - 1) / BLOCK_DIM_X, (h + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y);

        rgbToGrayscaleKernel<<<grid, block>>>(d_rgb, d_gray, w, h, 3);
        sobelSharedKernel<<<grid, block>>>(d_gray, d_out, w, h);

        std::vector<unsigned char> outBuffer(grayBytes);
        cudaMemcpy(outBuffer.data(), d_out, grayBytes, cudaMemcpyDeviceToHost);
        cudaEventRecord(imgStop);
        cudaEventSynchronize(imgStop);

        float itemMs = 0.0f;
        cudaEventElapsedTime(&itemMs, imgStart, imgStop);
        cumulativeGpuTime += itemMs;

        std::string stemName = fs::path(files[i]).stem().string();
        csv << processedCount << "," << stemName << "," << w << "," << h << "," << itemMs << "\n";

        std::string destPath = outputDir + "/edge_" + stemName + ".png";
        stbi_write_png(destPath.c_str(), w, h, 1, outBuffer.data(), w);

        if (processedCount == 0)
        {
            stbi_write_png("./docs/sample_input.png", w, h, 3, imgData, w * 3);
            stbi_write_png("./docs/sample_output.png", w, h, 1, outBuffer.data(), w);
        }

        stbi_image_free(imgData);
        cudaFree(d_rgb);
        cudaFree(d_gray);
        cudaFree(d_out);
        cudaEventDestroy(imgStart);
        cudaEventDestroy(imgStop);

        processedCount++;
    }

    cudaDeviceSynchronize();
    cudaEventRecord(stopTotal, 0);
    cudaEventSynchronize(stopTotal);

    float totalWallClockMs = 0.0f;
    cudaEventElapsedTime(&totalWallClockMs, startTotal, stopTotal);
    csv.close();

    float avgLatency = (processedCount > 0) ? (cumulativeGpuTime / processedCount) : 0.0f;
    float fps = (avgLatency > 0.0001f) ? (1000.0f / avgLatency) : 0.0f;

    std::cout << "\n================ EXECUTION SUMMARY ================\n";
    std::cout << "Successfully Processed : " << processedCount << " images\n";
    std::cout << "Total Wall Clock Time  : " << std::fixed << std::setprecision(3) << totalWallClockMs << " ms\n";
    std::cout << "Cumulative Kernel Time : " << std::fixed << std::setprecision(3) << cumulativeGpuTime << " ms\n";
    std::cout << "Average Latency/Image  : " << std::fixed << std::setprecision(3) << avgLatency << " ms\n";
    std::cout << "Effective Throughput   : " << std::fixed << std::setprecision(2) << fps << " FPS\n";
    std::cout << "Artifacts Written To   : " << outputDir << "/ and ./docs/\n";
    std::cout << "====================================================\n";

    cudaEventDestroy(startTotal);
    cudaEventDestroy(stopTotal);

    return 0;
}