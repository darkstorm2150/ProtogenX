#include <iostream>
#include <cuda_runtime.h>
#include <filesystem>  // for file existence check
#include "ModelLoader.h"

// Constants
const int GPU_DEVICE_ID = 0; // Default Device ID

int main(int argc, char** argv) {
    // Initialize CUDA
    cudaError_t err = cudaSetDevice(GPU_DEVICE_ID);
    if (err != cudaSuccess) {
        std::cerr << "Failed to set CUDA device: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    // Check if GPU is available
    cudaDeviceProp deviceProp;
    err = cudaGetDeviceProperties(&deviceProp, GPU_DEVICE_ID);
    if (err != cudaSuccess) {
        std::cerr << "Failed to get device properties: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    if (deviceProp.major == 9999 && deviceProp.minor == 9999) {
        std::cerr << "No CUDA GPU detected!" << std::endl;
        return -1;
    }
    else {
        if (deviceProp.major < 3) {
            std::cerr << "CUDA capability major version is less than 3, which might not support all CUDA features used in this program." << std::endl;
            return -1;
        }
        std::cout << "Using GPU: " << deviceProp.name << std::endl;
    }

    // Verify File Existence
    std::string modelPath = "C:\\Users\\thesi\\source\\repos\\Protogen X\\x64\\Debug\\Model.safetensors";
    if (!std::filesystem::exists(modelPath)) {
        std::cerr << "Model file does not exist at the specified path!" << std::endl;
        return -1;
    }

    // Load the model
    ModelLoader loader(modelPath);
    if (!loader.load()) {
        std::cerr << "Failed to load model!" << std::endl;
        return -1;
    }

    std::cout << "Model loaded successfully!" << std::endl;

    // Future phases: processing the model, running inference, etc.
}
