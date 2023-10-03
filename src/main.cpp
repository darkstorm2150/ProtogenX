#include <iostream>
#ifdef _WIN32
#include <windows.h>
#undef max  // Remove the macro definition for max
#else
#include <cstdlib>
#endif
#include <cuda_runtime.h>
#include <filesystem>
#include "ModelLoader.h"
#include "CudaManager.h"
#include "InferenceRunner.h"

// Constants
const int GPU_DEVICE_ID = 0; // Default Device ID
const std::string MODEL_PATH = "C:\\Users\\thesi\\source\\repos\\Protogen X\\x64\\Debug\\Model.safetensors";

int main(int argc, char** argv) {
    CudaManager cudaManager(GPU_DEVICE_ID, MODEL_PATH);
    if (!cudaManager.initialize()) {
        std::cerr << "Failed to initialize CUDA." << std::endl;
        return -1;
    }

    ModelLoader loader(cudaManager.getModelPath());
    if (!loader.load()) {
        std::cerr << "Failed to load model!" << std::endl;
        return -1;
    }

    InferenceRunner inferenceRunner;  // Create an instance of InferenceRunner
    if (!inferenceRunner.run()) {  // Call the run method of InferenceRunner
        std::cerr << "Failed to run inference!" << std::endl;
        return -1;
    }

    std::cout << "Inference run successfully!" << std::endl;

    // Future phases: processing the model, running inference, etc.

    return 0;
}
