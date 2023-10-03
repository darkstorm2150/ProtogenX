#include "CudaManager.h"

CudaManager::CudaManager(int deviceId, const std::string& modelPath)
    : deviceId(deviceId), modelPath(modelPath) {}

CudaManager::~CudaManager() {
    // Optional: Add any necessary cleanup
}

bool CudaManager::initialize() {
    return initializeCuda() && checkDeviceProperties() && checkFileExistence();
}

const std::string& CudaManager::getModelPath() const {
    return modelPath;
}

bool CudaManager::initializeCuda() {
    cudaError_t err = cudaSetDevice(deviceId);
    if (err != cudaSuccess) {
        std::cerr << "Failed to set CUDA device: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    return true;
}

bool CudaManager::checkDeviceProperties() {
    cudaDeviceProp deviceProp;
    cudaError_t err = cudaGetDeviceProperties(&deviceProp, deviceId);
    if (err != cudaSuccess) {
        std::cerr << "Failed to get device properties: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    if (deviceProp.major == 9999 && deviceProp.minor == 9999) {
        std::cerr << "No CUDA GPU detected!" << std::endl;
        return false;
    }
    if (deviceProp.major < 3) {
        std::cerr << "CUDA capability major version is less than 3, which might not support all CUDA features used in this program." << std::endl;
        return false;
    }
    std::cout << "Using GPU: " << deviceProp.name << std::endl;
    return true;
}

bool CudaManager::checkFileExistence() {
    if (!std::filesystem::exists(modelPath)) {
        std::cerr << "Model file does not exist at the specified path!" << std::endl;
        return false;
    }
    return true;
}
