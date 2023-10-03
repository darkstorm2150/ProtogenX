#ifndef CUDA_MANAGER_H
#define CUDA_MANAGER_H

#include <cuda_runtime.h>
#include <iostream>
#include <filesystem>
#include <string>

class CudaManager {
public:
    CudaManager(int deviceId, const std::string& modelPath);
    ~CudaManager();
    bool initialize();
    const std::string& getModelPath() const;
private:
    int deviceId;
    std::string modelPath;
    bool initializeCuda();
    bool checkDeviceProperties();
    bool checkFileExistence();
};

#endif // CUDA_MANAGER_H
