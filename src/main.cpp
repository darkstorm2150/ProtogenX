#include <iostream>
#include <cuda_runtime.h>
#include "ModelLoader.h"

// Constants
const int GPU_DEVICE_ID = 0; // Default Device ID

int main(int argc, char** argv) {
	// Initialize CUDA
	cudaSetDevice(GPU_DEVICE_ID);

	// Check if GPU is avaiable
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, GPU_DEVICE_ID);
	if (deviceProp.major == 9999 && deviceProp.minor == 9999) {
		std::cerr << "No CUDA GPU detected!" << std::endl;
		return -1;
	} else {
		std::cout << "Using GPU: " << deviceProp.name << std::endl;
	}

	// Load the model (placeholder)
	ModelLoader loader("path/to/model.safetensor");
	if (!loader.load()) {
		std::cerr << "Failed to load model!" << std::endl;
		return -1;
	}

	// Future phases: processing the model, running inference, etc.
}