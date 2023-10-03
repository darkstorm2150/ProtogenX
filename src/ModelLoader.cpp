#include "ModelLoader.h"

ModelLoader::ModelLoader(const std::string& filePath) : filePath_(filePath) {}

bool ModelLoader::load() {
	std::ifstream file(filePath_, std::ios::binary | std::ios::ate);
	if (!file.is_open()) {
		std::cerr << "Failed to open model file: " << filePath_ << std::endl;
		return false;
	}

	std::streamsize size = file.tellg();
	file.seekg(0, std::ios::beg);

	buffer_.resize(size);
	if (!file.read(buffer_.data(), size)) {
		std::cerr << "Failed to read model file: " << filePath_ << std::endl;
		return false;
	}

	return true;
}