#ifndef MODELLOADER_H
#define MODELLOADER_H

#include <iostream>
#include <fstream>
#include <vector>

class ModelLoader {
public:
	ModelLoader(const std::string& filePath);
	bool load();

	// Future methods to extract model parameters, weights, etc.
	// ...

private:
	std::string filePath_;
	std::vector<char> buffer_;
};

#endif // MODELLOADER_H