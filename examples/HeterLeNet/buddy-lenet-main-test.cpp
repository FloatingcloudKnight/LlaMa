//===- buddy-lenet-main.cpp -----------------------------------------------===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#include <buddy/Core/Container.h>
#include <buddy/DIP/ImgContainer.h>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

constexpr size_t ParamsSize0 = 156;
constexpr size_t ParamsSize1 = 0;
constexpr size_t ParamsSize2 = 2416;
constexpr size_t ParamsSize3 = 30840;
constexpr size_t ParamsSize4 = 11014;

const std::string ImgName = "1-28*28.png";

/// Declare LeNet forward function.
extern "C" void _mlir_ciface_forward0(MemRef<float, 4> *output0,
                                     MemRef<float, 1> *arg0,
                                     dip::Image<float, 4> *input);
extern "C" void _mlir_ciface_forward1(MemRef<float, 4> *output1,
                                     MemRef<float, 4> *output0);
extern "C" void _mlir_ciface_forward2(MemRef<float, 4> *output2,
                                     MemRef<float, 1> *arg2,
                                     MemRef<float, 4> *output1);
extern "C" void _mlir_ciface_forward3(MemRef<float, 2> *output3,
                                    MemRef<float, 1> *arg3,
                                    MemRef<float, 4> *output2);
extern "C" void _mlir_ciface_forward4(MemRef<float, 2> *output,
                                     MemRef<float, 1> *arg4,
                                     MemRef<float, 2> *output3);
/// Print [Log] label in bold blue format.
void printLogLabel() { std::cout << "\033[34;1m[Log] \033[0m"; }

/// Load parameters into data container.
void loadParameters(const std::string &paramFilePath,
                    MemRef<float, 1> &params) {
  const auto loadStart = std::chrono::high_resolution_clock::now();
  // Open the parameter file in binary mode.
  std::ifstream paramFile(paramFilePath, std::ios::in | std::ios::binary);
  if (!paramFile.is_open()) {
    throw std::runtime_error("[Error] Failed to open params file!");
  }
  printLogLabel();
  std::cout << "Loading params..." << std::endl;
  printLogLabel();
  // Print the canonical path of the parameter file.
  std::cout << "Params file: " << std::filesystem::canonical(paramFilePath)
            << std::endl;
  // Read the parameter data into the provided memory reference.
  paramFile.read(reinterpret_cast<char *>(params.getData()),
                 sizeof(float) * (params.getSize()));
  if (paramFile.fail()) {
    throw std::runtime_error("Error occurred while reading params file!");
  }
  paramFile.close();
  const auto loadEnd = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double, std::milli> loadTime =
      loadEnd - loadStart;
  printLogLabel();
  std::cout << "Params load time: " << (double)(loadTime.count()) / 1000
            << "s\n"
            << std::endl;
}

/// Softmax function to convert logits to probabilities.
void softmax(float *input, size_t size) {
  size_t i;
  float max_value = -INFINITY;
  double sum = 0.0;
  // Find the maximum value in the input array for numerical stability.
  for (i = 0; i < size; ++i) {
    if (max_value < input[i]) {
      max_value = input[i];
    }
  }
  // Calculate the sum of the exponentials of the input elements, normalized by
  // the max value.
  for (i = 0; i < size; ++i) {
    sum += exp(input[i] - max_value);
  }
  // Normalize the input array with the softmax calculation.
  for (i = 0; i < size; ++i) {
    input[i] = exp(input[i] - max_value) / sum;
  }
}

int main() {
  // Print the title of this example.
  const std::string title = "LeNet Inference Powered by Buddy Compiler";
  std::cout << "\033[33;1m" << title << "\033[0m" << std::endl;

  // Define the sizes of the output tensors.
  intptr_t sizesOutput[2] = {1, 10};
  intptr_t sizesOutput0[4] = {1, 6, 24, 24};
  intptr_t sizesOutput1[4] = {1, 6, 12, 12};
  intptr_t sizesOutput2[4] = {1, 16, 4, 4};
  intptr_t sizesOutput3[2] = {1, 120};


  // Create input and output containers for the image and model output.
  std::string lenetDir = LENET_EXAMPLE_PATH;
  std::string lenetBuildDir = LENET_EXAMPLE_BUILD_PATH;
  std::string imgPath = lenetDir + "/images/" + ImgName;
  dip::Image<float, 4> input(imgPath, dip::DIP_GRAYSCALE, true /* norm */);
  MemRef<float, 2> output(sizesOutput);
  MemRef<float, 4> output0(sizesOutput0, 0.0f);
  MemRef<float, 4> output1(sizesOutput1, 0.0f);
  MemRef<float, 4> output2(sizesOutput2, 0.0f);
  MemRef<float, 2> output3(sizesOutput3, 0.0f);


  // Load model parameters from the specified file.
  
  std::string paramsDir0 = lenetBuildDir + "/arg0.data";
  std::cout << paramsDir0 << std::endl;
  MemRef<float, 1> paramsContainer0({ParamsSize0});
  loadParameters(paramsDir0, paramsContainer0);
  // std::string paramsDir1 = lenetBuildDir + "/arg1.data";
  // MemRef<float, 1> paramsContainer1({ParamsSize1});
  // loadParameters(paramsDir1, paramsContainer1);
  std::string paramsDir2 = lenetBuildDir + "/arg2.data";
  MemRef<float, 1> paramsContainer2({ParamsSize2});
  loadParameters(paramsDir2, paramsContainer2);
  std::string paramsDir3 = lenetBuildDir + "/arg3.data";
  MemRef<float, 1> paramsContainer3({ParamsSize3});
  loadParameters(paramsDir3, paramsContainer3);
  std::string paramsDir4 = lenetBuildDir + "/arg4.data";
  MemRef<float, 1> paramsContainer4({ParamsSize4});
  loadParameters(paramsDir4, paramsContainer4);
  
  // Timing forward0
  auto t0_start = std::chrono::high_resolution_clock::now();
  _mlir_ciface_forward0(&output0, &paramsContainer0, &input);
  auto t0_end = std::chrono::high_resolution_clock::now();
  std::cout << "[Time] forward0: "
            << std::chrono::duration<double, std::milli>(t0_end - t0_start).count()
            << " ms" << std::endl;

  // Timing forward1
  auto t1_start = std::chrono::high_resolution_clock::now();
  _mlir_ciface_forward1(&output1, &output0);
  auto t1_end = std::chrono::high_resolution_clock::now();
  std::cout << "[Time] forward1: "
            << std::chrono::duration<double, std::milli>(t1_end - t1_start).count()
            << " ms" << std::endl;

  // Timing forward2
  auto t2_start = std::chrono::high_resolution_clock::now();
  _mlir_ciface_forward2(&output2, &paramsContainer2, &output1);
  auto t2_end = std::chrono::high_resolution_clock::now();
  std::cout << "[Time] forward2: "
            << std::chrono::duration<double, std::milli>(t2_end - t2_start).count()
            << " ms" << std::endl;


  // Timing forward4
  auto t4_start = std::chrono::high_resolution_clock::now();
  _mlir_ciface_forward3(&output3, &paramsContainer3, &output2);
  auto t4_end = std::chrono::high_resolution_clock::now();
  std::cout << "[Time] forward4: "
            << std::chrono::duration<double, std::milli>(t4_end - t4_start).count()
            << " ms" << std::endl;

  // Timing forward5
  auto t5_start = std::chrono::high_resolution_clock::now();
  _mlir_ciface_forward4(&output, &paramsContainer4, &output3);
  auto t5_end = std::chrono::high_resolution_clock::now();
  std::cout << "[Time] forward5: "
            << std::chrono::duration<double, std::milli>(t5_end - t5_start).count()
            << " ms" << std::endl;


  // Apply softmax to the output logits to get probabilities.
  auto out = output.getData();
  softmax(out, 10);

  // Find the classification and print the result.
  float maxVal = 0;
  float maxIdx = 0;
  for (int i = 0; i < 10; ++i) {
    if (out[i] > maxVal) {
      maxVal = out[i];
      maxIdx = i;
    }
  }

  std::cout << "Classification: " << maxIdx << std::endl;
  std::cout << "Probability: " << maxVal << std::endl;

  return 0;
}
