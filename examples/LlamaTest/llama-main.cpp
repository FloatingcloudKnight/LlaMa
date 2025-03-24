//===- llama-main.cpp -----------------------------------------------------===//
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
#include <buddy/LLM/TextContainer.h>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <iostream>

using namespace buddy;

constexpr size_t ParamsSize = 6738415680;
constexpr size_t MaxVocabSize = 32000;
constexpr size_t MaxTokenLength = 40;
constexpr size_t HiddenSize = 4096;

/// Declare LLaMA forward function.
// extern "C" void _mlir_ciface_forward(MemRef<float, 3> *, MemRef<float, 1> *,
//                                       Text<size_t, 2> *);
extern "C" void _mlir_ciface_forward0(MemRef<float, 3> *, MemRef<float, 1> *,
                                      Text<size_t, 2> *);
extern "C" void _mlir_ciface_forward1(MemRef<float, 3> *, MemRef<float, 1> *,
                                      MemRef<float, 3> *);
// extern "C" void _mlir_ciface_forward2(MemRef<float, 3> *, MemRef<float, 1> *,
//                                       MemRef<float, 3> *);
// extern "C" void _mlir_ciface_forward3(MemRef<float, 3> *, MemRef<float, 1> *,
//                                       MemRef<float, 3> *);
// extern "C" void _mlir_ciface_forward4(MemRef<float, 3> *, MemRef<float, 1> *,
//                                       MemRef<float, 3> *);
// extern "C" void _mlir_ciface_forward5(MemRef<float, 3> *, MemRef<float, 1> *,
//                                       MemRef<float, 3> *);
// -----------------------------------------------------------------------------
// Helper Functions
// -----------------------------------------------------------------------------

/// Capture input message.
void getUserInput(std::string &inputStr) {
  std::cout << "\nPlease send a message:" << std::endl;
  std::cout << ">>> ";
  getline(std::cin, inputStr);
  std::cout << std::endl;
}

/// Print [Log] label in bold blue format.
void printLogLabel() { std::cout << "\033[34;1m[Log] \033[0m"; }

/// Print information for each iteration.
void printIterInfo(size_t iterIdx, std::string str, double time) {
  std::cout << "\033[32;1m[Iteration " << iterIdx << "] \033[0m";
  std::cout << "Token: " << str << " | "
            << "Time: " << time << "s" << std::endl;
}

/// Tokenize input data in the container.
void tokenizeInput(const std::string &vocabFile,
                   Text<size_t, 2> &inputContainer) {
  printLogLabel();
  std::cout << "Vocab file: " << std::filesystem::canonical(vocabFile)
            << std::endl;
  const auto buddyTokenizeStart = std::chrono::high_resolution_clock::now();
  inputContainer.tokenizeLlama(vocabFile, MaxTokenLength);
  const auto buddyTokenizeEnd = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double, std::milli> buddyTokenizeTime =
      buddyTokenizeEnd - buddyTokenizeStart;
  printLogLabel();
  std::cout << "Tokenize time: " << buddyTokenizeTime.count() << "ms"
            << std::endl;
}

/// Load parameters into data container.
void loadParameters(const std::string &paramFilePath,
                    MemRef<float, 1> &params) {
  const auto loadStart = std::chrono::high_resolution_clock::now();
  std::ifstream paramFile(paramFilePath, std::ios::in | std::ios::binary);
  if (!paramFile.is_open()) {
    throw std::runtime_error("[Error] Failed to open params file!");
  }
  printLogLabel();
  std::cout << "Loading params..." << std::endl;
  printLogLabel();
  std::cout << "Params file: " << std::filesystem::canonical(paramFilePath)
            << std::endl;
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

/// Find the index of the max value.
int findMaxIndex(const float *start, const float *end) {
  return std::distance(start, std::max_element(start, end));
}

void verify(float *A, float *B, int batch, int size) {
  const std::string PASS = "\033[32mPASS\033[0m";
  const std::string FAIL = "\033[31mFAIL\033[0m";
  const double epsilon = 1e-6; // Tolerance for floating point comparison

  bool isPass = true;
  for (int i = 0; i < batch; ++i) {
    for (int j = 0; j < size; ++j) {
      int k = i * size + j;
      if (std::fabs(A[k] - B[k]) > epsilon) {
        std::cout << FAIL << std::endl;
        std::cout << "Batch=" << i << " Index=" << j << ":\tA[k]=" << A[k]
          << " B[k]=" << B[k] << std::endl;
        isPass = false;
        break;
      }
    }
    if (!isPass) {
      break;
    }
  }
  if (isPass) {
  std::cout << PASS << std::endl;
  }
}

// -----------------------------------------------------------------------------
// LLaMA Inference Main Entry
// -----------------------------------------------------------------------------

int main() {
  /// Print the title of this example.
  const std::string title = "LLaMA 2 Inference Powered by Buddy Compiler";
  std::cout << "\033[33;1m" << title << "\033[0m" << std::endl;

  /// Define directories of vacabulary and parameter file.
  std::string llamaDir = LLAMA_DIS_EXAMPLE_PATH;
  std::string llamaBuildDir = LLAMA_EXAMPLE_BUILD_PATH;
  const std::string vocabDir = llamaDir + "/vocab.txt";
  const std::string paramsDir = llamaBuildDir + "/arg0.data";

  /// Get user message.
  std::string inputStr;
  getUserInput(inputStr);

  /// Initialize data containers
  //  - Input container.
  //  - Result container
  //  - Parameters container.
  MemRef<float, 3> resultContainer({1, MaxTokenLength, HiddenSize}, false, 0);
  MemRef<float, 3> resultContainer0({1, MaxTokenLength, HiddenSize}, false, 0);
  Text<size_t, 2> inputContainer(inputStr);
  MemRef<float, 1> paramsContainer({ParamsSize});

  /// Fill data into containers
  //  - Input: register vocabulary and tokenize the input string.
  //  - Output: register vocabulary.
  //  - Parameters: load parameters from the `arg0` file into the container.
  tokenizeInput(vocabDir, inputContainer);
  loadParameters(paramsDir, paramsContainer);

  /// Run LLaMA Inference
  //  - Perform the forward function.
  //  - Find and append the generated token.
  //  - Continue iterating until the terminal condition is met.
  int generateLen = MaxTokenLength - inputContainer.getTokenCnt();
  const auto inferenceStart = std::chrono::high_resolution_clock::now();

  // Execute the forward pass of the model.
  _mlir_ciface_forward0(&resultContainer, &paramsContainer, &inputContainer);
  _mlir_ciface_forward1(&resultContainer, &paramsContainer, &resultContainer);

  const auto inferenceEnd = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double, std::milli> inferenceTime =
      inferenceEnd - inferenceStart;

  // _mlir_ciface_forward(&resultContainer0, &paramsContainer, &inputContainer);
  // Print the generated token and inference time.
  printIterInfo(0, "null", inferenceTime.count() / 1000);

  verify(resultContainer0.getData(), resultContainer.getData(), 1, MaxTokenLength * HiddenSize);

  /// Print the final result
  std::cout << "\n\033[33;1m[Input]\033[0m " << inputStr << std::endl;
  
  free(resultContainer.release());
  free(resultContainer0.release());

  return 0;
}
