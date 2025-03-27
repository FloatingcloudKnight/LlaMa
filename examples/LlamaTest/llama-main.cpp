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
#include <variant>
#include <vector>
#include <string>

using namespace buddy;

constexpr size_t ParamsSize0 = 131072064;
constexpr size_t ParamsSize1 = 202383360;
constexpr size_t ParamsSize2 = 131076096;
constexpr size_t MaxVocabSize = 32000;
constexpr size_t MaxTokenLength = 40;
constexpr size_t HiddenSize = 4096;
constexpr size_t HiddenSize0 = 128;
constexpr size_t HiddenSize1 = 41;

struct MemRefContainer {
  MemRef<float, 3> memRef3D0;
  MemRef<float, 2> memRef2D;
  MemRef<float, 3> memRef3D1;
  MemRef<float, 3> memRef3D2;

  MemRefContainer(MemRef<float, 3> m1, MemRef<float, 2> m2, MemRef<float, 3> m3, MemRef<float, 3> m4)
      : memRef3D0(m1), memRef2D(m2), memRef3D1(m3), memRef3D2(m4) {}
};

/// Declare LLaMA forward function.
// extern "C" void _mlir_ciface_forward(MemRef<float, 3> *, MemRef<float, 1> *,
//                                       Text<size_t, 2> *);
extern "C" void _mlir_ciface_forward0(MemRefContainer *, MemRef<float, 1> *,
                                      Text<size_t, 2> *);
extern "C" void _mlir_ciface_forward1(MemRef<float, 3> *, MemRef<float, 1> *,
                                      MemRef<float, 3> *,
                                      MemRef<float, 3> *,
                                      MemRef<float, 3> *,
                                      MemRef<float, 2> *);
extern "C" void _mlir_ciface_forward33(MemRef<float, 3> *, MemRef<float, 1> *,
                                      MemRef<float, 3> *);
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
  // const std::string paramsDir0 = llamaBuildDir + "/arg0.data";
  // const std::string paramsDir1 = llamaBuildDir + "/arg1.data";
  // const std::string paramsDir2 = llamaBuildDir + "/arg33.data";

  std::vector<std::string> paramsDirs;  // 用容器存储路径

  for (int i = 0; i < 34; i++) {  // N 为需要生成的数量
      // 使用 emplace_back 直接构造字符串，避免拷贝
      paramsDirs.emplace_back(
          llamaBuildDir + "/arg" + std::to_string(i) + ".data"
      );
  }

  /// Get user message.
  std::string inputStr;
  getUserInput(inputStr);

  /// Initialize data containers
  //  - Input container.
  //  - Result container
  //  - Parameters container.
  Text<size_t, 2> outputContainer;
  MemRef<float, 3> myMemRef1({1, MaxTokenLength, HiddenSize});
  MemRef<float, 2> myMemRef2({MaxTokenLength, HiddenSize1});
  MemRef<float, 3> myMemRef3({1, MaxTokenLength, HiddenSize0});
  MemRef<float, 3> myMemRef4({1, MaxTokenLength, HiddenSize0});
  MemRefContainer resultContainer(myMemRef1, myMemRef2, myMemRef3, myMemRef4);
  MemRefContainer *resultContainerPtr = &resultContainer;
  MemRef<float, 3> resultContainer0({1, MaxTokenLength, HiddenSize});
  Text<size_t, 2> inputContainer(inputStr);
  std::vector<MemRef<float, 1>> paramsContainers;
  MemRef<float, 1> paramsContainer0({ParamsSize0});
  for(int i = 0; i < 32; i++){
    MemRef<float, 1> paramsContainer1({ParamsSize1});
    paramsContainers.push_back(paramsContainer1);
  }
  MemRef<float, 1> paramsContainer2({ParamsSize2});
  /// Fill data into containers
  //  - Input: register vocabulary and tokenize the input string.
  //  - Output: register vocabulary.
  //  - Parameters: load parameters from the `arg0` file into the container.
  tokenizeInput(vocabDir, inputContainer);
  outputContainer.loadVocab(vocabDir);
  loadParameters(paramsDirs[0], paramsContainer0);
  for(int i = 0; i < 32; i++){
    loadParameters(paramsDirs[i+1], paramsContainers[i]);
  }
  loadParameters(paramsDirs[33], paramsContainer2);
  /// Run LLaMA Inference
  //  - Perform the forward function.
  //  - Find and append the generated token.
  //  - Continue iterating until the terminal condition is met.
  /// Run LLaMA Inference
  //  - Perform the forward function.
  //  - Find and append the generated token.
  //  - Continue iterating until the terminal condition is met.
  int generateLen = MaxTokenLength - inputContainer.getTokenCnt();
  for (int i = 0; i < generateLen; i++) {
    const auto inferenceStart = std::chrono::high_resolution_clock::now();
    // Execute the forward pass of the model.
    _mlir_ciface_forward0(resultContainerPtr, &paramsContainer0, &inputContainer);
    resultContainer0 = resultContainerPtr->memRef3D0;
    auto resultContainer1 = resultContainerPtr->memRef2D;
    auto resultContainer2 = resultContainerPtr->memRef3D1;
    auto resultContainer3 = resultContainerPtr->memRef3D2;
    _mlir_ciface_forward1(&resultContainer0, &paramsContainers[0], &resultContainer0, &resultContainer2, &resultContainer3, &resultContainer1);
    for(int m = 1; m < 32; m++){
      _mlir_ciface_forward1(&resultContainer0, &paramsContainers[m], &resultContainer0, &resultContainer2, &resultContainer3, &resultContainer1);
    }
    _mlir_ciface_forward33(&resultContainer0, &paramsContainer2, &resultContainer0);

    const auto inferenceEnd = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double, std::milli> inferenceTime =
        inferenceEnd - inferenceStart;

    // Determine the generated token.
    int tokenIndex = inputContainer.getTokenCnt() - 1;
    const float *startPtr =
        resultContainer0.getData() + tokenIndex * MaxVocabSize;
    const float *endPtr = startPtr + MaxVocabSize;
    int maxIndex = findMaxIndex(startPtr, endPtr);
    std::string tok = inputContainer.getStr(maxIndex);
    // Print the generated token and inference time.
    printIterInfo(i, tok, inferenceTime.count() / 1000);

    // Stop if a separator token (2, </s>) or line break token (13 <0x0A>) is
    // generated.
    if (maxIndex == 2) {
      break;
    }
    // Append the generated token into the input and output container.
    inputContainer.appendTokenIdx(maxIndex);
    outputContainer.appendTokenIdx(maxIndex);
    
    free(myMemRef1.release());
    free(myMemRef2.release());
    free(myMemRef3.release());
    free(myMemRef4.release());
    free(resultContainer0.release());
  }

  /// Print the final result
  std::cout << "\n\033[33;1m[Input]\033[0m " << inputStr << std::endl;
  std::cout << "\033[33;1m[Output]\033[0m " << outputContainer.revertLlama()
            << std::endl;

  return 0;
}
