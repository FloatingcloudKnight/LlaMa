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

#include <atomic>
#include <buddy/Core/Container.h>
#include <buddy/LLM/TextContainer.h>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <variant>
#include <vector>

using namespace buddy;

constexpr size_t MaxVocabSize = 32000;
constexpr size_t MaxTokenLength = 40;
constexpr size_t SubMaxTokenLength = 5;
constexpr size_t HiddenSize = 4096;
constexpr size_t HiddenSize0 = 128;
constexpr size_t HiddenSize1 = 41;

std::mutex mutex;
std::mutex mutex0;
std::mutex mutex1;
std::mutex mutex2;
std::mutex mutex3;
std::mutex mutex4;
std::mutex mutex5;
std::mutex mutex6;
std::mutex mutex7;
std::condition_variable cv;
std::condition_variable subcv;
bool subdone0 = true, subdone1 = true, subdone2 = true, subdone3 = true,
     subdone4 = true, subdone5 = true, subdone6 = true, subdone7 = true;

struct MemRefContainer {
  MemRef<float, 3> memRef3D0;
  MemRef<float, 2> memRef2D;
  MemRef<float, 3> memRef3D1;
  MemRef<float, 3> memRef3D2;

  MemRefContainer(MemRef<float, 3> m1, MemRef<float, 2> m2, MemRef<float, 3> m3,
                  MemRef<float, 3> m4)
      : memRef3D0(std::move(m1)), memRef2D(std::move(m2)),
        memRef3D1(std::move(m3)), memRef3D2(std::move(m4)) {}
};

/// Declare LLaMa forward function.
extern "C" void _mlir_ciface_forward0(MemRefContainer *, MemRef<float, 1> *,
                                      Text<size_t, 2> *);
extern "C" void _mlir_ciface_forward1(MemRef<float, 3> *, MemRef<float, 1> *,
                                      MemRef<float, 3> *);
extern "C" void _mlir_ciface_forward2(MemRef<float, 2> *, MemRef<float, 1> *,
                                      MemRef<float, 3> *, MemRef<float, 3> *,
                                      MemRef<float, 3> *, MemRef<float, 2> *);
extern "C" void _mlir_ciface_forward3(MemRef<float, 3> *, MemRef<float, 2> *,
                                      MemRef<float, 3> *);
extern "C" void _mlir_ciface_forward5(MemRef<float, 2> *, MemRef<float, 1> *,
                                      MemRef<float, 3> *);
extern "C" void _mlir_ciface_forward193(MemRef<float, 3> *, MemRef<float, 1> *,
                                        MemRef<float, 3> *);
extern "C" void _mlir_ciface_forward194(MemRef<float, 3> *, MemRef<float, 3> *,
                                        MemRef<float, 3> *, MemRef<float, 3> *,
                                        MemRef<float, 3> *, MemRef<float, 3> *,
                                        MemRef<float, 3> *, MemRef<float, 3> *,
                                        MemRef<float, 3> *);
extern "C" void _mlir_ciface_forward195(MemRef<float, 2> *, MemRef<float, 2> *,
                                        MemRef<float, 2> *, MemRef<float, 2> *,
                                        MemRef<float, 2> *, MemRef<float, 2> *,
                                        MemRef<float, 2> *, MemRef<float, 2> *,
                                        MemRef<float, 2> *);
extern "C" void _mlir_ciface_forward196(MemRef<float, 3> *, MemRef<float, 3> *);

static void printDebugLogFile(const std::vector<float> &vec,
                              const std::string &filename) {
  std::string logDir = std::string(LLAMA_DIS_EXAMPLE_PATH) + "/" + filename;
  std::ofstream file(logDir, std::ios::app);
  if (!file.is_open()) {
    std::cerr << "无法打开文件: " << logDir << std::endl;
    return;
  }

  file << "[DEBUG]: " << std::endl;
  // 设置输出格式（固定小数点，保留6位小数）
  file << "[";
  file << std::fixed << std::setprecision(6);
  for (size_t i = 0; i < vec.size(); ++i) {
    file << vec[i];
    if (i < vec.size() - 1)
      file << " ";
  }
  file << "]" << '\n';
  file.close();
}

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
    std::cout << paramFilePath << std::endl;
    throw std::runtime_error("[Error] Failed to open params file!");
  }
  // printLogLabel();
  // std::cout << "Loading params..." << std::endl;
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
  // printLogLabel();
  // std::cout << "Params load time: " << (double)(loadTime.count()) / 1000
  //           << "s\n"
  //           << std::endl;
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

  int split_group[] = {
      1, 1, 8, 1, 1, 8, 1, 1, 8, 1, 1, 8, 1, 1, 8, 1, 1, 8, 1, 1, 8, 1, 1, 8, 1,
      1, 8, 1, 1, 8, 1, 1, 8, 1, 1, 8, 1, 1, 8, 1, 1, 8, 1, 1, 8, 1, 1, 8, 1, 1,
      8, 1, 1, 8, 1, 1, 8, 1, 1, 8, 1, 1, 8, 1, 1, 8, 1, 1, 8, 1, 1, 8, 1, 1, 8,
      1, 1, 8, 1, 1, 8, 1, 1, 8, 1, 1, 8, 1, 1, 8, 1, 1, 8, 1, 1, 8, 1, 1, 8, 1,
      1, 8, 1, 1, 8, 1, 1, 8, 1, 1, 8, 1, 1, 8, 1, 1, 8, 1, 1, 8, 1, 1, 8, 1, 1,
      8, 1, 1, 8, 1, 1, 8, 1, 1, 8, 1, 1, 8, 1, 1, 8, 1, 1, 8, 1, 1, 8, 1, 1, 8,
      1, 1, 8, 1, 1, 8, 1, 1, 8, 1, 1, 8, 1, 1, 8, 1, 1, 8, 1, 1, 8, 1, 1, 8, 1,
      1, 8, 1, 1, 8, 1, 1, 8, 1, 1, 8, 1, 1, 8, 1, 1, 8, 1, 1};
  constexpr size_t param_size_group[] = {
      131072064, 4096, 8388608,  0, 4096,     16908288, 0, 4096, 8388608,
      0,         4096, 16908288, 0, 4096,     8388608,  0, 4096, 16908288,
      0,         4096, 8388608,  0, 4096,     16908288, 0, 4096, 8388608,
      0,         4096, 16908288, 0, 4096,     8388608,  0, 4096, 16908288,
      0,         4096, 8388608,  0, 4096,     16908288, 0, 4096, 8388608,
      0,         4096, 16908288, 0, 4096,     8388608,  0, 4096, 16908288,
      0,         4096, 8388608,  0, 4096,     16908288, 0, 4096, 8388608,
      0,         4096, 16908288, 0, 4096,     8388608,  0, 4096, 16908288,
      0,         4096, 8388608,  0, 4096,     16908288, 0, 4096, 8388608,
      0,         4096, 16908288, 0, 4096,     8388608,  0, 4096, 16908288,
      0,         4096, 8388608,  0, 4096,     16908288, 0, 4096, 8388608,
      0,         4096, 16908288, 0, 4096,     8388608,  0, 4096, 16908288,
      0,         4096, 8388608,  0, 4096,     16908288, 0, 4096, 8388608,
      0,         4096, 16908288, 0, 4096,     8388608,  0, 4096, 16908288,
      0,         4096, 8388608,  0, 4096,     16908288, 0, 4096, 8388608,
      0,         4096, 16908288, 0, 4096,     8388608,  0, 4096, 16908288,
      0,         4096, 8388608,  0, 4096,     16908288, 0, 4096, 8388608,
      0,         4096, 16908288, 0, 4096,     8388608,  0, 4096, 16908288,
      0,         4096, 8388608,  0, 4096,     16908288, 0, 4096, 8388608,
      0,         4096, 16908288, 0, 4096,     8388608,  0, 4096, 16908288,
      0,         4096, 8388608,  0, 4096,     16908288, 0, 4096, 8388608,
      0,         4096, 16908288, 0, 131076096};
  /// Define directories of vacabulary and parameter file.
  std::string llamaDir = LLAMA_DIS_EXAMPLE_PATH;
  std::string llamaBuildDir = LLAMA_EXAMPLE_BUILD_PATH;
  const std::string vocabDir = llamaDir + "/vocab.txt";

  std::vector<std::string> paramsDirs; // 用容器存储路径

  for (int i = 0; i < sizeof(split_group) / sizeof(split_group[0]);
       i++) { // N 为需要生成的数量
    for (int j = 0; j < split_group[i]; j++) {
      // 使用 emplace_back 直接构造字符串，避免拷贝
      paramsDirs.emplace_back(llamaBuildDir + "/subgraph" + std::to_string(i) +
                              "_arg" + std::to_string(j) + ".data");
    }
  }

  /// Get user message.
  std::string inputStr;
  getUserInput(inputStr);

  /// Initialize data containers
  //  - Input container.
  //  - Result container
  //  - Parameters container.
  Text<size_t, 2> outputContainer;
  MemRef<float, 3> memRef3D0({1, MaxTokenLength, HiddenSize});
  MemRef<float, 2> memRef2D({MaxTokenLength, HiddenSize1});
  MemRef<float, 3> memRef3D1({1, MaxTokenLength, HiddenSize0});
  MemRef<float, 3> memRef3D2({1, MaxTokenLength, HiddenSize0});
  MemRefContainer resultContainer(memRef3D0, memRef2D, memRef3D1, memRef3D2);
  MemRefContainer *resultContainerPtr = &resultContainer;
  MemRef<float, 3> resultMemRef({1, MaxTokenLength, HiddenSize});
  MemRef<float, 3> sub3DMemRef0({1, SubMaxTokenLength, HiddenSize});
  MemRef<float, 2> resultMemRef2D0({MaxTokenLength, HiddenSize});
  MemRef<float, 3> sub3DMemRef1({1, SubMaxTokenLength, HiddenSize});
  MemRef<float, 2> resultMemRef2D1({MaxTokenLength, HiddenSize});
  MemRef<float, 3> sub3DMemRef2({1, SubMaxTokenLength, HiddenSize});
  MemRef<float, 2> resultMemRef2D2({MaxTokenLength, HiddenSize});
  MemRef<float, 3> sub3DMemRef3({1, SubMaxTokenLength, HiddenSize});
  MemRef<float, 2> resultMemRef2D3({MaxTokenLength, HiddenSize});
  MemRef<float, 3> sub3DMemRef4({1, SubMaxTokenLength, HiddenSize});
  MemRef<float, 2> resultMemRef2D4({MaxTokenLength, HiddenSize});
  MemRef<float, 3> sub3DMemRef5({1, SubMaxTokenLength, HiddenSize});
  MemRef<float, 2> resultMemRef2D5({MaxTokenLength, HiddenSize});
  MemRef<float, 3> sub3DMemRef6({1, SubMaxTokenLength, HiddenSize});
  MemRef<float, 2> resultMemRef2D6({MaxTokenLength, HiddenSize});
  MemRef<float, 3> sub3DMemRef7({1, SubMaxTokenLength, HiddenSize});
  MemRef<float, 2> resultMemRef2D7({MaxTokenLength, HiddenSize});
  MemRef<float, 2> subResultContainer2D[8] = {
      MemRef<float, 2>({SubMaxTokenLength, HiddenSize}, false, 0),
      MemRef<float, 2>({SubMaxTokenLength, HiddenSize}, false, 0),
      MemRef<float, 2>({SubMaxTokenLength, HiddenSize}, false, 0),
      MemRef<float, 2>({SubMaxTokenLength, HiddenSize}, false, 0),
      MemRef<float, 2>({SubMaxTokenLength, HiddenSize}, false, 0),
      MemRef<float, 2>({SubMaxTokenLength, HiddenSize}, false, 0),
      MemRef<float, 2>({SubMaxTokenLength, HiddenSize}, false, 0),
      MemRef<float, 2>({SubMaxTokenLength, HiddenSize}, false, 0)};
  MemRef<float, 3> subResultContainer[8] = {
      MemRef<float, 3>({1, SubMaxTokenLength, HiddenSize}, false, 0),
      MemRef<float, 3>({1, SubMaxTokenLength, HiddenSize}, false, 0),
      MemRef<float, 3>({1, SubMaxTokenLength, HiddenSize}, false, 0),
      MemRef<float, 3>({1, SubMaxTokenLength, HiddenSize}, false, 0),
      MemRef<float, 3>({1, SubMaxTokenLength, HiddenSize}, false, 0),
      MemRef<float, 3>({1, SubMaxTokenLength, HiddenSize}, false, 0),
      MemRef<float, 3>({1, SubMaxTokenLength, HiddenSize}, false, 0),
      MemRef<float, 3>({1, SubMaxTokenLength, HiddenSize}, false, 0)};
  Text<size_t, 2> inputContainer(inputStr);

  /// Fill data into containers
  //  - Input: register vocabulary and tokenize the input string.
  //  - Output: register vocabulary.
  //  - Parameters: load parameters from the `arg0` file into the container.
  tokenizeInput(vocabDir, inputContainer);
  outputContainer.loadVocab(vocabDir);

  MemRef<float, 1> paramsContainer0({param_size_group[0]});
  loadParameters(paramsDirs[0], paramsContainer0);
  std::vector<MemRef<float, 1>> paramsContainers;
  int params_count = 1;
  for (int i = 1; i < 193; i++) {
    for (int j = 0; j < split_group[i]; j++) {
      if (param_size_group[i] > 0) {
        MemRef<float, 1> paramsContainer1({param_size_group[i]});
        loadParameters(paramsDirs[params_count], paramsContainer1);
        paramsContainers.push_back(paramsContainer1);
      }
      params_count++;
    }
  }
  MemRef<float, 1> paramsContainer2({param_size_group[193]});
  loadParameters(paramsDirs[params_count], paramsContainer2);

  // std::thread inferenceThread1(subInferenceTokenThread,
  // std::ref(inputQueue1),
  //                              std::ref(concatQueue0));

  /// Run LLaMA Inference
  //  - Perform the forward function.
  //  - Find and append the generated token.
  //  - Continue iterating until the terminal condition is met.
  // int generateLen = MaxTokenLength - inputContainer.getTokenCnt();
  int generateLen = 5; // For testing, we set it to 1.

  std::thread t0([&]() {
    for (int i = 0; i < generateLen; i++) {
      {
        std::unique_lock<std::mutex> lock(mutex0);
        cv.wait(lock, [] { return !subdone0; });
      }
      for (int m = 0; m < 32; m++) {
        _mlir_ciface_forward1(&sub3DMemRef0, &paramsContainers[m * 18],
                              &subResultContainer[0]);
        {
          std::lock_guard<std::mutex> lock(mutex0);
          subdone0 = true;
        }
        if (subdone0 && subdone1 && subdone2 && subdone3 && subdone4 &&
            subdone5 && subdone6 && subdone7)
          subcv.notify_one();
        {
          std::unique_lock<std::mutex> lock(mutex0);
          cv.wait(lock, [] { return !subdone0; });
        }
        _mlir_ciface_forward2(&resultMemRef2D0, &paramsContainers[m * 18 + 1],
                              &resultMemRef, &resultContainerPtr->memRef3D1,
                              &resultContainerPtr->memRef3D2,
                              &resultContainerPtr->memRef2D);
        {
          std::lock_guard<std::mutex> lock(mutex0);
          subdone0 = true;
        }
        if (subdone0 && subdone1 && subdone2 && subdone3 && subdone4 &&
            subdone5 && subdone6 && subdone7)
          subcv.notify_one();
        {
          std::unique_lock<std::mutex> lock(mutex0);
          cv.wait(lock, [] { return !subdone0; });
        }
        _mlir_ciface_forward3(&subResultContainer[0], &subResultContainer2D[0],
                              &subResultContainer[0]);
        _mlir_ciface_forward1(&sub3DMemRef0, &paramsContainers[m * 18 + 9],
                              &subResultContainer[0]);
        {
          std::lock_guard<std::mutex> lock(mutex0);
          subdone0 = true;
        }
        if (subdone0 && subdone1 && subdone2 && subdone3 && subdone4 &&
            subdone5 && subdone6 && subdone7)
          subcv.notify_one();
        {
          std::unique_lock<std::mutex> lock(mutex0);
          cv.wait(lock, [] { return !subdone0; });
        }
        _mlir_ciface_forward5(&resultMemRef2D0, &paramsContainers[m * 18 + 10],
                              &resultMemRef);
        {
          std::lock_guard<std::mutex> lock(mutex0);
          subdone0 = true;
        }
        if (subdone0 && subdone1 && subdone2 && subdone3 && subdone4 &&
            subdone5 && subdone6 && subdone7)
          subcv.notify_one();
        {
          std::unique_lock<std::mutex> lock(mutex0);
          cv.wait(lock, [] { return !subdone0; });
        }
        _mlir_ciface_forward3(&subResultContainer[0], &subResultContainer2D[0],
                              &subResultContainer[0]);
      }
      {
        std::lock_guard<std::mutex> lock(mutex0);
        subdone0 = true;
      }
      if (subdone0 && subdone1 && subdone2 && subdone3 && subdone4 &&
          subdone5 && subdone6 && subdone7)
        subcv.notify_one();
    }
  });

  // 线程1：处理 subResultContainer[1]
  std::thread t1([&]() {
    for (int i = 0; i < generateLen; i++) {
      {
        std::unique_lock<std::mutex> lock(mutex1);
        cv.wait(lock, [] { return !subdone1; });
      }
      for (int m = 0; m < 32; m++) {
        _mlir_ciface_forward1(&sub3DMemRef1, &paramsContainers[m * 18],
                              &subResultContainer[1]);
        {
          std::lock_guard<std::mutex> lock(mutex1);
          subdone1 = true;
        }
        if (subdone0 && subdone1 && subdone2 && subdone3 && subdone4 &&
            subdone5 && subdone6 && subdone7)
          subcv.notify_one();
        {
          std::unique_lock<std::mutex> lock(mutex1);
          cv.wait(lock, [] { return !subdone1; });
        }
        _mlir_ciface_forward2(&resultMemRef2D1, &paramsContainers[m * 18 + 2],
                              &resultMemRef, &resultContainerPtr->memRef3D1,
                              &resultContainerPtr->memRef3D2,
                              &resultContainerPtr->memRef2D);
        {
          std::lock_guard<std::mutex> lock(mutex1);
          subdone1 = true;
        }
        if (subdone0 && subdone1 && subdone2 && subdone3 && subdone4 &&
            subdone5 && subdone6 && subdone7)
          subcv.notify_one();
        {
          std::unique_lock<std::mutex> lock(mutex1);
          cv.wait(lock, [] { return !subdone1; });
        }
        _mlir_ciface_forward3(&subResultContainer[1], &subResultContainer2D[1],
                              &subResultContainer[1]);
        _mlir_ciface_forward1(&sub3DMemRef1, &paramsContainers[m * 18 + 9],
                              &subResultContainer[1]);
        {
          std::lock_guard<std::mutex> lock(mutex1);
          subdone1 = true;
        }
        if (subdone0 && subdone1 && subdone2 && subdone3 && subdone4 &&
            subdone5 && subdone6 && subdone7)
          subcv.notify_one();
        {
          std::unique_lock<std::mutex> lock(mutex1);
          cv.wait(lock, [] { return !subdone1; });
        }
        _mlir_ciface_forward5(&resultMemRef2D1, &paramsContainers[m * 18 + 11],
                              &resultMemRef);
        {
          std::lock_guard<std::mutex> lock(mutex1);
          subdone1 = true;
        }

        if (subdone0 && subdone1 && subdone2 && subdone3 && subdone4 &&
            subdone5 && subdone6 && subdone7)
          subcv.notify_one();
        {
          std::unique_lock<std::mutex> lock(mutex1);
          cv.wait(lock, [] { return !subdone1; });
        }
        _mlir_ciface_forward3(&subResultContainer[1], &subResultContainer2D[1],
                              &subResultContainer[1]);
      }
      {
        std::lock_guard<std::mutex> lock(mutex1);
        subdone1 = true;
      }
      if (subdone0 && subdone1 && subdone2 && subdone3 && subdone4 &&
          subdone5 && subdone6 && subdone7)
        subcv.notify_one();
    }
  });

  // 线程2：处理 subResultContainer[2]
  std::thread t2([&]() {
    for (int i = 0; i < generateLen; i++) {
      {
        std::unique_lock<std::mutex> lock(mutex2);
        cv.wait(lock, [] { return !subdone2; });
      }
      for (int m = 0; m < 32; m++) {
        _mlir_ciface_forward1(&sub3DMemRef2, &paramsContainers[m * 18],
                              &subResultContainer[2]);
        {
          std::lock_guard<std::mutex> lock(mutex2);
          subdone2 = true;
        }
        if (subdone0 && subdone1 && subdone2 && subdone3 && subdone4 &&
            subdone5 && subdone6 && subdone7)
          subcv.notify_one();
        {
          std::unique_lock<std::mutex> lock(mutex2);
          cv.wait(lock, [] { return !subdone2; });
        }
        _mlir_ciface_forward2(&resultMemRef2D2, &paramsContainers[m * 18 + 3],
                              &resultMemRef, &resultContainerPtr->memRef3D1,
                              &resultContainerPtr->memRef3D2,
                              &resultContainerPtr->memRef2D);
        {
          std::lock_guard<std::mutex> lock(mutex2);
          subdone2 = true;
        }
        if (subdone0 && subdone1 && subdone2 && subdone3 && subdone4 &&
            subdone5 && subdone6 && subdone7)
          subcv.notify_one();
        {
          std::unique_lock<std::mutex> lock(mutex2);
          cv.wait(lock, [] { return !subdone2; });
        }
        _mlir_ciface_forward3(&subResultContainer[2], &subResultContainer2D[2],
                              &subResultContainer[2]);
        _mlir_ciface_forward1(&sub3DMemRef2, &paramsContainers[m * 18 + 9],
                              &subResultContainer[2]);
        {
          std::lock_guard<std::mutex> lock(mutex2);
          subdone2 = true;
        }
        if (subdone0 && subdone1 && subdone2 && subdone3 && subdone4 &&
            subdone5 && subdone6 && subdone7)
          subcv.notify_one();
        {
          std::unique_lock<std::mutex> lock(mutex2);
          cv.wait(lock, [] { return !subdone2; });
        }
        _mlir_ciface_forward5(&resultMemRef2D2, &paramsContainers[m * 18 + 12],
                              &resultMemRef);
        {
          std::lock_guard<std::mutex> lock(mutex2);
          subdone2 = true;
        }
        if (subdone0 && subdone1 && subdone2 && subdone3 && subdone4 &&
            subdone5 && subdone6 && subdone7)
          subcv.notify_one();
        {
          std::unique_lock<std::mutex> lock(mutex2);
          cv.wait(lock, [] { return !subdone2; });
        }
        _mlir_ciface_forward3(&subResultContainer[2], &subResultContainer2D[2],
                              &subResultContainer[2]);
      }
      {
        std::lock_guard<std::mutex> lock(mutex2);
        subdone2 = true;
      }
      if (subdone0 && subdone1 && subdone2 && subdone3 && subdone4 &&
          subdone5 && subdone6 && subdone7)
        subcv.notify_one();
    }
  });
  // 线程3：处理 subResultContainer[3]
  std::thread t3([&]() {
    for (int i = 0; i < generateLen; i++) {
      {
        std::unique_lock<std::mutex> lock(mutex3);
        cv.wait(lock, [] { return !subdone3; });
      }
      for (int m = 0; m < 32; m++) {
        _mlir_ciface_forward1(&sub3DMemRef3, &paramsContainers[m * 18],
                              &subResultContainer[3]);
        {
          std::lock_guard<std::mutex> lock(mutex3);
          subdone3 = true;
        }

        if (subdone0 && subdone1 && subdone2 && subdone3 && subdone4 &&
            subdone5 && subdone6 && subdone7)
          subcv.notify_one();
        {
          std::unique_lock<std::mutex> lock(mutex3);
          cv.wait(lock, [] { return !subdone3; });
        }
        _mlir_ciface_forward2(&resultMemRef2D3, &paramsContainers[m * 18 + 4],
                              &resultMemRef, &resultContainerPtr->memRef3D1,
                              &resultContainerPtr->memRef3D2,
                              &resultContainerPtr->memRef2D);
        {
          std::lock_guard<std::mutex> lock(mutex3);
          subdone3 = true;
        }
        if (subdone0 && subdone1 && subdone2 && subdone3 && subdone4 &&
            subdone5 && subdone6 && subdone7)
          subcv.notify_one();
        {
          std::unique_lock<std::mutex> lock(mutex3);
          cv.wait(lock, [] { return !subdone3; });
        }
        _mlir_ciface_forward3(&subResultContainer[3], &subResultContainer2D[3],
                              &subResultContainer[3]);
        _mlir_ciface_forward1(&sub3DMemRef3, &paramsContainers[m * 18 + 9],
                              &subResultContainer[3]);
        {
          std::lock_guard<std::mutex> lock(mutex3);
          subdone3 = true;
        }
        if (subdone0 && subdone1 && subdone2 && subdone3 && subdone4 &&
            subdone5 && subdone6 && subdone7)
          subcv.notify_one();
        {
          std::unique_lock<std::mutex> lock(mutex3);
          cv.wait(lock, [] { return !subdone3; });
        }
        _mlir_ciface_forward5(&resultMemRef2D3, &paramsContainers[m * 18 + 13],
                              &resultMemRef);
        {
          std::lock_guard<std::mutex> lock(mutex3);
          subdone3 = true;
        }
        if (subdone0 && subdone1 && subdone2 && subdone3 && subdone4 &&
            subdone5 && subdone6 && subdone7)
          subcv.notify_one();
        {
          std::unique_lock<std::mutex> lock(mutex3);
          cv.wait(lock, [] { return !subdone3; });
        }
        _mlir_ciface_forward3(&subResultContainer[3], &subResultContainer2D[3],
                              &subResultContainer[3]);
      }
      {
        std::lock_guard<std::mutex> lock(mutex3);
        subdone3 = true;
      }
      if (subdone0 && subdone1 && subdone2 && subdone3 && subdone4 &&
          subdone5 && subdone6 && subdone7)
        subcv.notify_one();
    }
  });

  std::thread t4([&]() {
    for (int i = 0; i < generateLen; i++) {
      {
        std::unique_lock<std::mutex> lock(mutex4);
        cv.wait(lock, [] { return !subdone4; });
      }
      for (int m = 0; m < 32; m++) {
        _mlir_ciface_forward1(&sub3DMemRef4, &paramsContainers[m * 18],
                              &subResultContainer[4]);
        {
          std::lock_guard<std::mutex> lock(mutex4);
          subdone4 = true;
        }
        if (subdone0 && subdone1 && subdone2 && subdone3 && subdone4 &&
            subdone5 && subdone6 && subdone7)
          subcv.notify_one();
        {
          std::unique_lock<std::mutex> lock(mutex4);
          cv.wait(lock, [] { return !subdone4; });
        }
        _mlir_ciface_forward2(&resultMemRef2D4, &paramsContainers[m * 18 + 5],
                              &resultMemRef, &resultContainerPtr->memRef3D1,
                              &resultContainerPtr->memRef3D2,
                              &resultContainerPtr->memRef2D);
        {
          std::lock_guard<std::mutex> lock(mutex4);
          subdone4 = true;
        }
        if (subdone0 && subdone1 && subdone2 && subdone3 && subdone4 &&
            subdone5 && subdone6 && subdone7)
          subcv.notify_one();
        {
          std::unique_lock<std::mutex> lock(mutex4);
          cv.wait(lock, [] { return !subdone4; });
        }
        _mlir_ciface_forward3(&subResultContainer[4], &subResultContainer2D[4],
                              &subResultContainer[4]);
        _mlir_ciface_forward1(&sub3DMemRef4, &paramsContainers[m * 18 + 9],
                              &subResultContainer[4]);
        {
          std::lock_guard<std::mutex> lock(mutex4);
          subdone4 = true;
        }
        if (subdone0 && subdone1 && subdone2 && subdone3 && subdone4 &&
            subdone5 && subdone6 && subdone7)
          subcv.notify_one();
        {
          std::unique_lock<std::mutex> lock(mutex4);
          cv.wait(lock, [] { return !subdone4; });
        }
        _mlir_ciface_forward5(&resultMemRef2D4, &paramsContainers[m * 18 + 14],
                              &resultMemRef);
        {
          std::lock_guard<std::mutex> lock(mutex4);
          subdone4 = true;
        }
        if (subdone0 && subdone1 && subdone2 && subdone3 && subdone4 &&
            subdone5 && subdone6 && subdone7)
          subcv.notify_one();
        {
          std::unique_lock<std::mutex> lock(mutex4);
          cv.wait(lock, [] { return !subdone4; });
        }
        _mlir_ciface_forward3(&subResultContainer[4], &subResultContainer2D[4],
                              &subResultContainer[4]);
      }
      {
        std::lock_guard<std::mutex> lock(mutex4);
        subdone4 = true;
      }
      if (subdone0 && subdone1 && subdone2 && subdone3 && subdone4 &&
          subdone5 && subdone6 && subdone7)
        subcv.notify_one();
    }
  });

  // 线程5：处理 subResultContainer[5]
  std::thread t5([&]() {
    for (int i = 0; i < generateLen; i++) {
      {
        std::unique_lock<std::mutex> lock(mutex5);
        cv.wait(lock, [] { return !subdone5; });
      }
      for (int m = 0; m < 32; m++) {
        _mlir_ciface_forward1(&sub3DMemRef5, &paramsContainers[m * 18],
                              &subResultContainer[5]);
        {
          std::lock_guard<std::mutex> lock(mutex5);
          subdone5 = true;
        }
        if (subdone0 && subdone1 && subdone2 && subdone3 && subdone4 &&
            subdone5 && subdone6 && subdone7)
          subcv.notify_one();
        {
          std::unique_lock<std::mutex> lock(mutex5);
          cv.wait(lock, [] { return !subdone5; });
        }
        _mlir_ciface_forward2(&resultMemRef2D5, &paramsContainers[m * 18 + 6],
                              &resultMemRef, &resultContainerPtr->memRef3D1,
                              &resultContainerPtr->memRef3D2,
                              &resultContainerPtr->memRef2D);
        {
          std::lock_guard<std::mutex> lock(mutex5);
          subdone5 = true;
        }
        if (subdone0 && subdone1 && subdone2 && subdone3 && subdone4 &&
            subdone5 && subdone6 && subdone7)
          subcv.notify_one();
        {
          std::unique_lock<std::mutex> lock(mutex5);
          cv.wait(lock, [] { return !subdone5; });
        }
        _mlir_ciface_forward3(&subResultContainer[5], &subResultContainer2D[5],
                              &subResultContainer[5]);
        _mlir_ciface_forward1(&sub3DMemRef5, &paramsContainers[m * 18 + 9],
                              &subResultContainer[5]);
        {
          std::lock_guard<std::mutex> lock(mutex5);
          subdone5 = true;
        }
        if (subdone0 && subdone1 && subdone2 && subdone3 && subdone4 &&
            subdone5 && subdone6 && subdone7)
          subcv.notify_one();
        {
          std::unique_lock<std::mutex> lock(mutex5);
          cv.wait(lock, [] { return !subdone5; });
        }
        _mlir_ciface_forward5(&resultMemRef2D5, &paramsContainers[m * 18 + 15],
                              &resultMemRef);
        {
          std::lock_guard<std::mutex> lock(mutex5);
          subdone5 = true;
        }

        if (subdone0 && subdone1 && subdone2 && subdone3 && subdone4 &&
            subdone5 && subdone6 && subdone7)
          subcv.notify_one();
        {
          std::unique_lock<std::mutex> lock(mutex5);
          cv.wait(lock, [] { return !subdone5; });
        }
        _mlir_ciface_forward3(&subResultContainer[5], &subResultContainer2D[5],
                              &subResultContainer[5]);
      }
      {
        std::lock_guard<std::mutex> lock(mutex5);
        subdone5 = true;
      }
      if (subdone0 && subdone1 && subdone2 && subdone3 && subdone4 &&
          subdone5 && subdone6 && subdone7)
        subcv.notify_one();
    }
  });

  // 线程6：处理 subResultContainer[6]
  std::thread t6([&]() {
    for (int i = 0; i < generateLen; i++) {
      {
        std::unique_lock<std::mutex> lock(mutex6);
        cv.wait(lock, [] { return !subdone6; });
      }
      for (int m = 0; m < 32; m++) {
        _mlir_ciface_forward1(&sub3DMemRef6, &paramsContainers[m * 18],
                              &subResultContainer[6]);
        {
          std::lock_guard<std::mutex> lock(mutex6);
          subdone6 = true;
        }
        if (subdone0 && subdone1 && subdone2 && subdone3 && subdone4 &&
            subdone5 && subdone6 && subdone7)
          subcv.notify_one();
        {
          std::unique_lock<std::mutex> lock(mutex6);
          cv.wait(lock, [] { return !subdone6; });
        }
        _mlir_ciface_forward2(&resultMemRef2D6, &paramsContainers[m * 18 + 7],
                              &resultMemRef, &resultContainerPtr->memRef3D1,
                              &resultContainerPtr->memRef3D2,
                              &resultContainerPtr->memRef2D);
        {
          std::lock_guard<std::mutex> lock(mutex6);
          subdone6 = true;
        }
        if (subdone0 && subdone1 && subdone2 && subdone3 && subdone4 &&
            subdone5 && subdone6 && subdone7)
          subcv.notify_one();
        {
          std::unique_lock<std::mutex> lock(mutex6);
          cv.wait(lock, [] { return !subdone6; });
        }
        _mlir_ciface_forward3(&subResultContainer[6], &subResultContainer2D[6],
                              &subResultContainer[6]);
        _mlir_ciface_forward1(&sub3DMemRef6, &paramsContainers[m * 18 + 9],
                              &subResultContainer[6]);
        {
          std::lock_guard<std::mutex> lock(mutex6);
          subdone6 = true;
        }
        if (subdone0 && subdone1 && subdone2 && subdone3 && subdone4 &&
            subdone5 && subdone6 && subdone7)
          subcv.notify_one();
        {
          std::unique_lock<std::mutex> lock(mutex6);
          cv.wait(lock, [] { return !subdone6; });
        }
        _mlir_ciface_forward5(&resultMemRef2D6, &paramsContainers[m * 18 + 16],
                              &resultMemRef);
        {
          std::lock_guard<std::mutex> lock(mutex6);
          subdone6 = true;
        }
        if (subdone0 && subdone1 && subdone2 && subdone3 && subdone4 &&
            subdone5 && subdone6 && subdone7)
          subcv.notify_one();
        {
          std::unique_lock<std::mutex> lock(mutex6);
          cv.wait(lock, [] { return !subdone6; });
        }
        _mlir_ciface_forward3(&subResultContainer[6], &subResultContainer2D[6],
                              &subResultContainer[6]);
      }
      {
        std::lock_guard<std::mutex> lock(mutex6);
        subdone6 = true;
      }
      if (subdone0 && subdone1 && subdone2 && subdone3 && subdone4 &&
          subdone5 && subdone6 && subdone7)
        subcv.notify_one();
    }
  });
  // 线程7：处理 subResultContainer[7]
  std::thread t7([&]() {
    for (int i = 0; i < generateLen; i++) {
      {
        std::unique_lock<std::mutex> lock(mutex7);
        cv.wait(lock, [] { return !subdone7; });
      }
      for (int m = 0; m < 32; m++) {
        _mlir_ciface_forward1(&sub3DMemRef7, &paramsContainers[m * 18],
                              &subResultContainer[7]);
        {
          std::lock_guard<std::mutex> lock(mutex7);
          subdone7 = true;
        }

        if (subdone0 && subdone1 && subdone2 && subdone3 && subdone4 &&
            subdone5 && subdone6 && subdone7)
          subcv.notify_one();
        {
          std::unique_lock<std::mutex> lock(mutex7);
          cv.wait(lock, [] { return !subdone7; });
        }
        _mlir_ciface_forward2(&resultMemRef2D7, &paramsContainers[m * 18 + 8],
                              &resultMemRef, &resultContainerPtr->memRef3D1,
                              &resultContainerPtr->memRef3D2,
                              &resultContainerPtr->memRef2D);
        {
          std::lock_guard<std::mutex> lock(mutex7);
          subdone7 = true;
        }
        if (subdone0 && subdone1 && subdone2 && subdone3 && subdone4 &&
            subdone5 && subdone6 && subdone7)
          subcv.notify_one();
        {
          std::unique_lock<std::mutex> lock(mutex7);
          cv.wait(lock, [] { return !subdone7; });
        }
        _mlir_ciface_forward3(&subResultContainer[7], &subResultContainer2D[7],
                              &subResultContainer[7]);
        _mlir_ciface_forward1(&sub3DMemRef7, &paramsContainers[m * 18 + 9],
                              &subResultContainer[7]);
        {
          std::lock_guard<std::mutex> lock(mutex7);
          subdone7 = true;
        }
        if (subdone0 && subdone1 && subdone2 && subdone3 && subdone4 &&
            subdone5 && subdone6 && subdone7)
          subcv.notify_one();
        {
          std::unique_lock<std::mutex> lock(mutex7);
          cv.wait(lock, [] { return !subdone7; });
        }
        _mlir_ciface_forward5(&resultMemRef2D7, &paramsContainers[m * 18 + 17],
                              &resultMemRef);
        {
          std::lock_guard<std::mutex> lock(mutex7);
          subdone7 = true;
        }
        if (subdone0 && subdone1 && subdone2 && subdone3 && subdone4 &&
            subdone5 && subdone6 && subdone7)
          subcv.notify_one();
        {
          std::unique_lock<std::mutex> lock(mutex7);
          cv.wait(lock, [] { return !subdone7; });
        }
        _mlir_ciface_forward3(&subResultContainer[7], &subResultContainer2D[7],
                              &subResultContainer[7]);
      }
      {
        std::lock_guard<std::mutex> lock(mutex7);
        subdone7 = true;
      }
      if (subdone0 && subdone1 && subdone2 && subdone3 && subdone4 &&
          subdone5 && subdone6 && subdone7)
        subcv.notify_one();
    }
  });

  for (int i = 0; i < generateLen; i++) {
    const auto inferenceStart = std::chrono::high_resolution_clock::now();

    // Execute the forward pass of the model.
    _mlir_ciface_forward0(resultContainerPtr, &paramsContainer0,
                          &inputContainer);

    // resultContainerPtr->memRef3D0.splitMemRef(
    //     std::move(resultContainerPtr->memRef3D0), subResultContainer[0],
    //     subResultContainer[1], subResultContainer[2], subResultContainer[3],
    //     1, 10);
    _mlir_ciface_forward196(subResultContainer, &resultContainerPtr->memRef3D0);
    {
      std::lock_guard<std::mutex> lock(mutex);
      subdone0 = false;
      subdone1 = false;
      subdone2 = false;
      subdone3 = false;
      subdone4 = false;
      subdone5 = false;
      subdone6 = false;
      subdone7 = false;
    }
    cv.notify_all();
    for (int m = 0; m < 32; m++) {
      // after forward1
      {
        std::unique_lock<std::mutex> lock(mutex);
        subcv.wait(lock, [] {
          return subdone0 && subdone1 && subdone2 && subdone3 && subdone4 &&
                 subdone5 && subdone6 && subdone7;
        });
      }
      _mlir_ciface_forward194(&resultMemRef, &sub3DMemRef0, &sub3DMemRef1,
                              &sub3DMemRef2, &sub3DMemRef3, &sub3DMemRef4, &sub3DMemRef5, &sub3DMemRef6, &sub3DMemRef7);

      {
        std::lock_guard<std::mutex> lock(mutex);
        subdone0 = false;
        subdone1 = false;
        subdone2 = false;
        subdone3 = false;
        subdone4 = false;
        subdone5 = false;
        subdone6 = false;
        subdone7 = false;
      }
      cv.notify_all();
      // after forward2
      {
        std::unique_lock<std::mutex> lock(mutex);
        subcv.wait(lock, [] {
          return subdone0 && subdone1 && subdone2 && subdone3 && subdone4 &&
                 subdone5 && subdone6 && subdone7;
        });
      }
      _mlir_ciface_forward195(subResultContainer2D, &resultMemRef2D0,
                              &resultMemRef2D1, &resultMemRef2D2,
                              &resultMemRef2D3, &resultMemRef2D4,
                              &resultMemRef2D5, &resultMemRef2D6,
                              &resultMemRef2D7);
      {
        std::lock_guard<std::mutex> lock(mutex);
        subdone0 = false;
        subdone1 = false;
        subdone2 = false;
        subdone3 = false;
        subdone4 = false;
        subdone5 = false;
        subdone6 = false;
        subdone7 = false;
      }
      cv.notify_all();
      // after forward1 again
      {
        std::unique_lock<std::mutex> lock(mutex);
        subcv.wait(lock, [] {
          return subdone0 && subdone1 && subdone2 && subdone3 && subdone4 &&
                 subdone5 && subdone6 && subdone7;
        });
      }
      _mlir_ciface_forward194(&resultMemRef, &sub3DMemRef0, &sub3DMemRef1,
                              &sub3DMemRef2, &sub3DMemRef3, &sub3DMemRef4, &sub3DMemRef5,
                              &sub3DMemRef6, &sub3DMemRef7);
      {
        std::lock_guard<std::mutex> lock(mutex);
        subdone0 = false;
        subdone1 = false;
        subdone2 = false;
        subdone3 = false;
        subdone4 = false;
        subdone5 = false;
        subdone6 = false;
        subdone7 = false;
      }
      cv.notify_all();
      // after forward5
      {
        std::unique_lock<std::mutex> lock(mutex);
        subcv.wait(lock, [] {
          return subdone0 && subdone1 && subdone2 && subdone3 && subdone4 &&
                 subdone5 && subdone6 && subdone7;
        });
      }
      _mlir_ciface_forward195(subResultContainer2D, &resultMemRef2D0,
                              &resultMemRef2D1, &resultMemRef2D2,
                              &resultMemRef2D3, &resultMemRef2D4,
                              &resultMemRef2D5, &resultMemRef2D6,
                              &resultMemRef2D7);
      {
        std::lock_guard<std::mutex> lock(mutex);
        subdone0 = false;
        subdone1 = false;
        subdone2 = false;
        subdone3 = false;
        subdone4 = false;
        subdone5 = false;
        subdone6 = false;
        subdone7 = false;
      }
      cv.notify_all();
    }
    // Wait for the threads to finish processing.
    {
      std::unique_lock<std::mutex> lock(mutex);
      subcv.wait(lock, [] {
        return subdone0 && subdone1 && subdone2 && subdone3 && subdone4 &&
               subdone5 && subdone6 && subdone7;
      });
    }
    _mlir_ciface_forward194(&resultMemRef, &subResultContainer[0],
                            &subResultContainer[1], &subResultContainer[2],
                            &subResultContainer[3], &subResultContainer[4],
                            &subResultContainer[5], &subResultContainer[6],
                            &subResultContainer[7]);
    _mlir_ciface_forward193(&resultMemRef, &paramsContainer2, &resultMemRef);
    const auto inferenceEnd = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double, std::milli> inferenceTime =
        inferenceEnd - inferenceStart;

    // Determine the generated token.
    int tokenIndex = inputContainer.getTokenCnt() - 1;
    const float *startPtr = resultMemRef.getData() + tokenIndex * MaxVocabSize;
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
  }

  t0.join();
  t1.join();
  t2.join();
  t3.join();
  t4.join();
  t5.join();
  t6.join();
  t7.join();

  /// Print the final result
  std::cout << "\n\033[33;1m[Input]\033[0m " << inputStr << std::endl;
  std::cout << "\033[33;1m[Output]\033[0m " << outputContainer.revertLlama()
            << std::endl;

  return 0;
}
