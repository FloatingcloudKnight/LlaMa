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

#include "SharedQueueTemp.h"
#include <buddy/Core/Container.h>
#include <buddy/LLM/TextContainer.h>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <shared_mutex>
#include <mutex>
#include <variant>
#include <vector>

using namespace buddy;

constexpr size_t MaxVocabSize = 32000;
constexpr size_t MaxTokenLength = 40;
constexpr size_t SubMaxTokenLength = 20;
constexpr size_t HiddenSize = 4096;
constexpr size_t HiddenSize0 = 128;
constexpr size_t HiddenSize1 = 41;

std::vector<MemRef<float, 1>> paramsContainers;
MemRef<float, 2> myMemRef1({MaxTokenLength, HiddenSize1});
MemRef<float, 3> myMemRef2({1, MaxTokenLength, HiddenSize0});
MemRef<float, 3> myMemRef3({1, MaxTokenLength, HiddenSize0});
MemRef<float, 3> concatResultContainer({1, MaxTokenLength, HiddenSize});

std::shared_mutex shared_mtx;

struct MemRefContainer {
  MemRef<float, 3> memRef3D0;
  MemRef<float, 2> memRef2D;
  MemRef<float, 3> memRef3D1;
  MemRef<float, 3> memRef3D2;

  MemRefContainer(MemRef<float, 3> m1, MemRef<float, 2> m2, MemRef<float, 3> m3,
                  MemRef<float, 3> m4)
      : memRef3D0(m1), memRef2D(m2), memRef3D1(m3), memRef3D2(m4) {}
};

// // 推理线程的输入结构
// struct InferenceInput {
//   MemRef<float, 3> inputMemRef;
//   MemRef<float, 2> mhaContainer0;
//   MemRef<float, 3> mhaContainer1;
//   MemRef<float, 3> mhaContainer2;

//   InferenceInput(MemRef<float, 3> inputtMemRef, MemRef<float, 2> mha0,
//                  MemRef<float, 3> mha1, MemRef<float, 3> mha2)
//       : inputMemRef(inputtMemRef), mhaContainer0(mha0), mhaContainer1(mha1),
//         mhaContainer2(mha2) {}
// };

class ConcatQueue : public SharedQueueTemp {
public:
  ConcatQueue() : SharedQueueTemp({"input0", "output0", "input1", "output1"}) {}
};
class InOutQueue : public SharedQueueTemp {
public:
  InOutQueue() : SharedQueueTemp({"input", "output"}) {}
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
                                        MemRef<float, 3> *);
extern "C" void _mlir_ciface_forward195(MemRef<float, 2> *, MemRef<float, 2> *,
                                        MemRef<float, 2> *);

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

void mainInferenceTokenThread(InOutQueue &inputQueue, ConcatQueue &concatQueue,
                              ConcatQueue &splitQueue,
                              const std::string inputName,
                              const std::string outputName,
                              const int threadNum) {
  MemRef<float, 3> subResultContainer({1, SubMaxTokenLength, HiddenSize});
  MemRef<float, 3> subResultContainer0({1, SubMaxTokenLength, HiddenSize});
  MemRef<float, 3> resultContainer({1, MaxTokenLength, HiddenSize});
  MemRef<float, 2> resultContainer2D({MaxTokenLength, HiddenSize});
  MemRef<float, 2> subResultContainer2D({SubMaxTokenLength, HiddenSize});
  MemRef<float, 2> subResultContainer2D0({SubMaxTokenLength, HiddenSize});
  MemRef<float, 3> input({1, SubMaxTokenLength, HiddenSize});
  while (true) {
    input = inputQueue.pop<MemRef<float, 3>>("input");

    // Execute the forward pass of the model.
    for (int m = 0; m < 32; m++) {
      _mlir_ciface_forward1(&subResultContainer, &paramsContainers[m * 6],
                            &input);
      concatQueue.push(inputName, std::move(subResultContainer));
      shared_mtx.lock_shared();
      _mlir_ciface_forward2(&resultContainer2D, &paramsContainers[m * 6 + 1 + threadNum],
                            &concatResultContainer, &myMemRef2, &myMemRef3,
                            &myMemRef1);
      shared_mtx.unlock_shared();
      splitQueue.push(inputName, std::move(resultContainer2D));
      subResultContainer2D = splitQueue.pop<MemRef<float, 2>>(outputName);
      _mlir_ciface_forward3(&input, &subResultContainer2D, &input);
      _mlir_ciface_forward1(&subResultContainer, &paramsContainers[m * 6 + 3],
                            &input);
      concatQueue.push(inputName, std::move(subResultContainer));
      shared_mtx.lock_shared();
      _mlir_ciface_forward5(&resultContainer2D, &paramsContainers[m * 6 + 4 + threadNum],
                            &concatResultContainer);
      shared_mtx.unlock_shared();
      splitQueue.push(inputName, std::move(resultContainer2D));
      subResultContainer2D = splitQueue.pop<MemRef<float, 2>>(outputName);
      _mlir_ciface_forward3(&input, &subResultContainer2D, &input);
    }
    inputQueue.push("output", std::move(input));
  }
}

// void subInferenceTokenThread(InOutQueue &inputQueue,
//                              ConcatQueue &concatQueue0) {
//   MemRef<float, 3> subResultContainer({1, SubMaxTokenLength, HiddenSize});
//   MemRef<float, 3> subResultContainer0({1, SubMaxTokenLength, HiddenSize});
//   MemRef<float, 3> resultContainer({1, MaxTokenLength, HiddenSize});
//   MemRef<float, 2> resultContainer2D({MaxTokenLength, HiddenSize});
//   MemRef<float, 2> subResultContainer2D({SubMaxTokenLength, HiddenSize});
//   MemRef<float, 2> subResultContainer2D0({SubMaxTokenLength, HiddenSize});
//   while (true) {
//     subResultContainer = inputQueue.pop<MemRef<float, 3>>("input");
//     MemRef<float, 3> input(std::move(subResultContainer));

//     // Execute the forward pass of the model.
//     for (int m = 0; m < 32; m++) {
//       _mlir_ciface_forward1(&subResultContainer, &paramsContainers[m * 6],
//                             &input);
//       concatQueue0.push("thread1", subResultContainer);
//       subResultContainer0 = concatQueue0.pop<MemRef<float, 3>>("thread0");
//       resultContainer.concatenateMemRefs(subResultContainer0,
//                                          subResultContainer, 1);
//       _mlir_ciface_forward2(&resultContainer2D, &paramsContainers[m * 6 + 2],
//                             &resultContainer, &myMemRef2, &myMemRef3,
//                             &myMemRef1);
//       subResultContainer2D0.splitMemRef(std::move(resultContainer2D),
//                                     subResultContainer2D,
//                                     subResultContainer2D0, 0, 20);
//       concatQueue0.push("thread1", subResultContainer2D);
//       subResultContainer2D = concatQueue0.pop<MemRef<float, 2>>("thread0");
//       subResultContainer2D0.addMemRef(subResultContainer2D0,
//                                       subResultContainer2D);
//       _mlir_ciface_forward3(&input, &subResultContainer2D0, &input);
//       _mlir_ciface_forward1(&subResultContainer, &paramsContainers[m * 6 +
//       3],
//                             &input);
//       concatQueue0.push("thread1", subResultContainer);
//       subResultContainer0 = concatQueue0.pop<MemRef<float, 3>>("thread0");
//       resultContainer.concatenateMemRefs(subResultContainer0,
//                                          subResultContainer, 1);
//       _mlir_ciface_forward5(&resultContainer2D, &paramsContainers[m * 6 + 5],
//                             &resultContainer);
//       subResultContainer2D0.splitMemRef(std::move(resultContainer2D),
//                                     subResultContainer2D,
//                                     subResultContainer2D0, 0, 20);
//       concatQueue0.push("thread1", subResultContainer2D);
//       subResultContainer2D = concatQueue0.pop<MemRef<float, 2>>("thread0");
//       subResultContainer2D0.addMemRef(subResultContainer2D0,
//                                       subResultContainer2D);
//       _mlir_ciface_forward3(&input, &subResultContainer2D0, &input);
//     }
//     inputQueue.push("output", input);
//   }
// }

void concatThreadTask(ConcatQueue &concatQueue) {
  MemRef<float, 3> subResultContainer({1, SubMaxTokenLength, HiddenSize});
  MemRef<float, 3> subResultContainer0({1, SubMaxTokenLength, HiddenSize});
  while (true) {
    shared_mtx.lock();
    subResultContainer = concatQueue.pop<MemRef<float, 3>>("input0");
    subResultContainer0 = concatQueue.pop<MemRef<float, 3>>("input1");
    _mlir_ciface_forward194(&concatResultContainer, &subResultContainer,
                                        &subResultContainer0);
    shared_mtx.unlock();
  }
}

void splitThreadTask(ConcatQueue &splitQueue) {
  MemRef<float, 2> subResultContainer[2] = {
      MemRef<float, 2>({SubMaxTokenLength, HiddenSize}, false, 0),
      MemRef<float, 2>({SubMaxTokenLength, HiddenSize}, false, 0)};
  MemRef<float, 2> resultContainer0({MaxTokenLength, HiddenSize});
  MemRef<float, 2> resultContainer1({MaxTokenLength, HiddenSize});
  while (true) {
    resultContainer0 = splitQueue.pop<MemRef<float, 2>>("input0");
    resultContainer1 = splitQueue.pop<MemRef<float, 2>>("input1");
    _mlir_ciface_forward195(subResultContainer, &resultContainer0,
                                        &resultContainer1);
    splitQueue.push("output0", std::move(subResultContainer[0]));
    splitQueue.push("output1", std::move(subResultContainer[1]));
  }
}

// -----------------------------------------------------------------------------
// LLaMA Inference Main Entry
// -----------------------------------------------------------------------------

int main() {
  /// Print the title of this example.
  const std::string title = "LLaMA 2 Inference Powered by Buddy Compiler";
  std::cout << "\033[33;1m" << title << "\033[0m" << std::endl;

  int split_group[] = {
      1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1,
      1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1,
      2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2,
      1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1,
      1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1,
      2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2,
      1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1,
      1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1};
  constexpr size_t param_size_group[] = {
      131072064, 4096, 33554432, 0, 4096,     67633152, 0, 4096, 33554432,
      0,         4096, 67633152, 0, 4096,     33554432, 0, 4096, 67633152,
      0,         4096, 33554432, 0, 4096,     67633152, 0, 4096, 33554432,
      0,         4096, 67633152, 0, 4096,     33554432, 0, 4096, 67633152,
      0,         4096, 33554432, 0, 4096,     67633152, 0, 4096, 33554432,
      0,         4096, 67633152, 0, 4096,     33554432, 0, 4096, 67633152,
      0,         4096, 33554432, 0, 4096,     67633152, 0, 4096, 33554432,
      0,         4096, 67633152, 0, 4096,     33554432, 0, 4096, 67633152,
      0,         4096, 33554432, 0, 4096,     67633152, 0, 4096, 33554432,
      0,         4096, 67633152, 0, 4096,     33554432, 0, 4096, 67633152,
      0,         4096, 33554432, 0, 4096,     67633152, 0, 4096, 33554432,
      0,         4096, 67633152, 0, 4096,     33554432, 0, 4096, 67633152,
      0,         4096, 33554432, 0, 4096,     67633152, 0, 4096, 33554432,
      0,         4096, 67633152, 0, 4096,     33554432, 0, 4096, 67633152,
      0,         4096, 33554432, 0, 4096,     67633152, 0, 4096, 33554432,
      0,         4096, 67633152, 0, 4096,     33554432, 0, 4096, 67633152,
      0,         4096, 33554432, 0, 4096,     67633152, 0, 4096, 33554432,
      0,         4096, 67633152, 0, 4096,     33554432, 0, 4096, 67633152,
      0,         4096, 33554432, 0, 4096,     67633152, 0, 4096, 33554432,
      0,         4096, 67633152, 0, 4096,     33554432, 0, 4096, 67633152,
      0,         4096, 33554432, 0, 4096,     67633152, 0, 4096, 33554432,
      0,         4096, 67633152, 0, 131076096};
  /// Define directories of vacabulary and parameter file.
  std::string llamaDir = LLAMA_DIS_EXAMPLE_PATH;
  std::string llamaBuildDir = LLAMA_EXAMPLE_BUILD_PATH;
  const std::string vocabDir = llamaDir + "/vocab.txt";

  std::vector<std::string> paramsDirs; // 用容器存储路径

  for (int i = 0; i < 194; i++) { // N 为需要生成的数量
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
  //   MemRef<float, 3> resultContainer[2] = {
  //       MemRef<float, 3>({1, MaxTokenLength, HiddenSize}, false, 0),
  //       MemRef<float, 3>({1, MaxTokenLength, MaxVocabSize}, false, 0)};
  MemRefContainer resultContainer(memRef3D0, memRef2D, memRef3D1, memRef3D2);
  MemRefContainer *resultContainerPtr = &resultContainer;
  MemRef<float, 3> resultMemRef({1, MaxTokenLength, HiddenSize});
  MemRef<float, 3> subResultMemRef0({1, SubMaxTokenLength, HiddenSize});
  MemRef<float, 3> subResultMemRef1({1, SubMaxTokenLength, HiddenSize});
  Text<size_t, 2> inputContainer(inputStr);

  /// Fill data into containers
  //  - Input: register vocabulary and tokenize the input string.
  //  - Output: register vocabulary.
  //  - Parameters: load parameters from the `arg0` file into the container.
  tokenizeInput(vocabDir, inputContainer);
  outputContainer.loadVocab(vocabDir);

  MemRef<float, 1> paramsContainer0({param_size_group[0]});
  loadParameters(paramsDirs[0], paramsContainer0);
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

  InOutQueue inputQueue0;
  InOutQueue inputQueue1;
  ConcatQueue concatQueue;
  ConcatQueue splitQueue;
  std::thread inferenceThread0(mainInferenceTokenThread, std::ref(inputQueue0),
                               std::ref(concatQueue), std::ref(splitQueue),
                               "input0", "output0", 0);
  std::thread inferenceThread1(mainInferenceTokenThread, std::ref(inputQueue1),
                               std::ref(concatQueue), std::ref(splitQueue),
                               "input1", "output1", 1);
  std::thread concatThread(concatThreadTask, std::ref(concatQueue));
  std::thread splitThread(splitThreadTask, std::ref(splitQueue));
  // std::thread inferenceThread1(subInferenceTokenThread,
  // std::ref(inputQueue1),
  //                              std::ref(concatQueue0));

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
    _mlir_ciface_forward0(resultContainerPtr, &paramsContainer0,
                          &inputContainer);
    if (i == 0){
      myMemRef1 = resultContainerPtr->memRef2D;
      myMemRef2 = resultContainerPtr->memRef3D1;
      myMemRef3 = resultContainerPtr->memRef3D2;
    }
    
    resultContainerPtr->memRef3D0.splitMemRef(
        std::move(resultContainerPtr->memRef3D0), subResultMemRef0,
        subResultMemRef1, 1, 20);

    inputQueue0.push("input", std::move(subResultMemRef0));
    inputQueue1.push("input", std::move(subResultMemRef1));

    subResultMemRef0 = inputQueue0.pop<MemRef<float, 3>>("output");
    subResultMemRef1 = inputQueue1.pop<MemRef<float, 3>>("output");

    resultMemRef.concatenateMemRefs(subResultMemRef0, subResultMemRef1,
                                    resultMemRef, 1);
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

  inferenceThread0.join();
  inferenceThread1.join();
  concatThread.join();
  splitThread.join();

  /// Print the final result
  std::cout << "\n\033[33;1m[Input]\033[0m " << inputStr << std::endl;
  std::cout << "\033[33;1m[Output]\033[0m " << outputContainer.revertLlama()
            << std::endl;

  return 0;
}
