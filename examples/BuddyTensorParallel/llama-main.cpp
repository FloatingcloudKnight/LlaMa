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
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <variant>
#include <vector>
#include <algorithm>
#include <iterator> 
#include <cstring>

using namespace buddy;

constexpr size_t MaxVocabSize = 151936;
constexpr size_t MaxTokenLength = 1024;
constexpr size_t SubMaxTokenLength = 512;

constexpr size_t NUM_LAYERS = 56;
constexpr size_t HiddenSize = 128;
constexpr size_t HiddenSize0 = 1536;
constexpr size_t HeadNum = 2;

struct MemRefContainerPrefill0 {
  MemRef<float, 3> data;
  MemRef<float, 2> mask;
  MemRef<float, 3> sin;
  MemRef<float, 3> cos;

  MemRefContainerPrefill0(MemRef<float, 3> m1, MemRef<float, 2> m2, MemRef<float, 3> m3,
                  MemRef<float, 3> m4)
      : data(m1), mask(m2), sin(m3), cos(m4) {}
};


struct MemRefContainerPrefill2 {
  MemRef<float, 4> kcache;
  MemRef<float, 4> vcache;
  MemRef<float, 2> data;

  MemRefContainerPrefill2(MemRef<float, 4> m1, MemRef<float, 4> m2, MemRef<float, 2> m3)
      : kcache(m1), vcache(m2), data(m3) {}
};




/// Declare LLaMA forward function.
// extern "C" void _mlir_ciface_forward(MemRef<float, 3> *, MemRef<float, 1> *,
//                                       Text<size_t, 2> *);
extern "C" void _mlir_ciface_forward0_prefill(MemRefContainerPrefill0 *, MemRef<float, 1> *,
                                      Text<size_t, 2> *);
extern "C" void _mlir_ciface_forward1_prefill(MemRef<float, 3> *, MemRef<float, 1> *,
                                      MemRef<float, 3> *);
extern "C" void _mlir_ciface_forward2_prefill( MemRefContainerPrefill2 *, MemRef<float, 1> *,
                                      MemRef<float, 3> *, MemRef<float, 3> *,
                                      MemRef<float, 3> *, MemRef<float, 4> *);
extern "C" void _mlir_ciface_forward3_prefill(MemRef<float, 3> *, MemRef<float, 2> *,
                                      MemRef<float, 3> *);
extern "C" void _mlir_ciface_forward5_prefill(MemRef<float, 2> *, MemRef<float, 1> *,
                                      MemRef<float, 3> *);
extern "C" void _mlir_ciface_forward169_prefill(MemRef<float, 3> *, MemRef<float, 1> *,
                                        MemRef<float, 3> *);

extern "C" void _mlir_ciface_forward0_decode(MemRefContainerPrefill0 *, MemRef<float, 1> *, MemRef<long long, 2> *,
                                      MemRef<long long, 1> *);
extern "C" void _mlir_ciface_forward1_decode(MemRef<float, 3> *, MemRef<float, 1> *,
                                      MemRef<float, 3> *);
extern "C" void _mlir_ciface_forward2_decode( MemRefContainerPrefill2 *, MemRef<float, 1> *, MemRef<long long, 1>*,
                                      MemRef<float, 4> *, MemRef<float, 4> *,
                                      MemRef<float, 3> *, MemRef<float, 3> *,
                                      MemRef<float, 3> *, MemRef<float, 4> *);
extern "C" void _mlir_ciface_forward3_decode(MemRef<float, 3> *, MemRef<float, 2> *,
                                      MemRef<float, 3> *);
extern "C" void _mlir_ciface_forward5_decode(MemRef<float, 2> *, MemRef<float, 1> *,
                                      MemRef<float, 3> *);
extern "C" void _mlir_ciface_forward169_decode(MemRef<float, 3> *, MemRef<float, 1> *,
                                        MemRef<float, 3> *);

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
  const std::string title = "DeepSeekR1  Inference Powered by Buddy Compiler";
  std::cout << "\033[33;1m" << title << "\033[0m" << std::endl;

  int split_group[] = {1, 
    1, 2, 1, 1, 2, 1, 
    1, 2, 1, 1, 2, 1, 
    1, 2, 1, 1, 2, 1, 
    1, 2, 1, 1, 2, 1, 
    1, 2, 1, 1, 2, 1, 
    1, 2, 1, 1, 2, 1, 
    1, 2, 1, 1, 2, 1, 
    1, 2, 1, 1, 2, 1, 
    1, 2, 1, 1, 2, 1, 
    1, 2, 1, 1, 2, 1, 
    1, 2, 1, 1, 2, 1, 
    1, 2, 1, 1, 2, 1, 
    1, 2, 1, 1, 2, 1, 
    1, 2, 1, 1, 2, 1, 
    1, 2, 1, 1, 2, 1, 
    1, 2, 1, 1, 2, 1, 
    1, 2, 1, 1, 2, 1, 
    1, 2, 1, 1, 2, 1, 
    1, 2, 1, 1, 2, 1, 
    1, 2, 1, 1, 2, 1, 
    1, 2, 1, 1, 2, 1, 
    1, 2, 1, 1, 2, 1, 
    1, 2, 1, 1, 2, 1, 
    1, 2, 1, 1, 2, 1, 
    1, 2, 1, 1, 2, 1, 
    1, 2, 1, 1, 2, 1, 
    1, 2, 1, 1, 2, 1, 
    1, 2, 1, 1, 2, 1, 1};
  constexpr size_t param_size_group[]= {233373760, 
    1536, 2753536, 0, 1536, 20643840, 0, 
    1536, 2753536, 0, 1536, 20643840, 0, 
    1536, 2753536, 0, 1536, 20643840, 0, 
    1536, 2753536, 0, 1536, 20643840, 0, 
    1536, 2753536, 0, 1536, 20643840, 0, 
    1536, 2753536, 0, 1536, 20643840, 0, 
    1536, 2753536, 0, 1536, 20643840, 0, 
    1536, 2753536, 0, 1536, 20643840, 0, 
    1536, 2753536, 0, 1536, 20643840, 0, 
    1536, 2753536, 0, 1536, 20643840, 0, 
    1536, 2753536, 0, 1536, 20643840, 0, 
    1536, 2753536, 0, 1536, 20643840, 0, 
    1536, 2753536, 0, 1536, 20643840, 0, 
    1536, 2753536, 0, 1536, 20643840, 0, 
    1536, 2753536, 0, 1536, 20643840, 0, 
    1536, 2753536, 0, 1536, 20643840, 0, 
    1536, 2753536, 0, 1536, 20643840, 0, 
    1536, 2753536, 0, 1536, 20643840, 0, 
    1536, 2753536, 0, 1536, 20643840, 0, 
    1536, 2753536, 0, 1536, 20643840, 0, 
    1536, 2753536, 0, 1536, 20643840, 0, 
    1536, 2753536, 0, 1536, 20643840, 0, 
    1536, 2753536, 0, 1536, 20643840, 0, 
    1536, 2753536, 0, 1536, 20643840, 0, 
    1536, 2753536, 0, 1536, 20643840, 0, 
    1536, 2753536, 0, 1536, 20643840, 0, 
    1536, 2753536, 0, 1536, 20643840, 0, 
    1536, 2753536, 0, 1536, 20643840, 0, 233375232};
  
  // int split_group_prifill[] = {1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1};
  // constexpr size_t param_size_group[] = {233373760, 1536, 2753536, 0, 1536, 20643840, 0, 1536, 2753536, 0, 1536, 20643840, 0, 1536, 2753536, 0, 1536, 20643840, 0, 1536, 2753536, 0, 1536, 20643840, 0, 1536, 2753536, 0, 1536, 20643840, 0, 1536, 2753536, 0, 1536, 20643840, 0, 1536, 2753536, 0, 1536, 20643840, 0, 1536, 2753536, 0, 1536, 20643840, 0, 1536, 2753536, 0, 1536, 20643840, 0, 1536, 2753536, 0, 1536, 20643840, 0, 1536, 2753536, 0, 1536, 20643840, 0, 1536, 2753536, 0, 1536, 20643840, 0, 1536, 2753536, 0, 1536, 20643840, 0, 1536, 2753536, 0, 1536, 20643840, 0, 1536, 2753536, 0, 1536, 20643840, 0, 1536, 2753536, 0, 1536, 20643840, 0, 1536, 2753536, 0, 1536, 20643840, 0, 1536, 2753536, 0, 1536, 20643840, 0, 1536, 2753536, 0, 1536, 20643840, 0, 1536, 2753536, 0, 1536, 20643840, 0, 1536, 2753536, 0, 1536, 20643840, 0, 1536, 2753536, 0, 1536, 20643840, 0, 1536, 2753536, 0, 1536, 20643840, 0, 1536, 2753536, 0, 1536, 20643840, 0, 1536, 2753536, 0, 1536, 20643840, 0, 1536, 2753536, 0, 1536, 20643840, 0, 1536, 2753536, 0, 1536, 20643840, 0, 1536, 2753536, 0, 1536, 20643840, 0, 233375232};

  /// Define directories of vacabulary and parameter file.
  std::string llamaDir = LLAMA_DIS_EXAMPLE_PATH;
  std::string llamaBuildDir = LLAMA_EXAMPLE_BUILD_PATH;
  const std::string vocabDir = llamaDir + "/vocab.txt";
  std::vector<std::string> paramsDirs; // 用容器存储路径

  for (int i = 0; i <= 169; i++) { // N 为需要生成的数量
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
  Text<size_t, 2> inputContainer(inputStr);
  MemRef<long long, 2> inputContainerDecode({1, 1}, 0LL);
  std::vector<MemRef<float, 1>> paramsContainers;
  MemRef<long long, 1> cachePosition({1}, 0LL);
  MemRef<float, 3> myMemRef1({1, MaxTokenLength, HiddenSize0});
  MemRef<float, 2> myMemRef2({MaxTokenLength, MaxTokenLength});
  MemRef<float, 3> myMemRef3({1, MaxTokenLength, HiddenSize});
  MemRef<float, 3> myMemRef4({1, MaxTokenLength, HiddenSize});
  MemRefContainerPrefill0 resultContainer(myMemRef1, myMemRef2, myMemRef3, myMemRef4);
  MemRefContainerPrefill0 *resultContainerPtr = &resultContainer;
  MemRef<float, 3> resultContainer0({1, MaxTokenLength, HiddenSize0});
  MemRef<float, 3> resultPrefill({1, MaxTokenLength, MaxVocabSize});
  MemRef<float, 3> resultDecode({1, 1, MaxVocabSize});
  MemRef<float, 3> subResultContainer0({1, SubMaxTokenLength, HiddenSize0});
  MemRef<float, 3> subResultContainer1({1, SubMaxTokenLength, HiddenSize0});
  MemRef<float, 3> sub3DContainer0({1, SubMaxTokenLength, HiddenSize0});
  MemRef<float, 3> sub3DContainer1({1, SubMaxTokenLength, HiddenSize0});
  MemRef<float, 3> tmp3DContainer({1, MaxTokenLength, HiddenSize0});
  
  std::vector<MemRef<float, 4>> globalKCaches;
  std::vector<MemRef<float, 4>> globalVCaches;
  for (int m = 0; m < 28 * 2; m++) { // 
  MemRef<float, 4> k({1, 1, MaxTokenLength, HiddenSize},0);
  MemRef<float, 4> v({1, 1, MaxTokenLength, HiddenSize},0);
  globalKCaches.push_back(std::move(k));
  globalVCaches.push_back(std::move(v));
}
  MemRef<float, 2> myMhaData0({MaxTokenLength, HiddenSize0});
  MemRefContainerPrefill2 kvContainer0(globalKCaches[0], globalKCaches[0], myMhaData0);
  MemRefContainerPrefill2 *kvContainerPtr0 = &kvContainer0;

  MemRef<float, 2> myMhaData1({MaxTokenLength, HiddenSize0});
  MemRefContainerPrefill2 kvContainer1(globalKCaches[1], globalKCaches[1], myMhaData1);
  MemRefContainerPrefill2 *kvContainerPtr1 = &kvContainer1;

  MemRef<float, 4> mask4DContainer({1, 1, MaxTokenLength, MaxTokenLength});
  MemRef<float, 2> tmp2DContainer0({MaxTokenLength, HiddenSize0});
  MemRef<float, 2> tmp2DContainer1({MaxTokenLength, HiddenSize0});
  MemRef<float, 2> sub2DContainer0({SubMaxTokenLength, HiddenSize0});
  MemRef<float, 2> sub2DContainer1({SubMaxTokenLength, HiddenSize0});

  // MemRefContainerPrefill2 args0(globalKCaches[0], 
  //                             globalVCaches[0], 
  //                             tmp2DContainer0);
  // MemRefContainerPrefill2 *args0Ptr = &args0; 
  // MemRefContainerPrefill2 args1(globalKCaches[1], 
  //                             globalVCaches[1], 
  //                             tmp2DContainer1); 
  // MemRefContainerPrefill2 *args1Ptr = &args1; 


  MemRef<float, 3> myData({1, 1, HiddenSize0});
  MemRef<float, 2> myTemp({1, MaxTokenLength});
  MemRef<float, 3> myCos({1, 1, HiddenSize});
  MemRef<float, 3> mySin({1, 1, HiddenSize});
  MemRefContainerPrefill0 resultContainerDecode(myData, myTemp, myCos, mySin);
  MemRefContainerPrefill0 *resultContainerDecodePtr = &resultContainerDecode;
  MemRef<float, 3> embeddingData({1, 1, HiddenSize0});
  MemRef<float, 3> decodeData({1, 1, HiddenSize0});
  MemRef<float, 2> mhaData0({1, HiddenSize0});
  MemRef<float, 2> mhaData1({1, HiddenSize0});
  MemRef<float, 2> mlpData0({1, HiddenSize0});
  MemRef<float, 2> mlpData1({1, HiddenSize0});
  MemRef<float, 4> mask4DContainerDecode({1, 1, 1, MaxTokenLength});



  /// Fill data into containers
  //  - Input: register vocabulary and tokenize the input string.
  //  - Output: register vocabulary.
  //  - Parameters: load parameters from the `arg0` file into the container.
  tokenizeInput(vocabDir, inputContainer);
  outputContainer.loadVocab(vocabDir); 

  MemRef<float, 1> paramsContainer0({param_size_group[0]});
  loadParameters(paramsDirs[0], paramsContainer0);
  int params_count = 1;
  for (int i = 1; i < 169; i++) {
    for (int j = 0; j < split_group[i]; j++) {
      if (param_size_group[i] > 0) {
        MemRef<float, 1> paramsContainer1({param_size_group[i]});
        loadParameters(paramsDirs[params_count], paramsContainer1);
        paramsContainers.push_back(paramsContainer1);
      }
      params_count++;
    }
  }
  MemRef<float, 1> paramsContainer2({param_size_group[169]});
  loadParameters(paramsDirs[params_count], paramsContainer2);

  /// Run LLaMA Inference
  //  - Perform the forward function.
  //  - Find and append the generated token.
  //  - Continue iterating until the terminal condition is met.
  
  // for (int i = 0; i < 1; i++) {
    const auto inferenceStart = std::chrono::high_resolution_clock::now();
    // Execute the forward pass of the model.
    _mlir_ciface_forward0_prefill(resultContainerPtr, &paramsContainer0,
                          &inputContainer);                  
    resultContainer0 = resultContainerPtr->data;
    auto resultContainer1 = resultContainerPtr->mask;
    std::memcpy(mask4DContainer.getData(), 
                resultContainer1.getData(), 
                resultContainer1.getSize() * sizeof(float));
    auto resultContainer2 = resultContainerPtr->sin;
    auto resultContainer3 = resultContainerPtr->cos;
    resultContainer0.splitMemRef(std::move(resultContainer0),
                                 subResultContainer0, subResultContainer1, 1,
                                 512);                        
    for (int m = 0; m < 28; m++) {
      _mlir_ciface_forward1_prefill(&sub3DContainer0, &paramsContainers[m * 6],
                            &subResultContainer0);
      _mlir_ciface_forward1_prefill(&sub3DContainer1, &paramsContainers[m * 6],
                            &subResultContainer1);
      tmp3DContainer.concatenateMemRefs(sub3DContainer0, sub3DContainer1,
                                        tmp3DContainer, 1);
      int splitIdx0 = m * 2;
      // MemRefContainerPrefill2 args0(globalKCaches[splitIdx0], 
      //                               globalVCaches[splitIdx0], 
      //                               myMhaData0);    
      kvContainer0.kcache = globalKCaches[splitIdx0];
      kvContainer0.vcache = globalVCaches[splitIdx0];                       
      _mlir_ciface_forward2_prefill(&kvContainer0, &paramsContainers[m * 6 + 1],
                            &tmp3DContainer, &resultContainer2,
                            &resultContainer3, &mask4DContainer);   
        tmp2DContainer0 = kvContainer0.data;
      int splitIdx1 = m * 2 + 1;
        // MemRefContainerPrefill2 args1(globalKCaches[splitIdx1], 
        //                             globalVCaches[splitIdx1], 
        //                             myMhaData1);
        kvContainer1.kcache = globalKCaches[splitIdx1];
        kvContainer1.vcache = globalVCaches[splitIdx1];                            
      _mlir_ciface_forward2_prefill(&kvContainer1, &paramsContainers[m * 6 + 2],
                            &tmp3DContainer, &resultContainer2,
                            &resultContainer3, &mask4DContainer);
        tmp2DContainer1 = kvContainer1.data;
      tmp2DContainer0.addMemRef(tmp2DContainer0, tmp2DContainer1);
      tmp2DContainer0.splitMemRef(std::move(tmp2DContainer0), sub2DContainer0,
                                  sub2DContainer1, 0, 512);

      _mlir_ciface_forward3_prefill(&subResultContainer0, &sub2DContainer0,
                            &subResultContainer0);
      _mlir_ciface_forward3_prefill(&subResultContainer1, &sub2DContainer1,
                            &subResultContainer1);
      _mlir_ciface_forward1_prefill(&sub3DContainer0, &paramsContainers[m * 6 + 3],
                            &subResultContainer0);
      _mlir_ciface_forward1_prefill(&sub3DContainer1, &paramsContainers[m * 6 + 3],
                            &subResultContainer1);
      tmp3DContainer.concatenateMemRefs(sub3DContainer0, sub3DContainer1,
                                        tmp3DContainer, 1);
      _mlir_ciface_forward5_prefill(&tmp2DContainer0, &paramsContainers[m * 6 + 4],
                            &tmp3DContainer);
      _mlir_ciface_forward5_prefill(&tmp2DContainer1, &paramsContainers[m * 6 + 5],
                            &tmp3DContainer);
      tmp2DContainer0.addMemRef(tmp2DContainer0, tmp2DContainer1);
      tmp2DContainer0.splitMemRef(std::move(tmp2DContainer0), sub2DContainer0,
                                  sub2DContainer1, 0, 512);
      _mlir_ciface_forward3_prefill(&subResultContainer0, &sub2DContainer0,
                            &subResultContainer0);
      _mlir_ciface_forward3_prefill(&subResultContainer1, &sub2DContainer1,
                            &subResultContainer1);
    }
    tmp3DContainer.concatenateMemRefs(subResultContainer0, subResultContainer1,
                                      tmp3DContainer, 1);
    _mlir_ciface_forward169_prefill(&resultPrefill, &paramsContainer2,
                            &tmp3DContainer);

    const auto inferenceEnd = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double, std::milli> inferenceTime =
        inferenceEnd - inferenceStart;

    // Determine the generated token.
    int tokenIndex = inputContainer.getTokenCnt() - 1;
    const float *startPtr =
        resultPrefill.getData() + tokenIndex * MaxVocabSize;
    const float *endPtr = startPtr + MaxVocabSize;
    int maxIndex = findMaxIndex(startPtr, endPtr);
    std::string tok = inputContainer.getStr(maxIndex);
    // Print the generated token and inference time.
    printIterInfo(0, tok, inferenceTime.count() / 1000);

    inputContainerDecode.getData()[0] = (long long)maxIndex;
    outputContainer.appendTokenIdx(maxIndex);
    MemRef<float, 3> logits_decode({1, 1, MaxVocabSize});
    cachePosition.getData()[0] = inputContainer.getTokenCnt();


    int generateLen = MaxTokenLength - inputContainer.getTokenCnt();
    for (int i = 1; i <= generateLen; i++) {
      const auto inferenceStart = std::chrono::high_resolution_clock::now();
      _mlir_ciface_forward0_decode(resultContainerDecodePtr, &paramsContainer0,
                          &inputContainerDecode, &cachePosition);
      embeddingData=resultContainerDecodePtr->data;
      auto embeddingMask = resultContainerDecodePtr->mask;
      std::memcpy(mask4DContainerDecode.getData(), 
                embeddingMask.getData(), 
                embeddingMask.getSize() * sizeof(float));
      auto embeddingSin = resultContainerDecodePtr->sin;
      auto embeddingCos = resultContainerDecodePtr->cos;
      for (int m = 0; m < 28; m++) {
        _mlir_ciface_forward1_decode(&decodeData, &paramsContainers[m * 6],
                            &embeddingData);
        int splitIdx0 = m * 2; 
        MemRefContainerPrefill2 res0(globalKCaches[splitIdx0], 
                               globalVCaches[splitIdx0], 
                               mhaData0); 
        // kvContainer0.kcache = globalKCaches[splitIdx0];
        // kvContainer0.vcache = globalVCaches[splitIdx0];  
        _mlir_ciface_forward2_decode( &res0, &paramsContainers[m * 6 + 1], &cachePosition,
                                      &globalKCaches[splitIdx0], &globalVCaches[splitIdx0],
                                      &decodeData, &embeddingSin,
                                      &embeddingCos, &mask4DContainerDecode);
        mhaData0 = res0.data;
        int splitIdx1 = m * 2+1; 
        // kvContainer1.kcache = globalKCaches[splitIdx1];
        // kvContainer1.vcache = globalVCaches[splitIdx1];
        MemRefContainerPrefill2 res1(globalKCaches[splitIdx1], 
                               globalVCaches[splitIdx1], 
                               mhaData1);  
        _mlir_ciface_forward2_decode( &res1, &paramsContainers[m * 6 + 2], &cachePosition,
                                      &globalKCaches[splitIdx1], &globalVCaches[splitIdx1],
                                      &decodeData, &embeddingSin,
                                      &embeddingCos, &mask4DContainerDecode);
        mhaData1 = res1.data;
        mhaData0.addMemRef(mhaData0, mhaData1);
        _mlir_ciface_forward3_decode(&embeddingData, &mhaData0,
                            &embeddingData);        
        _mlir_ciface_forward1_decode(&decodeData, &paramsContainers[m * 6 + 3],
                            &embeddingData);
        _mlir_ciface_forward5_decode(&mlpData0, &paramsContainers[m * 6 + 4],
                            &decodeData);
        _mlir_ciface_forward5_decode(&mlpData1, &paramsContainers[m * 6 + 5],
                            &decodeData);
        mlpData0.addMemRef(mlpData0, mlpData1);
        _mlir_ciface_forward3_decode(&embeddingData, &mlpData0,
                            &embeddingData);  
      }
      _mlir_ciface_forward169_decode(&resultDecode, &paramsContainer2,
                            &embeddingData);
      const auto inferenceEnd = std::chrono::high_resolution_clock::now();
      const std::chrono::duration<double, std::milli> inferenceTime =
        inferenceEnd - inferenceStart;
      
    // Determine the generated token.
    const float *startPtr = resultDecode.getData();
    const float *endPtr = startPtr + MaxVocabSize;
    maxIndex = findMaxIndex(startPtr, endPtr);
    std::string tok = inputContainer.getStr(maxIndex);
    // Print the generated token and inference time.
    printIterInfo(i, tok, inferenceTime.count() / 1000);
      
    
    // Stop if a separator token (2, </s>) or line break token (13 <0x0A>) is
    // generated.
    if (maxIndex == 151643) {
      break;
    }

    // Append the generated token into the input and output container.
    inputContainerDecode.getData()[0] = maxIndex;
    outputContainer.appendTokenIdx(maxIndex);
    cachePosition.getData()[0] += 1;

    }

  /// Print the final result
  // std::cout << "\n\033[33;1m[Total time]\033[0m " << total_time << std::endl;
  std::cout << "\033[33;1m[Input]\033[0m " << inputStr << std::endl;
  std::cout << "\033[33;1m[Output]\033[0m "
            << outputContainer.revertDeepSeekR1() << std::endl;

  return 0;
}
