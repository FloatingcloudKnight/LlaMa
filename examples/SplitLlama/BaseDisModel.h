#ifndef BASEDISMODEL_H // 作用：防止BaseDisModel.h被重复引用
#define BASEDISMODEL_H
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

using namespace buddy;
class BaseDisModel {
public:
  /// Print [Log] label in bold blue format.
  static void printLogLabel() { std::cout << "\033[34;1m[Log] \033[0m"; }

  /// Load parameters into data container.
  static void loadParameters(const std::string &paramFilePath,
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

  static void getParameters(const size_t *paramSize_group, size_t group_len, int size,
                     const std::string &splitNum,
                     std::vector<MemRef<float, 1>> &paramsContainers) {

    std::string llamaBuildDir = LLAMA_EXAMPLE_BUILD_PATH;

    for (size_t i = 0; i < group_len; i++) {
      if (paramSize_group[i] == size) {
        std::string paramsDir = llamaBuildDir + "/subgraph" +
                                std::to_string(i) + "_arg" + splitNum + ".data";
        MemRef<float, 1> paramsContainer({paramSize_group[i]});

        BaseDisModel::loadParameters(paramsDir, paramsContainer);
        paramsContainers.push_back(std::move(paramsContainer));
      }
    }
  }

  //  Tokenize input data in the container.
  static void tokenizeInput(const std::string &vocabFile,
                    Text<size_t, 2> &inputContainer, const size_t MaxTokenLength ) {
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

  // Add the inference Token to the input sequence , and translate the result of
  // that Token inputContainer: Current input container, token will be appended
  // msg: Token to be appended
  static void appendToken(Text<size_t, 2> &inputContainer, std::string &msg) {
    int maxIndex = std::stoi(msg);
    // Determine the generated token.
    int tokenIndex = inputContainer.getTokenCnt() - 1;
    std::string tok = inputContainer.getStr(maxIndex);
    // printIterInfo
    std::cout << "\033[32;1m[Iteration " << tokenIndex << "] \033[0m";
    std::cout << "Token: " << tok << std::endl;

    // Append the generated token into the input and output container.
    inputContainer.appendTokenIdx(maxIndex);
  }

  //Generate a Token based on the inference result and add the token to the output sequence
  //Return value: the index of the token generated; if the generation is finished (the terminator is encountered), -1 is returned.
  // resultContainer: container for the result of one inference of the model 
  // outputContainer: store the output sequence of tokens 
  // currentToken: the number of tokens currently generated 
  // tokenCnt: the number of tokens in the original input
  // MaxVocabSize:The maximum number of tokens allowed in the model's vocabulary.
  // separatorTokenIndex: The vocabulary index of the end-of-inference token.
  static int generatedToken( MemRef<float, 3> & resultContainer,
                             Text<size_t, 2> & outputContainer,
                             uint32_t &currentToken, uint32_t tokenCnt,
                             const size_t MaxVocabSize,
                             const size_t separatorTokenIndex) {
    int tokenIndex = currentToken + tokenCnt - 1;
    currentToken++;
    // Determine the generated token.
    const float *startPtr = resultContainer.getData() + tokenIndex * MaxVocabSize;
    const float *endPtr = startPtr + MaxVocabSize;
    // int maxIndex = findMaxIndex(startPtr, endPtr);
    int maxIndex = std::distance(startPtr, std::max_element(startPtr, endPtr));

    // Stop if a separator token or line break token (13<0x0A>) is generated.
    if (maxIndex == separatorTokenIndex)
      return -1;

    outputContainer.appendTokenIdx(maxIndex);

    return maxIndex;
  }
};

#endif // BASEDISMODEL_H
