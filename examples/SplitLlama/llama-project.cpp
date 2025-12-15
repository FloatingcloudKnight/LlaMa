#include "BaseDisModel.h"
#include <buddy/Core/Container.h>
#include <buddy/LLM/TextContainer.h>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mpi.h>
#include <stdio.h>

using namespace buddy;

constexpr size_t MaxVocabSize = 32000;
constexpr size_t MaxTokenLength = 40;
constexpr size_t SubMaxTokenLength = 10;
constexpr size_t HiddenSize = 4096;
constexpr size_t HiddenSize0 = 128;
constexpr size_t HiddenSize1 = 41;

struct MemRefContainer {
  MemRef<float, 3> memRef3D0;
  MemRef<float, 2> memRef2D;
  MemRef<float, 3> memRef3D1;
  MemRef<float, 3> memRef3D2;

  MemRefContainer(MemRef<float, 3> m1, MemRef<float, 2> m2, MemRef<float, 3> m3,
                  MemRef<float, 3> m4)
      : memRef3D0(m1), memRef2D(m2), memRef3D1(m3), memRef3D2(m4) {}
};

/// Declare LLaMA forward function.
// extern "C" void _mlir_ciface_forward(MemRef<float, 3> *, MemRef<float, 1> *,
//                                       Text<size_t, 2> *);
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
  // printLogLabel();
  // std::cout << "Vocab file: " << std::filesystem::canonical(vocabFile)
  //           << std::endl;
  // const auto buddyTokenizeStart = std::chrono::high_resolution_clock::now();
  inputContainer.tokenizeLlama(vocabFile, MaxTokenLength);
  // const auto buddyTokenizeEnd = std::chrono::high_resolution_clock::now();
  // const std::chrono::duration<double, std::milli> buddyTokenizeTime =
  //     buddyTokenizeEnd - buddyTokenizeStart;
  // printLogLabel();
  // std::cout << "Tokenize time: " << buddyTokenizeTime.count() << "ms"
  //           << std::endl;
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
  // printLogLabel();
  // std::cout << "Params file: " << std::filesystem::canonical(paramFilePath)
  //           << std::endl;
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

int main(int argc, char *argv[]) {

  /// Define directories of vacabulary and parameter file.
  std::string llamaDir = LLAMA_SPLIT_EXAMPLE_PATH;
  std::string llamaBuildDir = LLAMA_EXAMPLE_BUILD_PATH;
  const std::string vocabDir = llamaDir + "/vocab.txt";

  // Common variables needed by all ranks

  int subSize = SubMaxTokenLength * HiddenSize;
  int offset0 = subSize;
  int offset1 = subSize * 2;
  int offset2 = subSize * 3;

  int rank, size;
  int generateLen = MaxTokenLength;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (rank == 0) {
    const std::string title = "LLaMA 2 Inference Powered by Buddy Compiler";
    std::cout << "\033[33;1m" << title << "\033[0m" << std::endl;

    // Rank 0 specific variables
    Text<size_t, 2> outputContainer;
    outputContainer.loadVocab(vocabDir);

    MemRef<float, 3> myMemRef1({1, MaxTokenLength, HiddenSize});
    MemRef<float, 2> myMemRef2({MaxTokenLength, HiddenSize1});
    MemRef<float, 3> myMemRef3({1, MaxTokenLength, HiddenSize0});
    MemRef<float, 3> myMemRef4({1, MaxTokenLength, HiddenSize0});
    MemRefContainer tmpContainer(myMemRef1, myMemRef2, myMemRef3, myMemRef4);
    MemRefContainer *inputResultContainerPtr = &tmpContainer;
    MemRef<float, 3> tmp3DMemRef({1, MaxTokenLength, HiddenSize});
    constexpr size_t param_size0 = 131072064;
    const std::string paramsDir0 = llamaBuildDir + "/subgraph0_arg0.data";
    constexpr size_t param_size1 = 131076096;
    const std::string paramsDir1 = llamaBuildDir + "/subgraph193_arg0.data";

    MemRef<float, 1> paramsContainer0({param_size0});
    loadParameters(paramsDir0, paramsContainer0);

    MemRef<float, 1> paramsContainer1({param_size1});
    loadParameters(paramsDir1, paramsContainer1);
    /// Get user message.
    std::string inputStr;
    getUserInput(inputStr);
    Text<size_t, 2> inputContainer(inputStr);
    tokenizeInput(vocabDir, inputContainer);
    generateLen = MaxTokenLength - inputContainer.getTokenCnt();

    float *inputPtr = nullptr;
    float *outputPtr = tmp3DMemRef.getData();
    MPI_Request send_req[4], recv_req[4];
    MPI_Request send_req_mha[12];
    for (int i = 0; i < generateLen; i++) {
      const auto inferenceStart = std::chrono::high_resolution_clock::now();
      _mlir_ciface_forward0(inputResultContainerPtr, &paramsContainer0,
                            &inputContainer);
      inputPtr = inputResultContainerPtr->memRef3D0.getData();

        MPI_Isend(inputPtr, subSize, MPI_FLOAT, 1, 0, MPI_COMM_WORLD,
                  &send_req[0]);
        MPI_Isend(inputPtr + offset0, subSize, MPI_FLOAT, 2, 0, MPI_COMM_WORLD,
                  &send_req[1]);
        MPI_Isend(inputPtr + offset1, subSize, MPI_FLOAT, 3, 0, MPI_COMM_WORLD,
                  &send_req[2]);
        MPI_Isend(inputPtr + offset2, subSize, MPI_FLOAT, 4, 0, MPI_COMM_WORLD,
                  &send_req[3]);
        MPI_Waitall(4, send_req, MPI_STATUSES_IGNORE);
        for(int m = 1; m < 20; m+=4){
          MPI_Isend(inputResultContainerPtr->memRef2D.getData(),
                  MaxTokenLength * HiddenSize1, MPI_FLOAT, m, 1, MPI_COMM_WORLD,
                  &send_req_mha[0]);
          MPI_Isend(inputResultContainerPtr->memRef3D1.getData(),
                    MaxTokenLength * HiddenSize0, MPI_FLOAT, m, 2, MPI_COMM_WORLD,
                    &send_req_mha[1]);
          MPI_Isend(inputResultContainerPtr->memRef3D2.getData(),
                    MaxTokenLength * HiddenSize0, MPI_FLOAT, m, 3, MPI_COMM_WORLD,
                    &send_req_mha[2]);
          MPI_Isend(inputResultContainerPtr->memRef2D.getData(),
                    MaxTokenLength * HiddenSize1, MPI_FLOAT, m+1, 1, MPI_COMM_WORLD,
                    &send_req_mha[3]);
          MPI_Isend(inputResultContainerPtr->memRef3D1.getData(),
                    MaxTokenLength * HiddenSize0, MPI_FLOAT, m+1, 2, MPI_COMM_WORLD,
                    &send_req_mha[4]);
          MPI_Isend(inputResultContainerPtr->memRef3D2.getData(),
                    MaxTokenLength * HiddenSize0, MPI_FLOAT, m+1, 3, MPI_COMM_WORLD,
                    &send_req_mha[5]);
          MPI_Isend(inputResultContainerPtr->memRef2D.getData(),
                    MaxTokenLength * HiddenSize1, MPI_FLOAT, m+2, 1, MPI_COMM_WORLD,
                    &send_req_mha[6]);
          MPI_Isend(inputResultContainerPtr->memRef3D1.getData(),
                    MaxTokenLength * HiddenSize0, MPI_FLOAT, m+2, 2, MPI_COMM_WORLD,
                    &send_req_mha[7]);
          MPI_Isend(inputResultContainerPtr->memRef3D2.getData(),
                    MaxTokenLength * HiddenSize0, MPI_FLOAT, m+2, 3, MPI_COMM_WORLD,
                    &send_req_mha[8]);
          MPI_Isend(inputResultContainerPtr->memRef2D.getData(),
                    MaxTokenLength * HiddenSize1, MPI_FLOAT, m+3, 1, MPI_COMM_WORLD,
                    &send_req_mha[9]);
          MPI_Isend(inputResultContainerPtr->memRef3D1.getData(),
                    MaxTokenLength * HiddenSize0, MPI_FLOAT, m+3, 2, MPI_COMM_WORLD,
                    &send_req_mha[10]);
          MPI_Isend(inputResultContainerPtr->memRef3D2.getData(),
                    MaxTokenLength * HiddenSize0, MPI_FLOAT, m+3, 3, MPI_COMM_WORLD,
                    &send_req_mha[11]);
          MPI_Waitall(12, send_req_mha, MPI_STATUSES_IGNORE);
        }

      MPI_Irecv(outputPtr, subSize, MPI_FLOAT, 17, 0, MPI_COMM_WORLD,
                &recv_req[0]);
      MPI_Irecv(outputPtr + offset0, subSize, MPI_FLOAT, 18, 0, MPI_COMM_WORLD,
                &recv_req[1]);
      MPI_Irecv(outputPtr + offset1, subSize, MPI_FLOAT, 19, 0, MPI_COMM_WORLD,
                &recv_req[2]);
      MPI_Irecv(outputPtr + offset2, subSize, MPI_FLOAT, 20, 0, MPI_COMM_WORLD,
                &recv_req[3]);
      MPI_Waitall(4, recv_req, MPI_STATUSES_IGNORE);
      _mlir_ciface_forward193(&inputResultContainerPtr->memRef3D0,
                              &paramsContainer1, &tmp3DMemRef);
      const auto inferenceEnd = std::chrono::high_resolution_clock::now();
      const std::chrono::duration<double, std::milli> inferenceTime =
          inferenceEnd - inferenceStart;
      inputPtr = inputResultContainerPtr->memRef3D0.getData();
      // Determine the generated token.
      int tokenIndex = inputContainer.getTokenCnt() - 1;
      const float *startPtr = inputPtr + tokenIndex * MaxVocabSize;
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
    /// Print the final result
    std::cout << "\n\033[33;1m[Input]\033[0m " << inputStr << std::endl;
    std::cout << "\033[33;1m[Output]\033[0m " << outputContainer.revertLlama()
              << std::endl;

    return 0;
  } else if (rank == 1 || rank == 5 || rank == 9 || rank == 13 || rank == 17) {
    // === RMSNorm ===
    // Rank 1 specific variables
    MemRef<float, 3> subResultContainer({1, SubMaxTokenLength, HiddenSize});
    MemRef<float, 3> sub3DContainer({1, SubMaxTokenLength, HiddenSize});
    MemRef<float, 2> mhaMemRef2D({MaxTokenLength, HiddenSize1});
    MemRef<float, 3> mhaMemRef3D1({1, MaxTokenLength, HiddenSize0});
    MemRef<float, 3> mhaMemRef3D2({1, MaxTokenLength, HiddenSize0});
    MemRef<float, 3> tmp3DMemRef({1, MaxTokenLength, HiddenSize});
    MemRef<float, 2> tmp2DContainer({MaxTokenLength, HiddenSize});
    MemRef<float, 2> sub2DContainer({SubMaxTokenLength, HiddenSize});
    MemRef<float, 2> sub2DContainer0({SubMaxTokenLength, HiddenSize});
    MemRef<float, 2> sub2DContainer1({SubMaxTokenLength, HiddenSize});
    

    // Get pointers once to avoid repeated calls
    float *subResultPtr = subResultContainer.getData();
    float *rmsPtr = sub3DContainer.getData();
    float *mhaMemRef2DPtr = mhaMemRef2D.getData();
    float *mhaMemRef3D1Ptr = mhaMemRef3D1.getData();
    float *mhaMemRef3D2Ptr = mhaMemRef3D2.getData();
    float *mhaPtr = tmp3DMemRef.getData();
    float *mhaOutputPtr = tmp2DContainer.getData();
    float *sub2DPtr = sub2DContainer.getData();
    float *sub2DPtr0 = sub2DContainer0.getData();
    float *sub2DPtr1 = sub2DContainer1.getData();

    constexpr size_t paramSizeRMS = 4096;
    constexpr size_t paramSizeMHA = 16777216;
    constexpr size_t paramSizeMLP = 33816576;
    int id = rank / 4;
    int init = id * 42;
    int endVal = (id + 1) * 42;
    int times = 7;
    int source = 0;
    int dest2 = rank + 1;
    int dest3 = rank + 2;
    int dest4 = rank + 3;
    int nextRank = rank + 4;
    if (rank == 17){
      endVal = 193;
      times = 4;
      nextRank = 0;
    }
    MPI_Request send_req[4], recv_req[4];
    MPI_Request mha_recv_req[3];
    bool mha_params_received = false;

    std::vector<std::string> paramsDirsRMS, paramsDirsRMS0;
    std::vector<std::string> paramsDirsMHA, paramsDirsMLP;
    std::vector<MemRef<float, 1>> paramsContainersRMS, paramsContainersRMS0;
    std::vector<MemRef<float, 1>> paramsContainersMHA, paramsContainersMLP;
    // Get Parameter file paths
    for (int i = init+1; i < endVal; i += 6) {
      paramsDirsRMS.emplace_back(llamaBuildDir + "/subgraph" +
                                 std::to_string(i) + "_arg0" + ".data");
      paramsDirsMHA.emplace_back(llamaBuildDir + "/subgraph" +
                                 std::to_string(i + 1) + "_arg0" + ".data");
      paramsDirsRMS0.emplace_back(llamaBuildDir + "/subgraph" +
                                  std::to_string(i + 3) + "_arg0" + ".data");
      paramsDirsMLP.emplace_back(llamaBuildDir + "/subgraph" +
                                 std::to_string(i + 4) + "_arg0" + ".data");
    }

    // Load parameters after Bcast to avoid blocking rank 2 at MPI_Barrier
    for (int i = 0; i < times; i++) {
      // First RMS
      MemRef<float, 1> paramsContainerRMS({paramSizeRMS});
      loadParameters(paramsDirsRMS[i], paramsContainerRMS);
      paramsContainersRMS.push_back(paramsContainerRMS);
      // MHA
      MemRef<float, 1> paramsContainerMHA({paramSizeMHA});
      loadParameters(paramsDirsMHA[i], paramsContainerMHA);
      paramsContainersMHA.push_back(paramsContainerMHA);
      // Second RMS
      MemRef<float, 1> paramsContainerRMS0({paramSizeRMS});
      loadParameters(paramsDirsRMS0[i], paramsContainerRMS0);
      paramsContainersRMS0.push_back(paramsContainerRMS0);
      // MLP
      MemRef<float, 1> paramsContainerMLP({paramSizeMLP});
      loadParameters(paramsDirsMLP[i], paramsContainerMLP);
      paramsContainersMLP.push_back(paramsContainerMLP);
    }

    for (int i = 0; i < generateLen; i++) {
      /* post MHA params receive once to overlap communication with work */
      MPI_Irecv(mhaMemRef2DPtr, MaxTokenLength * HiddenSize1, MPI_FLOAT,
                source, 1, MPI_COMM_WORLD, &mha_recv_req[0]);
      MPI_Irecv(mhaMemRef3D1Ptr, MaxTokenLength * HiddenSize0, MPI_FLOAT,
                source, 2, MPI_COMM_WORLD, &mha_recv_req[1]);
      MPI_Irecv(mhaMemRef3D2Ptr, MaxTokenLength * HiddenSize0, MPI_FLOAT,
                source, 3, MPI_COMM_WORLD, &mha_recv_req[2]);
      for (int m = 0; m < times; m++) {
        if (m == 0) {
          if (rank == 1) {
            MPI_Irecv(subResultPtr, subSize, MPI_FLOAT, source, 0,
                      MPI_COMM_WORLD, &recv_req[0]);
            MPI_Wait(&recv_req[0], MPI_STATUS_IGNORE);
          } else {
            MPI_Irecv(subResultPtr, subSize, MPI_FLOAT, rank - 4, 0,
                      MPI_COMM_WORLD, &recv_req[0]);
            MPI_Wait(&recv_req[0], MPI_STATUS_IGNORE);
          }
        }

        // const auto inferenceStart =
        // std::chrono::high_resolution_clock::now();
        // ----- RMS -----
        _mlir_ciface_forward1(&sub3DContainer, &paramsContainersRMS[m],
                              &subResultContainer);
        rmsPtr = sub3DContainer.getData();
        mhaPtr = tmp3DMemRef.getData();
        // const auto inferenceEnd = std::chrono::high_resolution_clock::now();
        // ----- AllGather -----
        MPI_Isend(rmsPtr, subSize, MPI_FLOAT, dest2, 0, MPI_COMM_WORLD,
                  &send_req[0]);
        MPI_Irecv(mhaPtr + offset0, subSize, MPI_FLOAT, dest2, 0,
                  MPI_COMM_WORLD, &recv_req[0]);
        for (int idx = 0; idx < subSize; idx++) {
          mhaPtr[idx] = rmsPtr[idx];
        }
        MPI_Wait(&send_req[0], MPI_STATUS_IGNORE);
        MPI_Wait(&recv_req[0], MPI_STATUS_IGNORE);

        MPI_Isend(mhaPtr, offset1, MPI_FLOAT, dest3, 0, MPI_COMM_WORLD,
                  &send_req[0]);
        MPI_Irecv(mhaPtr + offset1, offset1, MPI_FLOAT, dest3, 0,
                  MPI_COMM_WORLD, &recv_req[0]);
        MPI_Wait(&send_req[0], MPI_STATUS_IGNORE);
        MPI_Wait(&recv_req[0], MPI_STATUS_IGNORE);

        // const std::chrono::duration<double, std::milli> inferenceTime =
        //     inferenceEnd - inferenceStart;
        // std::cout << "[Log] RMSNorm: " << inferenceTime.count() / 1000 << "s"
        // << std::endl;
        // ----- MHA -----
        if(m == 0){
          MPI_Waitall(3, mha_recv_req, MPI_STATUSES_IGNORE);
        }
        _mlir_ciface_forward2(&tmp2DContainer, &paramsContainersMHA[m],
                              &tmp3DMemRef, &mhaMemRef3D1, &mhaMemRef3D2,
                              &mhaMemRef2D);
        // ----- Reduce-Scatter -----
        mhaOutputPtr = tmp2DContainer.getData();
        MPI_Isend(mhaOutputPtr + offset0, subSize, MPI_FLOAT, dest2, 0,
                  MPI_COMM_WORLD, &send_req[0]);
        MPI_Irecv(sub2DPtr, subSize, MPI_FLOAT, dest4, 0, MPI_COMM_WORLD,
                  &recv_req[0]);
        MPI_Wait(&send_req[0], MPI_STATUS_IGNORE);
        MPI_Wait(&recv_req[0], MPI_STATUS_IGNORE);

        MPI_Isend(mhaOutputPtr + offset1, subSize, MPI_FLOAT, dest3, 0,
                  MPI_COMM_WORLD, &send_req[0]);
        MPI_Irecv(sub2DPtr0, subSize, MPI_FLOAT, dest3, 0, MPI_COMM_WORLD,
                  &recv_req[0]);
        for (int idx = 0; idx < subSize; idx++) {
          sub2DPtr[idx] += mhaOutputPtr[idx];
        }
        MPI_Wait(&send_req[0], MPI_STATUS_IGNORE);
        MPI_Wait(&recv_req[0], MPI_STATUS_IGNORE);

        MPI_Isend(mhaOutputPtr + offset2, subSize, MPI_FLOAT, dest4, 0,
                  MPI_COMM_WORLD, &send_req[0]);
        MPI_Irecv(sub2DPtr1, subSize, MPI_FLOAT, dest2, 0, MPI_COMM_WORLD,
                  &recv_req[0]);
        for (int idx = 0; idx < subSize; idx++) {
          sub2DPtr[idx] += sub2DPtr0[idx];
        }
        MPI_Wait(&send_req[0], MPI_STATUS_IGNORE);
        MPI_Wait(&recv_req[0], MPI_STATUS_IGNORE);
        for (int idx = 0; idx < subSize; idx++) {
          sub2DPtr[idx] += sub2DPtr1[idx];
        }
        // ----- RMS & Add -----
        _mlir_ciface_forward3(&subResultContainer, &sub2DContainer,
                              &subResultContainer);
        _mlir_ciface_forward1(&sub3DContainer, &paramsContainersRMS0[m],
                              &subResultContainer);
        rmsPtr = sub3DContainer.getData();
        mhaPtr = tmp3DMemRef.getData();
        // const auto inferenceEnd = std::chrono::high_resolution_clock::now();
        // ----- AllGather -----
        MPI_Isend(rmsPtr, subSize, MPI_FLOAT, dest2, 0, MPI_COMM_WORLD,
                  &send_req[0]);
        MPI_Irecv(mhaPtr + offset0, subSize, MPI_FLOAT, dest2, 0,
                  MPI_COMM_WORLD, &recv_req[0]);
        for (int idx = 0; idx < subSize; idx++) {
          mhaPtr[idx] = rmsPtr[idx];
        }
        MPI_Wait(&send_req[0], MPI_STATUS_IGNORE);
        MPI_Wait(&recv_req[0], MPI_STATUS_IGNORE);

        MPI_Isend(mhaPtr, offset1, MPI_FLOAT, dest3, 0, MPI_COMM_WORLD,
                  &send_req[0]);
        MPI_Irecv(mhaPtr + offset1, offset1, MPI_FLOAT, dest3, 0,
                  MPI_COMM_WORLD, &recv_req[0]);
        MPI_Wait(&send_req[0], MPI_STATUS_IGNORE);
        MPI_Wait(&recv_req[0], MPI_STATUS_IGNORE);
        // ----- MLP -----
        _mlir_ciface_forward5(&tmp2DContainer, &paramsContainersMLP[m],
                              &tmp3DMemRef);
        // ----- Reduce-Scatter -----
        mhaOutputPtr = tmp2DContainer.getData();
        MPI_Isend(mhaOutputPtr + offset0, subSize, MPI_FLOAT, dest2, 0,
                  MPI_COMM_WORLD, &send_req[0]);
        MPI_Irecv(sub2DPtr, subSize, MPI_FLOAT, dest4, 0, MPI_COMM_WORLD,
                  &recv_req[0]);
        MPI_Wait(&send_req[0], MPI_STATUS_IGNORE);
        MPI_Wait(&recv_req[0], MPI_STATUS_IGNORE);

        MPI_Isend(mhaOutputPtr + offset1, subSize, MPI_FLOAT, dest3, 0,
                  MPI_COMM_WORLD, &send_req[0]);
        MPI_Irecv(sub2DPtr0, subSize, MPI_FLOAT, dest3, 0, MPI_COMM_WORLD,
                  &recv_req[0]);
        for (int idx = 0; idx < subSize; idx++) {
          sub2DPtr[idx] += mhaOutputPtr[idx];
        }
        MPI_Wait(&send_req[0], MPI_STATUS_IGNORE);
        MPI_Wait(&recv_req[0], MPI_STATUS_IGNORE);

        MPI_Isend(mhaOutputPtr + offset2, subSize, MPI_FLOAT, dest4, 0,
                  MPI_COMM_WORLD, &send_req[0]);
        MPI_Irecv(sub2DPtr1, subSize, MPI_FLOAT, dest2, 0, MPI_COMM_WORLD,
                  &recv_req[0]);
        for (int idx = 0; idx < subSize; idx++) {
          sub2DPtr[idx] += sub2DPtr0[idx];
        }
        MPI_Wait(&send_req[0], MPI_STATUS_IGNORE);
        MPI_Wait(&recv_req[0], MPI_STATUS_IGNORE);
        for (int idx = 0; idx < subSize; idx++) {
          sub2DPtr[idx] += sub2DPtr1[idx];
        }
        // ----- Add -----
        _mlir_ciface_forward3(&subResultContainer, &sub2DContainer,
                              &subResultContainer);                 
        if (m == (times - 1)) {
          subResultPtr = subResultContainer.getData();
          MPI_Send(subResultPtr, subSize, MPI_FLOAT, nextRank, 0,
                   MPI_COMM_WORLD);
        }
      }
    }
  } else if (rank == 2 || rank == 6 || rank == 10 || rank == 14 || rank == 18) {
    // === RMSNorm ===
    // Rank 2 specific variables
    MemRef<float, 3> subResultContainer({1, SubMaxTokenLength, HiddenSize});
    MemRef<float, 3> sub3DContainer({1, SubMaxTokenLength, HiddenSize});
    MemRef<float, 2> mhaMemRef2D({MaxTokenLength, HiddenSize1});
    MemRef<float, 3> mhaMemRef3D1({1, MaxTokenLength, HiddenSize0});
    MemRef<float, 3> mhaMemRef3D2({1, MaxTokenLength, HiddenSize0});
    MemRef<float, 3> tmp3DMemRef({1, MaxTokenLength, HiddenSize});
    MemRef<float, 2> tmp2DContainer({MaxTokenLength, HiddenSize});
    MemRef<float, 2> sub2DContainer({SubMaxTokenLength, HiddenSize});
    MemRef<float, 2> sub2DContainer0({SubMaxTokenLength, HiddenSize});
    MemRef<float, 2> sub2DContainer1({SubMaxTokenLength, HiddenSize});

    // Get pointers once to avoid repeated calls
    float *subResultPtr = subResultContainer.getData();
    float *rmsPtr = sub3DContainer.getData();
    float *mhaMemRef2DPtr = mhaMemRef2D.getData();
    float *mhaMemRef3D1Ptr = mhaMemRef3D1.getData();
    float *mhaMemRef3D2Ptr = mhaMemRef3D2.getData();
    float *mhaPtr = tmp3DMemRef.getData();
    float *mhaOutputPtr = tmp2DContainer.getData();
    float *sub2DPtr = sub2DContainer.getData();
    float *sub2DPtr0 = sub2DContainer0.getData();
    float *sub2DPtr1 = sub2DContainer1.getData();

    constexpr size_t paramSizeRMS = 4096;
    constexpr size_t paramSizeMHA = 16777216;
    constexpr size_t paramSizeMLP = 33816576;
    int id = rank / 4;
    int init = id * 42;
    int endVal = (id + 1) * 42;
    int times = 7;
    int source = 0;
    int dest1 = rank - 1;
    int dest3 = rank + 1;
    int dest4 = rank + 2;
    int nextRank = rank + 4;
    if (rank == 18){
      endVal = 193;
      times = 4;
      nextRank = 0;
    }
    MPI_Request send_req[4], recv_req[4];
    MPI_Request mha_recv_req[3];
    bool mha_params_received = false;
      

    std::vector<std::string> paramsDirsRMS, paramsDirsRMS0;
    std::vector<std::string> paramsDirsMHA, paramsDirsMLP;
    std::vector<MemRef<float, 1>> paramsContainersRMS, paramsContainersRMS0;
    std::vector<MemRef<float, 1>> paramsContainersMHA, paramsContainersMLP;
    // Get Parameter file paths
    for (int i = init+1; i < endVal; i += 6) {
      paramsDirsRMS.emplace_back(llamaBuildDir + "/subgraph" +
                                 std::to_string(i) + "_arg0" + ".data");
      paramsDirsRMS0.emplace_back(llamaBuildDir + "/subgraph" +
                                  std::to_string(i + 3) + "_arg0" + ".data");
      paramsDirsMHA.emplace_back(llamaBuildDir + "/subgraph" +
                                 std::to_string(i + 1) + "_arg1" + ".data");
      paramsDirsMLP.emplace_back(llamaBuildDir + "/subgraph" +
                                 std::to_string(i + 4) + "_arg1" + ".data");
    }

    // Load parameters after Bcast to avoid blocking rank 2 at MPI_Barrier
    for (int i = 0; i < times; i++) {
      // First RMS
      MemRef<float, 1> paramsContainerRMS({paramSizeRMS});
      loadParameters(paramsDirsRMS[i], paramsContainerRMS);
      paramsContainersRMS.push_back(paramsContainerRMS);
      // MHA
      MemRef<float, 1> paramsContainerMHA({paramSizeMHA});
      loadParameters(paramsDirsMHA[i], paramsContainerMHA);
      paramsContainersMHA.push_back(paramsContainerMHA);
      // Second RMS
      MemRef<float, 1> paramsContainerRMS0({paramSizeRMS});
      loadParameters(paramsDirsRMS0[i], paramsContainerRMS0);
      paramsContainersRMS0.push_back(paramsContainerRMS0);
      // MLP
      MemRef<float, 1> paramsContainerMLP({paramSizeMLP});
      loadParameters(paramsDirsMLP[i], paramsContainerMLP);
      paramsContainersMLP.push_back(paramsContainerMLP);
    }

    for (int i = 0; i < generateLen; i++) {
      MPI_Irecv(mhaMemRef2DPtr, MaxTokenLength * HiddenSize1, MPI_FLOAT,
                source, 1, MPI_COMM_WORLD, &mha_recv_req[0]);
      MPI_Irecv(mhaMemRef3D1Ptr, MaxTokenLength * HiddenSize0, MPI_FLOAT,
                source, 2, MPI_COMM_WORLD, &mha_recv_req[1]);
      MPI_Irecv(mhaMemRef3D2Ptr, MaxTokenLength * HiddenSize0, MPI_FLOAT,
                source, 3, MPI_COMM_WORLD, &mha_recv_req[2]);
      for (int m = 0; m < times; m++) {
        if (m == 0) {
          if(rank == 2){
            MPI_Irecv(subResultPtr, subSize, MPI_FLOAT, source, 0, MPI_COMM_WORLD,
                    &recv_req[0]);
          MPI_Wait(&recv_req[0], MPI_STATUS_IGNORE);
          } else {
            MPI_Irecv(subResultPtr, subSize, MPI_FLOAT, rank - 4, 0,
                      MPI_COMM_WORLD, &recv_req[0]);
            MPI_Wait(&recv_req[0], MPI_STATUS_IGNORE);
          }
        }

        // ----- RMS -----
        _mlir_ciface_forward1(&sub3DContainer, &paramsContainersRMS[m],
                              &subResultContainer);
        rmsPtr = sub3DContainer.getData();
        mhaPtr = tmp3DMemRef.getData();
        // ----- AllGather -----
        MPI_Isend(rmsPtr, subSize, MPI_FLOAT, dest1, 0, MPI_COMM_WORLD,
                  &send_req[0]);
        MPI_Irecv(mhaPtr, subSize, MPI_FLOAT, dest1, 0, MPI_COMM_WORLD,
                  &recv_req[0]);
        for (int idx = 0; idx < subSize; idx++) {
          mhaPtr[idx + offset0] = rmsPtr[idx];
        }
        MPI_Wait(&send_req[0], MPI_STATUS_IGNORE);
        MPI_Wait(&recv_req[0], MPI_STATUS_IGNORE);

        MPI_Isend(mhaPtr, offset1, MPI_FLOAT, dest4, 0, MPI_COMM_WORLD,
                  &send_req[0]);
        MPI_Irecv(mhaPtr + offset1, offset1, MPI_FLOAT, dest4, 0,
                  MPI_COMM_WORLD, &recv_req[0]);
        MPI_Wait(&send_req[0], MPI_STATUS_IGNORE);
        MPI_Wait(&recv_req[0], MPI_STATUS_IGNORE);

        // ----- MHA -----
        if(m == 0){
          MPI_Waitall(3, mha_recv_req, MPI_STATUSES_IGNORE);
        }
        _mlir_ciface_forward2(&tmp2DContainer, &paramsContainersMHA[m],
                              &tmp3DMemRef, &mhaMemRef3D1, &mhaMemRef3D2,
                              &mhaMemRef2D);
        // ----- Reduce-Scatter -----
        mhaOutputPtr = tmp2DContainer.getData();
        MPI_Isend(mhaOutputPtr + offset1, subSize, MPI_FLOAT, dest3, 0,
                  MPI_COMM_WORLD, &send_req[0]);
        MPI_Irecv(sub2DPtr, subSize, MPI_FLOAT, dest1, 0, MPI_COMM_WORLD,
                  &recv_req[0]);
        MPI_Wait(&send_req[0], MPI_STATUS_IGNORE);
        MPI_Wait(&recv_req[0], MPI_STATUS_IGNORE);

        MPI_Isend(mhaOutputPtr + offset2, subSize, MPI_FLOAT, dest4, 0,
                  MPI_COMM_WORLD, &send_req[0]);
        MPI_Irecv(sub2DPtr0, subSize, MPI_FLOAT, dest4, 0, MPI_COMM_WORLD,
                  &recv_req[0]);
        for (int idx = 0; idx < subSize; idx++) {
          sub2DPtr[idx] += mhaOutputPtr[idx + offset0];
        }
        MPI_Wait(&send_req[0], MPI_STATUS_IGNORE);
        MPI_Wait(&recv_req[0], MPI_STATUS_IGNORE);

        MPI_Isend(mhaOutputPtr, subSize, MPI_FLOAT, dest1, 0, MPI_COMM_WORLD,
                  &send_req[0]);
        MPI_Irecv(sub2DPtr1, subSize, MPI_FLOAT, dest3, 0, MPI_COMM_WORLD,
                  &recv_req[0]);
        for (int idx = 0; idx < subSize; idx++) {
          sub2DPtr[idx] += sub2DPtr0[idx];
        }
        MPI_Wait(&send_req[0], MPI_STATUS_IGNORE);
        MPI_Wait(&recv_req[0], MPI_STATUS_IGNORE);
        for (int idx = 0; idx < subSize; idx++) {
          sub2DPtr[idx] += sub2DPtr1[idx];
        }

        // ----- RMS & Add -----
        _mlir_ciface_forward3(&subResultContainer, &sub2DContainer,
                              &subResultContainer);
        _mlir_ciface_forward1(&sub3DContainer, &paramsContainersRMS0[m],
                              &subResultContainer);
        rmsPtr = sub3DContainer.getData();
        mhaPtr = tmp3DMemRef.getData();
        // const auto inferenceEnd = std::chrono::high_resolution_clock::now();
        // ----- AllGather -----
        MPI_Isend(rmsPtr, subSize, MPI_FLOAT, dest1, 0, MPI_COMM_WORLD,
                  &send_req[0]);
        MPI_Irecv(mhaPtr, subSize, MPI_FLOAT, dest1, 0, MPI_COMM_WORLD,
                  &recv_req[0]);
        for (int idx = 0; idx < subSize; idx++) {
          mhaPtr[idx + offset0] = rmsPtr[idx];
        }
        MPI_Wait(&send_req[0], MPI_STATUS_IGNORE);
        MPI_Wait(&recv_req[0], MPI_STATUS_IGNORE);

        MPI_Isend(mhaPtr, offset1, MPI_FLOAT, dest4, 0, MPI_COMM_WORLD,
                  &send_req[0]);
        MPI_Irecv(mhaPtr + offset1, offset1, MPI_FLOAT, dest4, 0,
                  MPI_COMM_WORLD, &recv_req[0]);
        MPI_Wait(&send_req[0], MPI_STATUS_IGNORE);
        MPI_Wait(&recv_req[0], MPI_STATUS_IGNORE);
        // ----- MLP -----
        _mlir_ciface_forward5(&tmp2DContainer, &paramsContainersMLP[m],
                              &tmp3DMemRef);
        // ----- Reduce-Scatter -----
        mhaOutputPtr = tmp2DContainer.getData();
        MPI_Isend(mhaOutputPtr + offset1, subSize, MPI_FLOAT, dest3, 0,
                  MPI_COMM_WORLD, &send_req[0]);
        MPI_Irecv(sub2DPtr, subSize, MPI_FLOAT, dest1, 0, MPI_COMM_WORLD,
                  &recv_req[0]);
        MPI_Wait(&send_req[0], MPI_STATUS_IGNORE);
        MPI_Wait(&recv_req[0], MPI_STATUS_IGNORE);

        MPI_Isend(mhaOutputPtr + offset2, subSize, MPI_FLOAT, dest4, 0,
                  MPI_COMM_WORLD, &send_req[0]);
        MPI_Irecv(sub2DPtr0, subSize, MPI_FLOAT, dest4, 0, MPI_COMM_WORLD,
                  &recv_req[0]);
        for (int idx = 0; idx < subSize; idx++) {
          sub2DPtr[idx] += mhaOutputPtr[idx + offset0];
        }
        MPI_Wait(&send_req[0], MPI_STATUS_IGNORE);
        MPI_Wait(&recv_req[0], MPI_STATUS_IGNORE);

        MPI_Isend(mhaOutputPtr, subSize, MPI_FLOAT, dest1, 0, MPI_COMM_WORLD,
                  &send_req[0]);
        MPI_Irecv(sub2DPtr1, subSize, MPI_FLOAT, dest3, 0, MPI_COMM_WORLD,
                  &recv_req[0]);
        for (int idx = 0; idx < subSize; idx++) {
          sub2DPtr[idx] += sub2DPtr0[idx];
        }
        MPI_Wait(&send_req[0], MPI_STATUS_IGNORE);
        MPI_Wait(&recv_req[0], MPI_STATUS_IGNORE);
        for (int idx = 0; idx < subSize; idx++) {
          sub2DPtr[idx] += sub2DPtr1[idx];
        }
        // ----- Add -----
        _mlir_ciface_forward3(&subResultContainer, &sub2DContainer,
                              &subResultContainer);
        if (m == (times - 1)) {
          subResultPtr = subResultContainer.getData();
          MPI_Send(subResultPtr, subSize, MPI_FLOAT, nextRank, 0, MPI_COMM_WORLD);
        }
      }
    }
  } else if (rank == 3 || rank == 7 || rank == 11 || rank == 15 ||
             rank == 19) {
    // === RMSNorm ===
    // Rank 3 specific variables
    MemRef<float, 3> subResultContainer({1, SubMaxTokenLength, HiddenSize});
    MemRef<float, 3> sub3DContainer({1, SubMaxTokenLength, HiddenSize});
    MemRef<float, 2> mhaMemRef2D({MaxTokenLength, HiddenSize1});
    MemRef<float, 3> mhaMemRef3D1({1, MaxTokenLength, HiddenSize0});
    MemRef<float, 3> mhaMemRef3D2({1, MaxTokenLength, HiddenSize0});
    MemRef<float, 3> tmp3DMemRef({1, MaxTokenLength, HiddenSize});
    MemRef<float, 2> tmp2DContainer({MaxTokenLength, HiddenSize});
    MemRef<float, 2> sub2DContainer({SubMaxTokenLength, HiddenSize});
    MemRef<float, 2> sub2DContainer0({SubMaxTokenLength, HiddenSize});
    MemRef<float, 2> sub2DContainer1({SubMaxTokenLength, HiddenSize});

    // Get pointers once to avoid repeated calls
    float *subResultPtr = subResultContainer.getData();
    float *rmsPtr = sub3DContainer.getData();
    float *mhaMemRef2DPtr = mhaMemRef2D.getData();
    float *mhaMemRef3D1Ptr = mhaMemRef3D1.getData();
    float *mhaMemRef3D2Ptr = mhaMemRef3D2.getData();
    float *mhaPtr = tmp3DMemRef.getData();
    float *mhaOutputPtr = tmp2DContainer.getData();
    float *sub2DPtr = sub2DContainer.getData();
    float *sub2DPtr0 = sub2DContainer0.getData();
    float *sub2DPtr1 = sub2DContainer1.getData();

    constexpr size_t paramSizeRMS = 4096;
    constexpr size_t paramSizeMHA = 16777216;
    constexpr size_t paramSizeMLP = 33816576;
    int id = rank / 4;
    int init = id * 42;
    int endVal = (id + 1) * 42;
    int times = 7;
    int source = 0;
    int dest1 = rank - 2;
    int dest2 = rank - 1;
    int dest4 = rank + 1;
    int nextRank = rank + 4;
    if (rank == 19){
      endVal = 193;
      times = 4;
      nextRank = 0;
    }
    MPI_Request send_req[4], recv_req[4];
    MPI_Request mha_recv_req[3];
    bool mha_params_received = false;

    std::vector<std::string> paramsDirsRMS, paramsDirsRMS0;
    std::vector<std::string> paramsDirsMHA, paramsDirsMLP;
    std::vector<MemRef<float, 1>> paramsContainersRMS, paramsContainersRMS0;
    std::vector<MemRef<float, 1>> paramsContainersMHA, paramsContainersMLP;
    // Get Parameter file paths
    for (int i = init+1; i < endVal; i += 6) {
      paramsDirsRMS.emplace_back(llamaBuildDir + "/subgraph" +
                                 std::to_string(i) + "_arg0" + ".data");
      paramsDirsRMS0.emplace_back(llamaBuildDir + "/subgraph" +
                                  std::to_string(i + 3) + "_arg0" + ".data");
      paramsDirsMHA.emplace_back(llamaBuildDir + "/subgraph" +
                                 std::to_string(i + 1) + "_arg2" + ".data");
      paramsDirsMLP.emplace_back(llamaBuildDir + "/subgraph" +
                                 std::to_string(i + 4) + "_arg2" + ".data");
    }

    // Load parameters after Bcast to avoid blocking rank 2 at MPI_Barrier
    for (int i = 0; i < times; i++) {
      // First RMS
      MemRef<float, 1> paramsContainerRMS({paramSizeRMS});
      loadParameters(paramsDirsRMS[i], paramsContainerRMS);
      paramsContainersRMS.push_back(paramsContainerRMS);
      // MHA
      MemRef<float, 1> paramsContainerMHA({paramSizeMHA});
      loadParameters(paramsDirsMHA[i], paramsContainerMHA);
      paramsContainersMHA.push_back(paramsContainerMHA);
      // Second RMS
      MemRef<float, 1> paramsContainerRMS0({paramSizeRMS});
      loadParameters(paramsDirsRMS0[i], paramsContainerRMS0);
      paramsContainersRMS0.push_back(paramsContainerRMS0);
      // MLP
      MemRef<float, 1> paramsContainerMLP({paramSizeMLP});
      loadParameters(paramsDirsMLP[i], paramsContainerMLP);
      paramsContainersMLP.push_back(paramsContainerMLP);
    }

    for (int i = 0; i < generateLen; i++) {
      /* post MHA params receive once to overlap communication with work */
      MPI_Irecv(mhaMemRef2DPtr, MaxTokenLength * HiddenSize1, MPI_FLOAT,
                source, 1, MPI_COMM_WORLD, &mha_recv_req[0]);
      MPI_Irecv(mhaMemRef3D1Ptr, MaxTokenLength * HiddenSize0, MPI_FLOAT,
                source, 2, MPI_COMM_WORLD, &mha_recv_req[1]);
      MPI_Irecv(mhaMemRef3D2Ptr, MaxTokenLength * HiddenSize0, MPI_FLOAT,
                source, 3, MPI_COMM_WORLD, &mha_recv_req[2]);
      for (int m = 0; m < times; m++) {
        if (m == 0) {
          if (rank == 3) {
            MPI_Irecv(subResultPtr, subSize, MPI_FLOAT, source, 0,
                      MPI_COMM_WORLD, &recv_req[0]);
            MPI_Wait(&recv_req[0], MPI_STATUS_IGNORE);
          } else {
            MPI_Irecv(subResultPtr, subSize, MPI_FLOAT, rank - 4, 0,
                      MPI_COMM_WORLD, &recv_req[0]);
            MPI_Wait(&recv_req[0], MPI_STATUS_IGNORE);
          }
        }

        // const auto inferenceStart =
        // std::chrono::high_resolution_clock::now();
        // ----- RMS -----
        _mlir_ciface_forward1(&sub3DContainer, &paramsContainersRMS[m],
                              &subResultContainer);
        rmsPtr = sub3DContainer.getData();
        mhaPtr = tmp3DMemRef.getData();
        // const auto inferenceEnd = std::chrono::high_resolution_clock::now();
        // ----- AllGather -----
        MPI_Isend(rmsPtr, subSize, MPI_FLOAT, dest4, 0, MPI_COMM_WORLD,
                  &send_req[0]);
        MPI_Irecv(mhaPtr + offset2, subSize, MPI_FLOAT, dest4, 0,
                  MPI_COMM_WORLD, &recv_req[0]);
        for (int idx = 0; idx < subSize; idx++) {
          mhaPtr[idx + offset1] = rmsPtr[idx];
        }
        MPI_Wait(&send_req[0], MPI_STATUS_IGNORE);
        MPI_Wait(&recv_req[0], MPI_STATUS_IGNORE);

        MPI_Isend(mhaPtr + offset1, offset1, MPI_FLOAT, dest1, 0,
                  MPI_COMM_WORLD, &send_req[0]);
        MPI_Irecv(mhaPtr, offset1, MPI_FLOAT, dest1, 0, MPI_COMM_WORLD,
                  &recv_req[0]);
        MPI_Wait(&send_req[0], MPI_STATUS_IGNORE);
        MPI_Wait(&recv_req[0], MPI_STATUS_IGNORE);

        // const std::chrono::duration<double, std::milli> inferenceTime =
        //     inferenceEnd - inferenceStart;
        // std::cout << "[Log] RMSNorm: " << inferenceTime.count() / 1000 << "s"
        // << std::endl;

        // ----- MHA -----
        if(m == 0){
          MPI_Waitall(3, mha_recv_req, MPI_STATUSES_IGNORE);
        }
        _mlir_ciface_forward2(&tmp2DContainer, &paramsContainersMHA[m],
                              &tmp3DMemRef, &mhaMemRef3D1, &mhaMemRef3D2,
                              &mhaMemRef2D);
        // ----- Reduce-Scatter -----
        mhaOutputPtr = tmp2DContainer.getData();
        MPI_Isend(mhaOutputPtr + offset2, subSize, MPI_FLOAT, dest4, 0,
                  MPI_COMM_WORLD, &send_req[0]);
        MPI_Irecv(sub2DPtr, subSize, MPI_FLOAT, dest2, 0, MPI_COMM_WORLD,
                  &recv_req[0]);
        MPI_Wait(&send_req[0], MPI_STATUS_IGNORE);
        MPI_Wait(&recv_req[0], MPI_STATUS_IGNORE);

        MPI_Isend(mhaOutputPtr, subSize, MPI_FLOAT, dest1, 0, MPI_COMM_WORLD,
                  &send_req[0]);
        MPI_Irecv(sub2DPtr0, subSize, MPI_FLOAT, dest1, 0, MPI_COMM_WORLD,
                  &recv_req[0]);
        for (int idx = 0; idx < subSize; idx++) {
          sub2DPtr[idx] += mhaOutputPtr[idx + offset1];
        }
        MPI_Wait(&send_req[0], MPI_STATUS_IGNORE);
        MPI_Wait(&recv_req[0], MPI_STATUS_IGNORE);

        MPI_Isend(mhaOutputPtr + offset0, subSize, MPI_FLOAT, dest2, 0,
                  MPI_COMM_WORLD, &send_req[0]);
        MPI_Irecv(sub2DPtr1, subSize, MPI_FLOAT, dest4, 0, MPI_COMM_WORLD,
                  &recv_req[0]);
        for (int idx = 0; idx < subSize; idx++) {
          sub2DPtr[idx] += sub2DPtr0[idx];
        }
        MPI_Wait(&send_req[0], MPI_STATUS_IGNORE);
        MPI_Wait(&recv_req[0], MPI_STATUS_IGNORE);
        for (int idx = 0; idx < subSize; idx++) {
          sub2DPtr[idx] += sub2DPtr1[idx];
        }

        // ----- RMS & Add -----
        _mlir_ciface_forward3(&subResultContainer, &sub2DContainer,
                              &subResultContainer);
        _mlir_ciface_forward1(&sub3DContainer, &paramsContainersRMS0[m],
                              &subResultContainer);
        rmsPtr = sub3DContainer.getData();
        mhaPtr = tmp3DMemRef.getData();
        // const auto inferenceEnd = std::chrono::high_resolution_clock::now();
        // ----- AllGather -----
        MPI_Isend(rmsPtr, subSize, MPI_FLOAT, dest4, 0, MPI_COMM_WORLD,
                  &send_req[0]);
        MPI_Irecv(mhaPtr + offset2, subSize, MPI_FLOAT, dest4, 0,
                  MPI_COMM_WORLD, &recv_req[0]);
        for (int idx = 0; idx < subSize; idx++) {
          mhaPtr[idx + offset1] = rmsPtr[idx];
        }
        MPI_Wait(&send_req[0], MPI_STATUS_IGNORE);
        MPI_Wait(&recv_req[0], MPI_STATUS_IGNORE);

        MPI_Isend(mhaPtr + offset1, offset1, MPI_FLOAT, dest1, 0,
                  MPI_COMM_WORLD, &send_req[0]);
        MPI_Irecv(mhaPtr, offset1, MPI_FLOAT, dest1, 0, MPI_COMM_WORLD,
                  &recv_req[0]);
        MPI_Wait(&send_req[0], MPI_STATUS_IGNORE);
        MPI_Wait(&recv_req[0], MPI_STATUS_IGNORE);
        // ----- MLP -----
        _mlir_ciface_forward5(&tmp2DContainer, &paramsContainersMLP[m],
                              &tmp3DMemRef);
        // ----- Reduce-Scatter -----
        mhaOutputPtr = tmp2DContainer.getData();
        MPI_Isend(mhaOutputPtr + offset2, subSize, MPI_FLOAT, dest4, 0,
                  MPI_COMM_WORLD, &send_req[0]);
        MPI_Irecv(sub2DPtr, subSize, MPI_FLOAT, dest2, 0, MPI_COMM_WORLD,
                  &recv_req[0]);
        MPI_Wait(&send_req[0], MPI_STATUS_IGNORE);
        MPI_Wait(&recv_req[0], MPI_STATUS_IGNORE);

        MPI_Isend(mhaOutputPtr, subSize, MPI_FLOAT, dest1, 0, MPI_COMM_WORLD,
                  &send_req[0]);
        MPI_Irecv(sub2DPtr0, subSize, MPI_FLOAT, dest1, 0, MPI_COMM_WORLD,
                  &recv_req[0]);
        for (int idx = 0; idx < subSize; idx++) {
          sub2DPtr[idx] += mhaOutputPtr[idx + offset1];
        }
        MPI_Wait(&send_req[0], MPI_STATUS_IGNORE);
        MPI_Wait(&recv_req[0], MPI_STATUS_IGNORE);

        MPI_Isend(mhaOutputPtr + offset0, subSize, MPI_FLOAT, dest2, 0,
                  MPI_COMM_WORLD, &send_req[0]);
        MPI_Irecv(sub2DPtr1, subSize, MPI_FLOAT, dest4, 0, MPI_COMM_WORLD,
                  &recv_req[0]);
        for (int idx = 0; idx < subSize; idx++) {
          sub2DPtr[idx] += sub2DPtr0[idx];
        }
        MPI_Wait(&send_req[0], MPI_STATUS_IGNORE);
        MPI_Wait(&recv_req[0], MPI_STATUS_IGNORE);
        for (int idx = 0; idx < subSize; idx++) {
          sub2DPtr[idx] += sub2DPtr1[idx];
        }
        // ----- Add -----
        _mlir_ciface_forward3(&subResultContainer, &sub2DContainer,
                              &subResultContainer);
        if (m == (times - 1)) {
          subResultPtr = subResultContainer.getData();
          MPI_Send(subResultPtr, subSize, MPI_FLOAT, nextRank, 0, MPI_COMM_WORLD);
        }
      }
    }
  } else if (rank == 4 || rank == 8 || rank == 12 || rank == 16 ||
             rank == 20) {
    // === RMSNorm ===
    // Rank 4 specific variables
    MemRef<float, 3> subResultContainer({1, SubMaxTokenLength, HiddenSize});
    MemRef<float, 3> sub3DContainer({1, SubMaxTokenLength, HiddenSize});
    MemRef<float, 2> mhaMemRef2D({MaxTokenLength, HiddenSize1});
    MemRef<float, 3> mhaMemRef3D1({1, MaxTokenLength, HiddenSize0});
    MemRef<float, 3> mhaMemRef3D2({1, MaxTokenLength, HiddenSize0});
    MemRef<float, 3> tmp3DMemRef({1, MaxTokenLength, HiddenSize});
    MemRef<float, 2> tmp2DContainer({MaxTokenLength, HiddenSize});
    MemRef<float, 2> sub2DContainer({SubMaxTokenLength, HiddenSize});
    MemRef<float, 2> sub2DContainer0({SubMaxTokenLength, HiddenSize});
    MemRef<float, 2> sub2DContainer1({SubMaxTokenLength, HiddenSize});

    // Get pointers once to avoid repeated calls
    float *subResultPtr = subResultContainer.getData();
    float *rmsPtr = sub3DContainer.getData();
    float *mhaMemRef2DPtr = mhaMemRef2D.getData();
    float *mhaMemRef3D1Ptr = mhaMemRef3D1.getData();
    float *mhaMemRef3D2Ptr = mhaMemRef3D2.getData();
    float *mhaPtr = tmp3DMemRef.getData();
    float *mhaOutputPtr = tmp2DContainer.getData();
    float *sub2DPtr = sub2DContainer.getData();
    float *sub2DPtr0 = sub2DContainer0.getData();
    float *sub2DPtr1 = sub2DContainer1.getData();

    constexpr size_t paramSizeRMS = 4096;
    constexpr size_t paramSizeMHA = 16777216;
    constexpr size_t paramSizeMLP = 33816576;
    int id = rank / 4 - 1;
    int init = id * 42;
    int endVal = (id + 1) * 42;
    int times = 7;
    int source = 0;
    int dest1 = rank - 3;
    int dest2 = rank - 2;
    int dest3 = rank - 1;
    int nextRank = rank + 4;
    if (rank == 20){
      endVal = 193;
      times = 4;
      nextRank = 0;
    }
    MPI_Request send_req[4], recv_req[4];
    MPI_Request mha_recv_req[3];
    bool mha_params_received = false;

    std::vector<std::string> paramsDirsRMS, paramsDirsRMS0;
    std::vector<std::string> paramsDirsMHA, paramsDirsMLP;
    std::vector<MemRef<float, 1>> paramsContainersRMS, paramsContainersRMS0;
    std::vector<MemRef<float, 1>> paramsContainersMHA, paramsContainersMLP;
    // Get Parameter file paths
    
    for (int i = init+1; i < endVal; i += 6) {
      paramsDirsRMS.emplace_back(llamaBuildDir + "/subgraph" +
                                 std::to_string(i) + "_arg0" + ".data");
      paramsDirsRMS0.emplace_back(llamaBuildDir + "/subgraph" +
                                 std::to_string(i + 3) + "_arg0" + ".data");
      paramsDirsMHA.emplace_back(llamaBuildDir + "/subgraph" +
                                 std::to_string(i + 1) + "_arg3" + ".data");
      paramsDirsMLP.emplace_back(llamaBuildDir + "/subgraph" +
                                 std::to_string(i + 4) + "_arg3" + ".data");
    }

    // Load parameters after Bcast to avoid blocking rank 2 at MPI_Barrier
    for (int i = 0; i < times; i++) {
      // First RMS
      MemRef<float, 1> paramsContainerRMS({paramSizeRMS});
      loadParameters(paramsDirsRMS[i], paramsContainerRMS);
      paramsContainersRMS.push_back(paramsContainerRMS);
      // MHA
      MemRef<float, 1> paramsContainerMHA({paramSizeMHA});
      loadParameters(paramsDirsMHA[i], paramsContainerMHA);
      paramsContainersMHA.push_back(paramsContainerMHA);
      // Second RMS
      MemRef<float, 1> paramsContainerRMS0({paramSizeRMS});
      loadParameters(paramsDirsRMS0[i], paramsContainerRMS0);
      paramsContainersRMS0.push_back(paramsContainerRMS0);
      // MLP
      MemRef<float, 1> paramsContainerMLP({paramSizeMLP});
      loadParameters(paramsDirsMLP[i], paramsContainerMLP);
      paramsContainersMLP.push_back(paramsContainerMLP);
    }

    for (int i = 0; i < generateLen; i++) {
      /* post MHA params receive once to overlap communication with work */
      MPI_Irecv(mhaMemRef2DPtr, MaxTokenLength * HiddenSize1, MPI_FLOAT,
                source, 1, MPI_COMM_WORLD, &mha_recv_req[0]);
      MPI_Irecv(mhaMemRef3D1Ptr, MaxTokenLength * HiddenSize0, MPI_FLOAT,
                source, 2, MPI_COMM_WORLD, &mha_recv_req[1]);
      MPI_Irecv(mhaMemRef3D2Ptr, MaxTokenLength * HiddenSize0, MPI_FLOAT,
                source, 3, MPI_COMM_WORLD, &mha_recv_req[2]);
      for (int m = 0; m < times; m++) {
        if (m == 0) {
          if (rank == 4) {
            MPI_Irecv(subResultPtr, subSize, MPI_FLOAT, source, 0,
                      MPI_COMM_WORLD, &recv_req[0]);
            MPI_Wait(&recv_req[0], MPI_STATUS_IGNORE);
          } else {
            MPI_Irecv(subResultPtr, subSize, MPI_FLOAT, rank - 4, 0,
                      MPI_COMM_WORLD, &recv_req[0]);
            MPI_Wait(&recv_req[0], MPI_STATUS_IGNORE);
          }
        }

        // const auto inferenceStart =
        // std::chrono::high_resolution_clock::now();
        // ----- RMS -----
        _mlir_ciface_forward1(&sub3DContainer, &paramsContainersRMS[m],
                              &subResultContainer);
        rmsPtr = sub3DContainer.getData();
        mhaPtr = tmp3DMemRef.getData();
        // const auto inferenceEnd = std::chrono::high_resolution_clock::now();
        // ----- AllGather -----
        MPI_Isend(rmsPtr, subSize, MPI_FLOAT, dest3, 0, MPI_COMM_WORLD,
                  &send_req[0]);
        MPI_Irecv(mhaPtr + offset1, subSize, MPI_FLOAT, dest3, 0,
                  MPI_COMM_WORLD, &recv_req[0]);
        for (int idx = 0; idx < subSize; idx++) {
          mhaPtr[idx + offset2] = rmsPtr[idx];
        }
        MPI_Wait(&send_req[0], MPI_STATUS_IGNORE);
        MPI_Wait(&recv_req[0], MPI_STATUS_IGNORE);

        MPI_Isend(mhaPtr + offset1, offset1, MPI_FLOAT, dest2, 0,
                  MPI_COMM_WORLD, &send_req[0]);
        MPI_Irecv(mhaPtr, offset1, MPI_FLOAT, dest2, 0, MPI_COMM_WORLD,
                  &recv_req[0]);
        MPI_Wait(&send_req[0], MPI_STATUS_IGNORE);
        MPI_Wait(&recv_req[0], MPI_STATUS_IGNORE);

        // const std::chrono::duration<double, std::milli> inferenceTime =
        //     inferenceEnd - inferenceStart;
        // std::cout << "[Log] RMSNorm: " << inferenceTime.count() / 1000 << "s"
        // << std::endl;

        // ----- MHA -----
        if(m == 0){
          MPI_Waitall(3, mha_recv_req, MPI_STATUSES_IGNORE);
        }
        _mlir_ciface_forward2(&tmp2DContainer, &paramsContainersMHA[m],
                              &tmp3DMemRef, &mhaMemRef3D1, &mhaMemRef3D2,
                              &mhaMemRef2D);
        mhaOutputPtr = tmp2DContainer.getData();
        // ----- Reduce-Scatter -----
        MPI_Isend(mhaOutputPtr, subSize, MPI_FLOAT, dest1, 0, MPI_COMM_WORLD,
                  &send_req[0]);
        MPI_Irecv(sub2DPtr, subSize, MPI_FLOAT, dest3, 0, MPI_COMM_WORLD,
                  &recv_req[0]);
        MPI_Wait(&send_req[0], MPI_STATUS_IGNORE);
        MPI_Wait(&recv_req[0], MPI_STATUS_IGNORE);

        MPI_Isend(mhaOutputPtr + offset0, subSize, MPI_FLOAT, dest2, 0,
                  MPI_COMM_WORLD, &send_req[0]);
        MPI_Irecv(sub2DPtr0, subSize, MPI_FLOAT, dest2, 0, MPI_COMM_WORLD,
                  &recv_req[0]);
        for (int idx = 0; idx < subSize; idx++) {
          sub2DPtr[idx] += mhaOutputPtr[idx + offset2];
        }
        MPI_Wait(&send_req[0], MPI_STATUS_IGNORE);
        MPI_Wait(&recv_req[0], MPI_STATUS_IGNORE);

        MPI_Isend(mhaOutputPtr + offset1, subSize, MPI_FLOAT, dest3, 0,
                  MPI_COMM_WORLD, &send_req[0]);
        MPI_Irecv(sub2DPtr1, subSize, MPI_FLOAT, dest1, 0, MPI_COMM_WORLD,
                  &recv_req[0]);
        for (int idx = 0; idx < subSize; idx++) {
          sub2DPtr[idx] += sub2DPtr0[idx];
        }
        MPI_Wait(&send_req[0], MPI_STATUS_IGNORE);
        MPI_Wait(&recv_req[0], MPI_STATUS_IGNORE);
        for (int idx = 0; idx < subSize; idx++) {
          sub2DPtr[idx] += sub2DPtr1[idx];
        }

        // ----- RMS & Add -----
        _mlir_ciface_forward3(&subResultContainer, &sub2DContainer,
                              &subResultContainer);
        _mlir_ciface_forward1(&sub3DContainer, &paramsContainersRMS0[m],
                              &subResultContainer);
        rmsPtr = sub3DContainer.getData();
        mhaPtr = tmp3DMemRef.getData();
        // const auto inferenceEnd = std::chrono::high_resolution_clock::now();
        // ----- AllGather -----
        MPI_Isend(rmsPtr, subSize, MPI_FLOAT, dest3, 0, MPI_COMM_WORLD,
                  &send_req[0]);
        MPI_Irecv(mhaPtr + offset1, subSize, MPI_FLOAT, dest3, 0,
                  MPI_COMM_WORLD, &recv_req[0]);
        for (int idx = 0; idx < subSize; idx++) {
          mhaPtr[idx + offset2] = rmsPtr[idx];
        }
        MPI_Wait(&send_req[0], MPI_STATUS_IGNORE);
        MPI_Wait(&recv_req[0], MPI_STATUS_IGNORE);

        MPI_Isend(mhaPtr + offset1, offset1, MPI_FLOAT, dest2, 0,
                  MPI_COMM_WORLD, &send_req[0]);
        MPI_Irecv(mhaPtr, offset1, MPI_FLOAT, dest2, 0, MPI_COMM_WORLD,
                  &recv_req[0]);
        MPI_Wait(&send_req[0], MPI_STATUS_IGNORE);
        MPI_Wait(&recv_req[0], MPI_STATUS_IGNORE);
        // ----- MLP -----
        _mlir_ciface_forward5(&tmp2DContainer, &paramsContainersMLP[m],
                              &tmp3DMemRef);
        mhaOutputPtr = tmp2DContainer.getData();
        // ----- Reduce-Scatter -----
        MPI_Isend(mhaOutputPtr, subSize, MPI_FLOAT, dest1, 0, MPI_COMM_WORLD,
                  &send_req[0]);
        MPI_Irecv(sub2DPtr, subSize, MPI_FLOAT, dest3, 0, MPI_COMM_WORLD,
                  &recv_req[0]);
        MPI_Wait(&send_req[0], MPI_STATUS_IGNORE);
        MPI_Wait(&recv_req[0], MPI_STATUS_IGNORE);

        MPI_Isend(mhaOutputPtr + offset0, subSize, MPI_FLOAT, dest2, 0,
                  MPI_COMM_WORLD, &send_req[0]);
        MPI_Irecv(sub2DPtr0, subSize, MPI_FLOAT, dest2, 0, MPI_COMM_WORLD,
                  &recv_req[0]);
        for (int idx = 0; idx < subSize; idx++) {
          sub2DPtr[idx] += mhaOutputPtr[idx + offset2];
        }
        MPI_Wait(&send_req[0], MPI_STATUS_IGNORE);
        MPI_Wait(&recv_req[0], MPI_STATUS_IGNORE);

        MPI_Isend(mhaOutputPtr + offset1, subSize, MPI_FLOAT, dest3, 0,
                  MPI_COMM_WORLD, &send_req[0]);
        MPI_Irecv(sub2DPtr1, subSize, MPI_FLOAT, dest1, 0, MPI_COMM_WORLD,
                  &recv_req[0]);
        for (int idx = 0; idx < subSize; idx++) {
          sub2DPtr[idx] += sub2DPtr0[idx];
        }
        MPI_Wait(&send_req[0], MPI_STATUS_IGNORE);
        MPI_Wait(&recv_req[0], MPI_STATUS_IGNORE);
        for (int idx = 0; idx < subSize; idx++) {
          sub2DPtr[idx] += sub2DPtr1[idx];
        }
        // ----- Add -----
        _mlir_ciface_forward3(&subResultContainer, &sub2DContainer,
                              &subResultContainer);
        if (m == (times - 1)) {
          subResultPtr = subResultContainer.getData();
          MPI_Send(subResultPtr, subSize, MPI_FLOAT, nextRank, 0, MPI_COMM_WORLD);
        }
      }
    }
  }

  MPI_Finalize();
  return 0;
}
