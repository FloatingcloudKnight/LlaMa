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
#include <vector>
#include <array>

using namespace buddy;

// ============================================================================
// Constants
// ============================================================================
constexpr size_t MaxVocabSize = 32000;
constexpr size_t MaxTokenLength = 40;
constexpr size_t SubMaxTokenLength = 5;
constexpr size_t HiddenSize = 4096;
constexpr size_t HiddenSize0 = 128;
constexpr size_t HiddenSize1 = 41;
constexpr size_t NUM_RANKS = 9;
constexpr size_t NUM_LAYERS = 32;

// Parameter sizes
constexpr size_t ParamSizeRMS = 4096;
constexpr size_t ParamSizeMHA = 8388608;
constexpr size_t ParamSizeMLP = 16908288;

// ============================================================================
// Data Structures
// ============================================================================
struct MemRefContainer {
  MemRef<float, 3> memRef3D0;
  MemRef<float, 2> memRef2D;
  MemRef<float, 3> memRef3D1;
  MemRef<float, 3> memRef3D2;

  MemRefContainer(MemRef<float, 3> m1, MemRef<float, 2> m2, MemRef<float, 3> m3,
                  MemRef<float, 3> m4)
      : memRef3D0(m1), memRef2D(m2), memRef3D1(m3), memRef3D2(m4) {}
};

/// Container for all parameters of a single rank
struct ParameterSet {
  std::vector<MemRef<float, 1>> rmsParams;      // RMS Norm Layer 1
  std::vector<MemRef<float, 1>> mhaParams;      // Multi-Head Attention
  std::vector<MemRef<float, 1>> rmsParams2;     // RMS Norm Layer 2
  std::vector<MemRef<float, 1>> mlpParams;      // Feed Forward Network

  ParameterSet() = default;
};

/// Configuration for AllGather communication pattern
struct AllGatherConfig {
  int dest1, dest2, dest3;
  size_t offsetRecv1, offsetRecv2, offsetRecv3;
  size_t sendSize, recvSize;
};

/// Configuration for Reduce-Scatter communication
struct ReduceScatterPair {
  int destRank;
  size_t sendOffset;
  int recvRank;
  int recvIndex;  // 0, 1, or 2 for ptr0, ptr1, ptr2
};

/// Per-rank communication topology
struct RankTopology {
  int rankId;
  AllGatherConfig allgatherConfig;
  std::vector<ReduceScatterPair> reduceScatterPairs;
  std::vector<int> broadcastDests;
};

// ============================================================================
// External MLIR Functions
// ============================================================================
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

// ============================================================================
// Utility Functions
// ============================================================================

/// Get user input from stdin
void getUserInput(std::string &inputStr) {
  std::cout << "\nPlease send a message:" << std::endl;
  std::cout << ">>> ";
  getline(std::cin, inputStr);
  std::cout << std::endl;
}

/// Print log label in blue
void printLogLabel() { std::cout << "\033[34;1m[Log] \033[0m"; }

/// Print iteration information
void printIterInfo(size_t iterIdx, std::string str, double time) {
  std::cout << "\033[32;1m[Iteration " << iterIdx << "] \033[0m";
  std::cout << "Token: " << str << " | "
            << "Time: " << time << "s" << std::endl;
}

/// Tokenize input text
void tokenizeInput(const std::string &vocabFile,
                   Text<size_t, 2> &inputContainer) {
  inputContainer.tokenizeLlama(vocabFile, MaxTokenLength);
}

/// Load parameters from file
void loadParameters(const std::string &paramFilePath,
                    MemRef<float, 1> &params) {
  std::ifstream paramFile(paramFilePath, std::ios::in | std::ios::binary);
  if (!paramFile.is_open()) {
    std::cerr << "Error: Failed to open params file: " << paramFilePath
              << std::endl;
    throw std::runtime_error("Failed to open params file!");
  }
  
  paramFile.read(reinterpret_cast<char *>(params.getData()),
                 sizeof(float) * params.getSize());
  
  if (paramFile.fail()) {
    throw std::runtime_error("Error occurred while reading params file!");
  }
  
  paramFile.close();
}

/// Find index of maximum value in float array
int findMaxIndex(const float *start, const float *end) {
  return std::distance(start, std::max_element(start, end));
}

// ============================================================================
// Parameter Management
// ============================================================================

/// Load all parameters for a worker rank (ranks 1-8)
ParameterSet loadAllParameters(const std::string &llamaBuildDir, int rankId) {
  ParameterSet params;

  for (int i = 0; i < NUM_LAYERS; i++) {
    // First RMS Normalization
    MemRef<float, 1> rmsParam({ParamSizeRMS});
    loadParameters(
        llamaBuildDir + "/subgraph" + std::to_string(1 + i * 6) + "_arg0.data",
        rmsParam);
    params.rmsParams.push_back(rmsParam);

    // Multi-Head Attention
    MemRef<float, 1> mhaParam({ParamSizeMHA});
    loadParameters(
        llamaBuildDir + "/subgraph" + std::to_string(2 + i * 6) + "_arg" +
        std::to_string(rankId - 1) + ".data",
        mhaParam);
    params.mhaParams.push_back(mhaParam);

    // Second RMS Normalization
    MemRef<float, 1> rmsParam2({ParamSizeRMS});
    loadParameters(
        llamaBuildDir + "/subgraph" + std::to_string(4 + i * 6) + "_arg0.data",
        rmsParam2);
    params.rmsParams2.push_back(rmsParam2);

    // Feed Forward Network (MLP)
    MemRef<float, 1> mlpParam({ParamSizeMLP});
    loadParameters(
        llamaBuildDir + "/subgraph" + std::to_string(5 + i * 6) + "_arg" +
        std::to_string(rankId - 1) + ".data",
        mlpParam);
    params.mlpParams.push_back(mlpParam);
  }

  return params;
}

// ============================================================================
// Communication Patterns
// ============================================================================

/// Perform AllGather communication
void performAllGather(float *mhaPtr, float *rmsPtr, size_t subSize,
                     const AllGatherConfig &config, MPI_Request *send_req,
                     MPI_Request *recv_req) {
  // Exchange 1: Send to dest1, receive from dest1
  MPI_Isend(rmsPtr, subSize, MPI_FLOAT, config.dest1, 0, MPI_COMM_WORLD,
            &send_req[0]);
  MPI_Irecv(mhaPtr + config.offsetRecv1, subSize, MPI_FLOAT, config.dest1, 0,
            MPI_COMM_WORLD, &recv_req[0]);
  
  // Copy local data
  for (size_t idx = 0; idx < subSize; idx++) {
    mhaPtr[idx] = rmsPtr[idx];
  }
  
  MPI_Wait(&send_req[0], MPI_STATUS_IGNORE);
  MPI_Wait(&recv_req[0], MPI_STATUS_IGNORE);

  // Exchange 2: Send to dest2, receive from dest2
  MPI_Isend(mhaPtr + config.offsetRecv1, config.recvSize, MPI_FLOAT,
            config.dest2, 0, MPI_COMM_WORLD, &send_req[0]);
  MPI_Irecv(mhaPtr + config.offsetRecv2, config.recvSize, MPI_FLOAT,
            config.dest2, 0, MPI_COMM_WORLD, &recv_req[0]);
  MPI_Wait(&send_req[0], MPI_STATUS_IGNORE);
  MPI_Wait(&recv_req[0], MPI_STATUS_IGNORE);

  // Exchange 3: Send to dest3, receive from dest3
  MPI_Isend(mhaPtr, config.recvSize, MPI_FLOAT, config.dest3, 0,
            MPI_COMM_WORLD, &send_req[0]);
  MPI_Irecv(mhaPtr + config.offsetRecv3, config.recvSize, MPI_FLOAT,
            config.dest3, 0, MPI_COMM_WORLD, &recv_req[0]);
  MPI_Wait(&send_req[0], MPI_STATUS_IGNORE);
  MPI_Wait(&recv_req[0], MPI_STATUS_IGNORE);
}

/// Accumulate array values
void accumulateArray(float *dest, const float *src, size_t size) {
  for (size_t idx = 0; idx < size; idx++) {
    dest[idx] += src[idx];
  }
}

/// Perform Reduce-Scatter communication
void performReduceScatter(float *outputPtr, 
                         std::array<float*, 3> accumPtrs,
                         size_t subSize,
                         const std::vector<ReduceScatterPair> &pairs,
                         MPI_Request *send_req,
                         MPI_Request *recv_req) {
  size_t numPairs = pairs.size();
  
  // Issue all sends and recvs
  for (size_t i = 0; i < numPairs; i++) {
    MPI_Isend(outputPtr + pairs[i].sendOffset, subSize, MPI_FLOAT,
              pairs[i].destRank, 0, MPI_COMM_WORLD, &send_req[i]);
    MPI_Irecv(accumPtrs[pairs[i].recvIndex], subSize, MPI_FLOAT,
              pairs[i].recvRank, 0, MPI_COMM_WORLD, &recv_req[i]);
  }

  // Wait and accumulate
  for (size_t i = 0; i < numPairs; i++) {
    MPI_Wait(&send_req[i], MPI_STATUS_IGNORE);
    MPI_Wait(&recv_req[i], MPI_STATUS_IGNORE);

    if (i > 0) {
      accumulateArray(accumPtrs[0], 
                     (i == 1) ? accumPtrs[1] : accumPtrs[2], 
                     subSize);
    }
  }
  
  // Final accumulation
  accumulateArray(accumPtrs[0], outputPtr, subSize);
}

// ============================================================================
// Worker Rank Processing
// ============================================================================

/// Main processing loop for worker ranks (ranks 1-8)
void processWorkerRank(int rank, int generateLen, size_t subSize,
                      const std::string &llamaBuildDir) {
  // Initialize containers
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

  // Get pointers
  float *subResultPtr = subResultContainer.getData();
  float *rmsPtr = sub3DContainer.getData();
  float *mhaMemRef2DPtr = mhaMemRef2D.getData();
  float *mhaMemRef3D1Ptr = mhaMemRef3D1.getData();
  float *mhaMemRef3D2Ptr = mhaMemRef3D2.getData();
  float *mhaPtr = tmp3DMemRef.getData();
  float *mhaOutputPtr = tmp2DContainer.getData();
  float *sub2DPtr0 = sub2DContainer.getData();
  float *sub2DPtr1 = sub2DContainer0.getData();
  float *sub2DPtr2 = sub2DContainer1.getData();

  // Load parameters
  ParameterSet params = loadAllParameters(llamaBuildDir, rank);

  MPI_Request send_req[32], recv_req[9];
  int source = 0;

  // Main inference loop
  for (int i = 0; i < generateLen; i++) {
    // Receive broadcasted attention data
    MPI_Irecv(mhaMemRef2DPtr, MaxTokenLength * HiddenSize1, MPI_FLOAT, source,
              1, MPI_COMM_WORLD, &recv_req[0]);
    MPI_Irecv(mhaMemRef3D1Ptr, MaxTokenLength * HiddenSize0, MPI_FLOAT, source,
              2, MPI_COMM_WORLD, &recv_req[1]);
    MPI_Irecv(mhaMemRef3D2Ptr, MaxTokenLength * HiddenSize0, MPI_FLOAT, source,
              3, MPI_COMM_WORLD, &recv_req[2]);
    MPI_Waitall(3, recv_req, MPI_STATUSES_IGNORE);

    // Process each transformer layer
    for (int m = 0; m < NUM_LAYERS; m++) {
      // Receive hidden states on first layer
      if (m == 0) {
        MPI_Irecv(subResultPtr, subSize, MPI_FLOAT, source, 0, MPI_COMM_WORLD,
                  &recv_req[0]);
        MPI_Wait(&recv_req[0], MPI_STATUS_IGNORE);
      }

      // ===== First RMS Normalization =====
      _mlir_ciface_forward1(&sub3DContainer, &params.rmsParams[m],
                            &subResultContainer);
      rmsPtr = sub3DContainer.getData();

      // Note: AllGather would be performed here with rank-specific config
      // For full implementation, see OPTIMIZATION_GUIDE.md

      // ===== Multi-Head Attention =====
      _mlir_ciface_forward2(&tmp2DContainer, &params.mhaParams[m], &tmp3DMemRef,
                            &mhaMemRef3D1, &mhaMemRef3D2, &mhaMemRef2D);
      mhaOutputPtr = tmp2DContainer.getData();

      // Note: Reduce-Scatter would be performed here
      // For full implementation, see OPTIMIZATION_GUIDE.md

      // ===== Second RMS Normalization =====
      _mlir_ciface_forward3(&subResultContainer, &sub2DContainer,
                            &subResultContainer);
      _mlir_ciface_forward1(&sub3DContainer, &params.rmsParams2[m],
                            &subResultContainer);

      // ===== Feed Forward Network (MLP) =====
      _mlir_ciface_forward5(&tmp2DContainer, &params.mlpParams[m],
                            &tmp3DMemRef);
      mhaOutputPtr = tmp2DContainer.getData();

      // ===== Residual Connection =====
      _mlir_ciface_forward3(&subResultContainer, &sub2DContainer,
                            &subResultContainer);

      // Send result back to rank 0 on last layer
      if (m == NUM_LAYERS - 1) {
        subResultPtr = subResultContainer.getData();
        MPI_Send(subResultPtr, subSize, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
      }
    }
  }
}

// ============================================================================
// Main Rank 0 Processing
// ============================================================================

void processRank0(const std::string &llamaDir, const std::string &llamaBuildDir,
                 int generateLen) {
  const std::string title = "LLaMA 2 Inference Powered by Buddy Compiler";
  std::cout << "\033[33;1m" << title << "\033[0m" << std::endl;

  const std::string vocabDir = llamaDir + "/vocab.txt";

  // Load vocabulary
  Text<size_t, 2> outputContainer;
  outputContainer.loadVocab(vocabDir);

  // Initialize memory containers
  MemRef<float, 3> myMemRef1({1, MaxTokenLength, HiddenSize});
  MemRef<float, 2> myMemRef2({MaxTokenLength, HiddenSize1});
  MemRef<float, 3> myMemRef3({1, MaxTokenLength, HiddenSize0});
  MemRef<float, 3> myMemRef4({1, MaxTokenLength, HiddenSize0});
  MemRefContainer tmpContainer(myMemRef1, myMemRef2, myMemRef3, myMemRef4);
  MemRefContainer *inputResultContainerPtr = &tmpContainer;
  MemRef<float, 3> tmp3DMemRef({1, MaxTokenLength, HiddenSize});

  // Load parameters
  constexpr size_t param_size0 = 131072064;
  constexpr size_t param_size1 = 131076096;
  const std::string paramsDir0 = llamaBuildDir + "/subgraph0_arg0.data";
  const std::string paramsDir1 = llamaBuildDir + "/subgraph193_arg0.data";

  MemRef<float, 1> paramsContainer0({param_size0});
  loadParameters(paramsDir0, paramsContainer0);

  MemRef<float, 1> paramsContainer1({param_size1});
  loadParameters(paramsDir1, paramsContainer1);

  // Get user input
  std::string inputStr;
  getUserInput(inputStr);
  Text<size_t, 2> inputContainer(inputStr);
  tokenizeInput(vocabDir, inputContainer);
  generateLen = MaxTokenLength - inputContainer.getTokenCnt();

  float *inputPtr = nullptr;
  float *outputPtr = tmp3DMemRef.getData();
  int subSize = SubMaxTokenLength * HiddenSize;

  MPI_Request send_req[32];

  // Main generation loop
  for (int i = 0; i < generateLen; i++) {
    const auto inferenceStart = std::chrono::high_resolution_clock::now();

    // Forward pass through first layer
    _mlir_ciface_forward0(inputResultContainerPtr, &paramsContainer0,
                          &inputContainer);
    inputPtr = inputResultContainerPtr->memRef3D0.getData();

    // Broadcast to all worker ranks
    for (int rankId = 1; rankId < NUM_RANKS; rankId++) {
      MPI_Isend(inputPtr + (rankId - 1) * subSize, subSize, MPI_FLOAT, rankId,
                0, MPI_COMM_WORLD, &send_req[rankId - 1]);
    }

    // Broadcast attention related data to all workers
    for (int rankId = 1; rankId < NUM_RANKS; rankId++) {
      size_t offset = (rankId - 2) * 3;
      MPI_Isend(inputResultContainerPtr->memRef2D.getData(),
                MaxTokenLength * HiddenSize1, MPI_FLOAT, rankId, 1,
                MPI_COMM_WORLD, &send_req[8 + offset]);
      MPI_Isend(inputResultContainerPtr->memRef3D1.getData(),
                MaxTokenLength * HiddenSize0, MPI_FLOAT, rankId, 2,
                MPI_COMM_WORLD, &send_req[9 + offset]);
      MPI_Isend(inputResultContainerPtr->memRef3D2.getData(),
                MaxTokenLength * HiddenSize0, MPI_FLOAT, rankId, 3,
                MPI_COMM_WORLD, &send_req[10 + offset]);
    }

    MPI_Waitall(32, send_req, MPI_STATUSES_IGNORE);

    // Receive results from all worker ranks
    MPI_Request recv_req[8];
    for (int rankId = 1; rankId < NUM_RANKS; rankId++) {
      MPI_Irecv(outputPtr + (rankId - 1) * subSize, subSize, MPI_FLOAT, rankId,
                0, MPI_COMM_WORLD, &recv_req[rankId - 1]);
    }
    MPI_Waitall(8, recv_req, MPI_STATUSES_IGNORE);

    // Forward pass through final layer
    _mlir_ciface_forward193(&inputResultContainerPtr->memRef3D0,
                            &paramsContainer1, &tmp3DMemRef);

    const auto inferenceEnd = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double, std::milli> inferenceTime =
        inferenceEnd - inferenceStart;

    // Decode generated token
    inputPtr = inputResultContainerPtr->memRef3D0.getData();
    int tokenIndex = inputContainer.getTokenCnt() - 1;
    const float *startPtr = inputPtr + tokenIndex * MaxVocabSize;
    const float *endPtr = startPtr + MaxVocabSize;
    int maxIndex = findMaxIndex(startPtr, endPtr);
    std::string tok = inputContainer.getStr(maxIndex);

    printIterInfo(i, tok, inferenceTime.count() / 1000);

    // Stop on end-of-sequence token
    if (maxIndex == 2) {
      break;
    }

    // Update for next iteration
    inputContainer.appendTokenIdx(maxIndex);
    outputContainer.appendTokenIdx(maxIndex);
  }
}

// ============================================================================
// Main Entry Point
// ============================================================================

int main(int argc, char *argv[]) {
  std::string llamaDir = LLAMA_SPLIT_EXAMPLE_PATH;
  std::string llamaBuildDir = LLAMA_EXAMPLE_BUILD_PATH;

  int rank, size;
  int generateLen = MaxTokenLength;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  try {
    if (rank == 0) {
      processRank0(llamaDir, llamaBuildDir, generateLen);
    } else {
      int subSize = SubMaxTokenLength * HiddenSize;
      processWorkerRank(rank, generateLen, subSize, llamaBuildDir);
    }
  } catch (const std::exception &e) {
    std::cerr << "Error in rank " << rank << ": " << e.what() << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  MPI_Finalize();
  return 0;
}
