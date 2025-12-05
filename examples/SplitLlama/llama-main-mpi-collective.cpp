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
#include <cstring>

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

struct ParameterSet {
  std::vector<MemRef<float, 1>> rmsParams;
  std::vector<MemRef<float, 1>> mhaParams;
  std::vector<MemRef<float, 1>> rmsParams2;
  std::vector<MemRef<float, 1>> mlpParams;
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
// MPI 集合通信包装类
// ============================================================================

/// 统一的 MPI 集合通信接口
class CollectiveComm {
public:
  /// 执行 AllGather - 所有 rank 将各自的数据发送给所有 rank
  /// 结果：每个 rank 都获得所有 rank 的数据拼接
  static void allgather(const float *sendBuf, size_t sendCount,
                       float *recvBuf) {
    MPI_Allgather(sendBuf, sendCount, MPI_FLOAT,
                 recvBuf, sendCount, MPI_FLOAT,
                 MPI_COMM_WORLD);
  }

  /// 执行 Reduce-Scatter - 汇聚所有数据，然后分散
  /// 结果：每个 rank 获得其对应部分的求和结果
  static void reduceScatter(const float *sendBuf, float *recvBuf,
                           size_t countPerRank) {
    int recvCounts[NUM_RANKS];
    for (size_t i = 0; i < NUM_RANKS; i++) {
      recvCounts[i] = countPerRank;
    }
    MPI_Reduce_scatter(sendBuf, recvBuf, recvCounts, MPI_FLOAT,
                      MPI_SUM, MPI_COMM_WORLD);
  }

  /// 执行 AllToAll - 所有 rank 的数据进行全排列交换
  /// 结果：rank i 从 rank j 接收第 i*dataPerRank 到 (i+1)*dataPerRank 的数据
  static void alltoall(const float *sendBuf, float *recvBuf,
                      size_t dataPerRank) {
    MPI_Alltoall(sendBuf, dataPerRank, MPI_FLOAT,
                recvBuf, dataPerRank, MPI_FLOAT,
                MPI_COMM_WORLD);
  }

  /// 执行 Broadcast - rank 0 将数据发送给所有其他 rank
  static void broadcast(float *data, size_t count, int root = 0) {
    MPI_Bcast(data, count, MPI_FLOAT, root, MPI_COMM_WORLD);
  }

  /// 执行全局 Reduce - 所有 rank 的数据进行求和，结果存储在 root rank
  static void reduce(const float *sendBuf, float *recvBuf, size_t count,
                    int root = 0) {
    MPI_Reduce(sendBuf, recvBuf, count, MPI_FLOAT,
              MPI_SUM, root, MPI_COMM_WORLD);
  }

  /// 全局同步屏障
  static void barrier() {
    MPI_Barrier(MPI_COMM_WORLD);
  }
};

// ============================================================================
// Utility Functions
// ============================================================================

void getUserInput(std::string &inputStr) {
  std::cout << "\nPlease send a message:" << std::endl;
  std::cout << ">>> ";
  getline(std::cin, inputStr);
  std::cout << std::endl;
}

void printLogLabel() { std::cout << "\033[34;1m[Log] \033[0m"; }

void printIterInfo(size_t iterIdx, std::string str, double time) {
  std::cout << "\033[32;1m[Iteration " << iterIdx << "] \033[0m";
  std::cout << "Token: " << str << " | "
            << "Time: " << time << "s" << std::endl;
}

void tokenizeInput(const std::string &vocabFile,
                   Text<size_t, 2> &inputContainer) {
  inputContainer.tokenizeLlama(vocabFile, MaxTokenLength);
}

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

int findMaxIndex(const float *start, const float *end) {
  return std::distance(start, std::max_element(start, end));
}

// ============================================================================
// Parameter Management
// ============================================================================

ParameterSet loadAllParameters(const std::string &llamaBuildDir, int rankId) {
  ParameterSet params;

  for (int i = 0; i < NUM_LAYERS; i++) {
    MemRef<float, 1> rmsParam({ParamSizeRMS});
    loadParameters(
        llamaBuildDir + "/subgraph" + std::to_string(1 + i * 6) + "_arg0.data",
        rmsParam);
    params.rmsParams.push_back(rmsParam);

    MemRef<float, 1> mhaParam({ParamSizeMHA});
    loadParameters(
        llamaBuildDir + "/subgraph" + std::to_string(2 + i * 6) + "_arg" +
        std::to_string(rankId - 1) + ".data",
        mhaParam);
    params.mhaParams.push_back(mhaParam);

    MemRef<float, 1> rmsParam2({ParamSizeRMS});
    loadParameters(
        llamaBuildDir + "/subgraph" + std::to_string(4 + i * 6) + "_arg0.data",
        rmsParam2);
    params.rmsParams2.push_back(rmsParam2);

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
// Worker Rank Processing with MPI Collective Ops
// ============================================================================

/// 简化的 Worker Rank 处理 - 使用 MPI 集合操作
void processWorkerRankSimplified(int rank, int generateLen, size_t subSize,
                                const std::string &llamaBuildDir) {
  // 初始化容器
  MemRef<float, 3> subResultContainer({1, SubMaxTokenLength, HiddenSize});
  MemRef<float, 3> sub3DContainer({1, SubMaxTokenLength, HiddenSize});
  MemRef<float, 2> mhaMemRef2D({MaxTokenLength, HiddenSize1});
  MemRef<float, 3> mhaMemRef3D1({1, MaxTokenLength, HiddenSize0});
  MemRef<float, 3> mhaMemRef3D2({1, MaxTokenLength, HiddenSize0});
  MemRef<float, 3> tmp3DMemRef({1, MaxTokenLength, HiddenSize});
  MemRef<float, 2> tmp2DContainer({MaxTokenLength, HiddenSize});
  MemRef<float, 2> sub2DContainer({SubMaxTokenLength, HiddenSize});

  // 全局 AllGather 缓冲区（注意：需要足够的内存）
  float rmsGlobal[NUM_RANKS * SubMaxTokenLength * HiddenSize];
  float mhaGlobal[NUM_RANKS * SubMaxTokenLength * HiddenSize];
  float mlpGlobal[NUM_RANKS * SubMaxTokenLength * HiddenSize];
  float allGatherBuf[NUM_RANKS * SubMaxTokenLength * HiddenSize];

  // 加载参数
  ParameterSet params = loadAllParameters(llamaBuildDir, rank);

  // 主推理循环
  for (int i = 0; i < generateLen; i++) {
    // 广播初始输入数据到所有 rank
    CollectiveComm::broadcast(mhaMemRef2D.getData(),
                             MaxTokenLength * HiddenSize1, 0);
    CollectiveComm::broadcast(mhaMemRef3D1.getData(),
                             MaxTokenLength * HiddenSize0, 0);
    CollectiveComm::broadcast(mhaMemRef3D2.getData(),
                             MaxTokenLength * HiddenSize0, 0);

    // 处理每一层
    for (int m = 0; m < NUM_LAYERS; m++) {
      // ===== 接收本 rank 的输入（仅在第一层） =====
      if (m == 0) {
        CollectiveComm::broadcast(subResultContainer.getData(),
                                 subSize, 0);
      }

      // ===== 第一个 RMS Norm 层 =====
      _mlir_ciface_forward1(&sub3DContainer, &params.rmsParams[m],
                           &subResultContainer);

      // ===== AllGather RMS 结果 =====
      // 所有 rank 共享 RMS 输出
      CollectiveComm::allgather(sub3DContainer.getData(), subSize,
                               rmsGlobal);
      
      // 使用全局 RMS 数据作为 MHA 输入
      memcpy(tmp3DMemRef.getData(), rmsGlobal,
             NUM_RANKS * SubMaxTokenLength * HiddenSize * sizeof(float));

      // ===== 多头注意力 =====
      _mlir_ciface_forward2(&tmp2DContainer, &params.mhaParams[m],
                           &tmp3DMemRef, &mhaMemRef3D1, &mhaMemRef3D2,
                           &mhaMemRef2D);

      // ===== Reduce-Scatter MHA 输出 =====
      // 每个 rank 获得其对应部分的求和结果
      CollectiveComm::allgather(tmp2DContainer.getData(), subSize,
                               mhaGlobal);

      float mhaReducedBuf[NUM_RANKS * SubMaxTokenLength * HiddenSize];
      // 复制全局数据用于 Reduce-Scatter
      memcpy(mhaReducedBuf, mhaGlobal,
             NUM_RANKS * SubMaxTokenLength * HiddenSize * sizeof(float));

      float mhaReduced[SubMaxTokenLength * HiddenSize];
      CollectiveComm::reduceScatter(mhaReducedBuf, mhaReduced, subSize);

      // 添加残差连接
      _mlir_ciface_forward3(&subResultContainer, &sub2DContainer,
                           &subResultContainer);

      // ===== 第二个 RMS Norm 层 =====
      _mlir_ciface_forward1(&sub3DContainer, &params.rmsParams2[m],
                           &subResultContainer);

      // 再次 AllGather
      CollectiveComm::allgather(sub3DContainer.getData(), subSize,
                               rmsGlobal);

      memcpy(tmp3DMemRef.getData(), rmsGlobal,
             NUM_RANKS * SubMaxTokenLength * HiddenSize * sizeof(float));

      // ===== 前馈网络 (MLP) =====
      _mlir_ciface_forward5(&tmp2DContainer, &params.mlpParams[m],
                           &tmp3DMemRef);

      // ===== Reduce-Scatter MLP 输出 =====
      CollectiveComm::allgather(tmp2DContainer.getData(), subSize,
                               mlpGlobal);

      memcpy(mhaReducedBuf, mlpGlobal,
             NUM_RANKS * SubMaxTokenLength * HiddenSize * sizeof(float));

      float mlpReduced[SubMaxTokenLength * HiddenSize];
      CollectiveComm::reduceScatter(mhaReducedBuf, mlpReduced, subSize);

      // ===== 最后的残差连接 =====
      _mlir_ciface_forward3(&subResultContainer, &sub2DContainer,
                           &subResultContainer);

      // 在最后一层后同步并发送结果
      if (m == NUM_LAYERS - 1) {
        CollectiveComm::barrier();
        
        // 如果有需要，将结果聚合回 rank 0
        if (rank != 0) {
          float tmpResult[SubMaxTokenLength * HiddenSize];
          memcpy(tmpResult, subResultContainer.getData(),
                SubMaxTokenLength * HiddenSize * sizeof(float));
          CollectiveComm::reduce(tmpResult, nullptr, subSize, 0);
        }
      }
    }
  }
}

// ============================================================================
// Rank 0 Processing
// ============================================================================

void processRank0(const std::string &llamaDir, const std::string &llamaBuildDir,
                 int generateLen) {
  const std::string title = "LLaMA 2 Inference Powered by Buddy Compiler";
  std::cout << "\033[33;1m" << title << "\033[0m" << std::endl;

  const std::string vocabDir = llamaDir + "/vocab.txt";

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
  constexpr size_t param_size1 = 131076096;
  const std::string paramsDir0 = llamaBuildDir + "/subgraph0_arg0.data";
  const std::string paramsDir1 = llamaBuildDir + "/subgraph193_arg0.data";

  MemRef<float, 1> paramsContainer0({param_size0});
  loadParameters(paramsDir0, paramsContainer0);

  MemRef<float, 1> paramsContainer1({param_size1});
  loadParameters(paramsDir1, paramsContainer1);

  std::string inputStr;
  getUserInput(inputStr);
  Text<size_t, 2> inputContainer(inputStr);
  tokenizeInput(vocabDir, inputContainer);
  generateLen = MaxTokenLength - inputContainer.getTokenCnt();

  float *inputPtr = nullptr;
  float *outputPtr = tmp3DMemRef.getData();
  int subSize = SubMaxTokenLength * HiddenSize;

  // 主生成循环
  for (int i = 0; i < generateLen; i++) {
    const auto inferenceStart = std::chrono::high_resolution_clock::now();

    // 第一层前向传播
    _mlir_ciface_forward0(inputResultContainerPtr, &paramsContainer0,
                         &inputContainer);
    inputPtr = inputResultContainerPtr->memRef3D0.getData();

    // 广播到所有 worker ranks
    CollectiveComm::broadcast(inputPtr, subSize * (NUM_RANKS - 1), 0);
    CollectiveComm::broadcast(inputResultContainerPtr->memRef2D.getData(),
                             MaxTokenLength * HiddenSize1, 0);
    CollectiveComm::broadcast(inputResultContainerPtr->memRef3D1.getData(),
                             MaxTokenLength * HiddenSize0, 0);
    CollectiveComm::broadcast(inputResultContainerPtr->memRef3D2.getData(),
                             MaxTokenLength * HiddenSize0, 0);

    // 从所有 worker ranks 接收结果
    float allResults[NUM_RANKS * subSize];
    CollectiveComm::allgather(outputPtr, subSize, allResults);

    // 复制回合并结果
    for (int rank = 1; rank < NUM_RANKS; rank++) {
      memcpy(outputPtr + (rank - 1) * subSize,
            allResults + rank * subSize, subSize * sizeof(float));
    }

    // 最后一层前向传播
    _mlir_ciface_forward193(&inputResultContainerPtr->memRef3D0,
                           &paramsContainer1, &tmp3DMemRef);

    const auto inferenceEnd = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double, std::milli> inferenceTime =
        inferenceEnd - inferenceStart;

    // 解码生成的 token
    inputPtr = inputResultContainerPtr->memRef3D0.getData();
    int tokenIndex = inputContainer.getTokenCnt() - 1;
    const float *startPtr = inputPtr + tokenIndex * MaxVocabSize;
    const float *endPtr = startPtr + MaxVocabSize;
    int maxIndex = findMaxIndex(startPtr, endPtr);
    std::string tok = inputContainer.getStr(maxIndex);

    printIterInfo(i, tok, inferenceTime.count() / 1000);

    // 停止条件
    if (maxIndex == 2) {
      break;
    }

    inputContainer.appendTokenIdx(maxIndex);
    outputContainer.appendTokenIdx(maxIndex);
  }
}

// ============================================================================
// Main
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
      processWorkerRankSimplified(rank, generateLen, subSize, llamaBuildDir);
    }
  } catch (const std::exception &e) {
    std::cerr << "Error in rank " << rank << ": " << e.what() << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  MPI_Finalize();
  return 0;
}
