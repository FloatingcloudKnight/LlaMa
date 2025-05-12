#ifndef LLAMAOUTPUT_H // 作用：防止llamaOutput.h被重复引用
#define LLAMAOUTPUT_H
#include <any>
#include <boost/asio.hpp>
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
#include <queue>
#include <sstream>
#include <string>
#include <thread>
#include <variant>
#include <vector>
#include <websocketpp/client.hpp>
#include <websocketpp/config/asio_no_tls.hpp>
#include <websocketpp/server.hpp>

using namespace buddy;
using websocketpp::lib::bind;
using websocketpp::lib::placeholders::_1;
using websocketpp::lib::placeholders::_2;

typedef websocketpp::server<websocketpp::config::asio> server;
typedef websocketpp::client<websocketpp::config::asio> client;

constexpr size_t MaxVocabSize = 32000;
constexpr size_t MaxTokenLength = 40;
constexpr size_t SubMaxTokenLength = 20;
constexpr size_t HiddenSize = 4096;
constexpr size_t HiddenSize0 = 128;
constexpr size_t HiddenSize1 = 41;
constexpr size_t ParamSize = 131076096;

/// Print [Log] label in bold blue format.
void printLogLabel() { std::cout << "\033[34;1m[Log] \033[0m"; }

/// Declare LLaMA forward function.
extern "C" void _mlir_ciface_forward193(MemRef<float, 3> *, MemRef<float, 1> *,
                                        MemRef<float, 3> *);

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

// 共享内存结构（线程安全队列）
class SharedQueue {
public:
  void push_input(const MemRef<float, 3> &data) {
    std::lock_guard<std::mutex> lock(input_mutex);
    inputQueue.push(data);
    input_cv.notify_one();
  }

  MemRef<float, 3> pop_input() {
    std::unique_lock<std::mutex> lock(input_mutex);
    input_cv.wait(lock, [this] { return !inputQueue.empty(); });
    auto data = inputQueue.front();
    inputQueue.pop();
    return data;
  }

  void push_input0(const MemRef<float, 3> &data) {
    std::lock_guard<std::mutex> lock(input0_mutex);
    input0Queue.push(data);
    input0_cv.notify_one();
  }

  MemRef<float, 3> pop_input0() {
    std::unique_lock<std::mutex> lock(input0_mutex);
    input0_cv.wait(lock, [this] { return !input0Queue.empty(); });
    auto data = input0Queue.front();
    input0Queue.pop();
    return data;
  }

  void push_output(const MemRef<float, 3> &data) {
    std::lock_guard<std::mutex> lock(output_mutex);
    outputQueue.push(data);
    output_cv.notify_one();
  }

  MemRef<float, 3> pop_output() {
    std::unique_lock<std::mutex> lock(output_mutex);
    output_cv.wait(lock, [this] { return !outputQueue.empty(); });
    auto data = outputQueue.front();
    outputQueue.pop();
    return data;
  }

private:
  std::queue<MemRef<float, 3>> inputQueue;
  std::queue<MemRef<float, 3>> input0Queue;
  std::queue<MemRef<float, 3>> outputQueue;
  std::mutex input_mutex;
  std::mutex input0_mutex;
  std::mutex output_mutex;
  std::condition_variable input_cv;
  std::condition_variable input0_cv;
  std::condition_variable output_cv;
};

//--------------------- OutputMess (主线程) ---------------------
class OutputMess {
public:
  OutputMess(SharedQueue &queue,
             const std::string &uri, const std::string &uri0,
             const std::string &uri1)
      : addClient(), addClient0(), inputClient(), sharedQueue(queue),
        outputContainer(), inputHdl(),
        currentToken(),
        resultContainer(MemRef<float, 3>({1, MaxTokenLength, HiddenSize})) {
    /// 客户端初始化
    addClient.set_access_channels(websocketpp::log::alevel::none);
    addClient.clear_access_channels(websocketpp::log::alevel::all);
    addClient.init_asio();
    addClient.set_open_handler([this](websocketpp::connection_hdl hdl) {
      addClient.send(hdl, "OutputMess", websocketpp::frame::opcode::text);
    });
    addClient.set_message_handler(
        bind(&OutputMess::on_addClient_message, this, _1, _2));
    websocketpp::lib::error_code addec;
    auto addcon = addClient.get_connection(uri, addec);
    addClient.connect(addcon);

    addClient0.set_access_channels(websocketpp::log::alevel::none);
    addClient0.clear_access_channels(websocketpp::log::alevel::all);
    addClient0.init_asio();
    addClient0.set_open_handler([this](websocketpp::connection_hdl hdl) {
      addClient0.send(hdl, "OutputMess", websocketpp::frame::opcode::text);
    });
    addClient0.set_message_handler(
        bind(&OutputMess::on_addClient0_message, this, _1, _2));
    websocketpp::lib::error_code addec0;
    auto addcon0 = addClient0.get_connection(uri0, addec0);
    addClient0.connect(addcon0);

    inputClient.set_access_channels(websocketpp::log::alevel::none);
    inputClient.clear_access_channels(websocketpp::log::alevel::all);
    inputClient.init_asio();
    inputClient.set_open_handler([this](websocketpp::connection_hdl hdl) {
      inputClient.send(hdl, "OutputMess", websocketpp::frame::opcode::text);
      std::lock_guard<std::mutex> lock(hdlMutex); // 加锁保护符号表
      inputHdl = hdl;
    });
    inputClient.set_message_handler(
        bind(&OutputMess::on_inputClient_message, this, _1, _2));
    websocketpp::lib::error_code inputec;
    auto inputcon = inputClient.get_connection(uri1, inputec);
    inputClient.connect(inputcon);

    outputContainer.loadVocab(vocabDir);
  }

  void run() {
    std::thread addClient_thread([this]() { addClient.run(); });
    std::thread addClient0_thread([this]() { addClient0.run(); });
    std::thread inputClient_thread([this]() { inputClient.run(); });

    // 新增：启动输出监听线程，向InputMess发送数据
    std::thread output_thread([this]() {
      while (true) {
        resultContainer = sharedQueue.pop_output();
        std::lock_guard<std::mutex> lock(hdlMutex); // 加锁保护符号表
        int tokenIndex = currentToken + tokenCnt - 1;
        currentToken++;
        // Determine the generated token.
        const float *startPtr =
            resultContainer.getData() + tokenIndex * MaxVocabSize;
        const float *endPtr = startPtr + MaxVocabSize;
        int maxIndex = findMaxIndex(startPtr, endPtr);

        // Stop if a separator token (2, </s>) or line break token (13 <0x0A>)
        // is generated.
        if (maxIndex == 2) {
          break;
        }
        // Append the generated token into the input and output container.
        outputContainer.appendTokenIdx(maxIndex);

        if (currentToken ==  (MaxTokenLength - tokenCnt)) {
          std::cout << "\033[33;1m[Output]\033[0m " << outputContainer.revertLlama()
                    << std::endl;
          currentToken = 0;
        } else if (currentToken < (MaxTokenLength - tokenCnt)) {
          inputClient.send(inputHdl, std::to_string(maxIndex),
                       websocketpp::frame::opcode::text);
          std::cout << "第" << currentToken << "次Token推理完成." << std::endl;
        } else {
          std::cout << "Transformer层计算次数过多, 当前次数为: " << currentToken
                    << std::endl;
        }
      }
    });
    addClient_thread.join();
    addClient0_thread.join();
    output_thread.join();
  }


private:
  client addClient;
  client addClient0;
  client inputClient;
  SharedQueue &sharedQueue;
  websocketpp::connection_hdl inputHdl;
  std::mutex hdlMutex; // 保护 hdlsSymbol 的互斥锁
  MemRef<float, 3> resultContainer;
  Text<size_t, 2> outputContainer;
  // //  确保对dataId的操作是​​原子​​的
  // std::atomic<uint32_t> dataId = 0;
  std::mutex data_mutex;
  // 记录已经进行过的llama推理次数
  uint32_t currentToken = 0;
  // 记录输入的token数量
  uint32_t tokenCnt = 0;
  /// Define directories of vacabulary and file.
  std::string llamaDir = LLAMA_SPLIT_EXAMPLE_PATH;
  const std::string vocabDir = llamaDir + "/vocab.txt";

  // void send_data(websocketpp::connection_hdl hdl, uint32_t dataId,
  //                const std::vector<std::vector<float>> &data) {
  //   const uint8_t total = data.size();

  //   for (uint8_t i = 0; i < total; ++i) {
  //     const auto &subdata = data[i];

  //     // 构造协议头
  //     std::vector<uint8_t> packet(10); // 4+1+1+2=10字节头
  //     memcpy(packet.data(), &dataId, 4);
  //     packet[4] = total;
  //     packet[5] = i;
  //     uint32_t num = subdata.size();
  //     memcpy(packet.data() + 6, &num, 4);

  //     // 添加浮点数据
  //     const uint8_t *binaryData =
  //         reinterpret_cast<const uint8_t *>(subdata.data());
  //     packet.insert(packet.end(), binaryData,
  //                   binaryData + subdata.size() * sizeof(float));

  //     inputClient.send(hdl, packet.data(), packet.size(),
  //                      websocketpp::frame::opcode::binary);
  //   }
  // }

  std::vector<float> getFloatData(client::message_ptr msg) {
    if (msg->get_opcode() != websocketpp::frame::opcode::binary) {
      std::cout << "忽略非二进制消息" << std::endl;
      return {};
    }

    const std::string &payload = msg->get_payload();
    if (payload.size() < 10) {
      std::cerr << "错误: 协议头不完整(需要至少10字节)" << std::endl;
      return {};
    }

    // 解析协议头
    uint32_t batch_id;
    uint8_t totalChunks, seqChunk;
    uint32_t num_elements;

    memcpy(&batch_id, payload.data(), 4);
    totalChunks = payload[4];
    seqChunk = payload[5];
    memcpy(&num_elements, payload.data() + 6, 4);

    // 验证分块序号有效性
    if (seqChunk >= totalChunks) {
      std::cerr << "错误：非法分块序号 " << (int)seqChunk
                << " (总块数=" << (int)totalChunks << ")" << std::endl;
      return {};
    }

    // 验证数据长度
    const size_t expectedSize = 10 + num_elements * sizeof(float);
    if (payload.size() != expectedSize) {
      std::cerr << "错误：数据长度不匹配(预期=" << expectedSize
                << " 实际=" << payload.size() << ")" << std::endl;
      return {};
    }

    // 提取浮点数据
    const float *float_data =
        reinterpret_cast<const float *>(payload.data() + 10);
    std::vector<float> chunk(float_data, float_data + num_elements);
    return chunk;
  }

  void on_addClient_message(websocketpp::connection_hdl hdl,
                            client::message_ptr msg) {
      std::lock_guard<std::mutex> lock(data_mutex);
      auto chunk = getFloatData(msg);
      intptr_t sizes[3] = {1, SubMaxTokenLength, HiddenSize};
      MemRef<float, 3> subResultContainer(chunk.data(), sizes);
      sharedQueue.push_input(subResultContainer);
      std::cout << "接收到AddMess0数据" << std::endl;
  }

  void on_addClient0_message(websocketpp::connection_hdl hdl,
                             client::message_ptr msg) {
      std::lock_guard<std::mutex> lock(data_mutex);
      auto chunk = getFloatData(msg);
      intptr_t sizes[3] = {1, SubMaxTokenLength, HiddenSize};
      MemRef<float, 3> subResultContainer(chunk.data(), sizes);
      sharedQueue.push_input0(subResultContainer);
      std::cout << "接收到AddMess1数据" << std::endl;
  }

  void on_inputClient_message(websocketpp::connection_hdl hdl,
                              client::message_ptr msg) {
    std::string payload = msg->get_payload();
    try {
      if (payload.empty())
        throw std::invalid_argument("空输入");
        tokenCnt = std::stoi(payload);
    } catch (const std::exception &e) {
      std::cout << "无效输入: " << payload << " (" << e.what() << ")"
                << std::endl;
    }
  }
};

//--------------------- Comp (子线程) ---------------------
class Comp {
public:
  Comp(SharedQueue &queue)
      : sharedQueue(queue),
        paramsContainer({ParamSize}) {
          /// Define directories of parameter file.
          std::string llamaBuildDir = LLAMA_EXAMPLE_BUILD_PATH;
          std::string paramsDir =
              llamaBuildDir + "/subgraph193_arg0.data"; // 权重文件路径
          loadParameters(paramsDir, paramsContainer);
        }

  void run() {
    while (true) {
      MemRef<float, 3> addInput = sharedQueue.pop_input();
      MemRef<float, 3> addInput0 = sharedQueue.pop_input0();
      MemRef<float, 3> input0({1, MaxTokenLength, HiddenSize});
      MemRef<float, 3> resultContainer({1, MaxTokenLength, HiddenSize});
      input0.concatenateMemRefs(addInput, addInput0, input0,
                                         1);
      _mlir_ciface_forward193(&resultContainer, &paramsContainer,
                              &input0);
      std::cout << "forward193 computed." << std::endl;
      sharedQueue.push_output(resultContainer);
    }
  }

private:
  SharedQueue &sharedQueue;
  MemRef<float, 1> paramsContainer;
};

#endif // LLAMAOUTPUT_H
