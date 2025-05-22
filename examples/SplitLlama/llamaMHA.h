#ifndef LLAMAMHA_H // 作用：防止llamaMHA.h被重复引用
#define LLAMAMHA_H
#include <any>
#include <boost/asio.hpp>
#include <buddy/Core/Container.h>
#include <buddy/LLM/TextContainer.h>
#include "BaseDisModel.h"
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

/// Declare LLaMA forward function.
extern "C" void _mlir_ciface_forward2(MemRef<float, 2> *, MemRef<float, 1> *,
                                      MemRef<float, 3> *, MemRef<float, 3> *,
                                      MemRef<float, 3> *, MemRef<float, 2> *);

// 共享内存结构（线程安全队列）
class SharedQueue {
public:
  void push_input(const std::any &data) {
    std::lock_guard<std::mutex> lock(inputMutex);
    inputQueue.push(data);
    input_cv.notify_one();
  }

  std::any pop_input() {
    std::unique_lock<std::mutex> lock(inputMutex);
    input_cv.wait(lock, [this] { return !inputQueue.empty(); });
    auto data = inputQueue.front();
    inputQueue.pop();
    return data;
  }

  void push_input0(const MemRef<float, 3> &data) {
    std::lock_guard<std::mutex> lock(input0Mutex);
    input0Queue.push(data);
    input0_cv.notify_one();
  }

  MemRef<float, 3> pop_input0() {
    std::unique_lock<std::mutex> lock(input0Mutex);
    input0_cv.wait(lock, [this] { return !input0Queue.empty(); });
    auto data = input0Queue.front();
    input0Queue.pop();
    return data;
  }

  void push_input1(const MemRef<float, 3> &data) {
    std::lock_guard<std::mutex> lock(input1Mutex);
    input1Queue.push(data);
    input1_cv.notify_one();
  }

  MemRef<float, 3> pop_input1() {
    std::unique_lock<std::mutex> lock(input1Mutex);
    input1_cv.wait(lock, [this] { return !input1Queue.empty(); });
    auto data = input1Queue.front();
    input1Queue.pop();
    return data;
  }

  void push_output(const MemRef<float, 2> &data) {
    std::lock_guard<std::mutex> lock(outputMutex);
    outputQueue.push(data);
    output_cv.notify_one();
  }

  MemRef<float, 2> pop_output() {
    std::unique_lock<std::mutex> lock(outputMutex);
    output_cv.wait(lock, [this] { return !outputQueue.empty(); });
    auto data = outputQueue.front();
    outputQueue.pop();
    return data;
  }

  u_int32_t inputQueueSize() {
    std::lock_guard<std::mutex> lock(inputMutex);
    return inputQueue.size();
  }

private:
  std::queue<std::any> inputQueue;
  std::queue<MemRef<float, 3>> input0Queue;
  std::queue<MemRef<float, 3>> input1Queue;
  std::queue<MemRef<float, 2>> outputQueue;
  std::mutex inputMutex;
  std::mutex input0Mutex;
  std::mutex input1Mutex;
  std::mutex outputMutex;
  std::condition_variable input_cv;
  std::condition_variable input0_cv;
  std::condition_variable input1_cv;
  std::condition_variable output_cv;
};

//--------------------- MHAMess (主线程) ---------------------
class MHAMess {
public:
  MHAMess(const std::string name, SharedQueue &queue, const uint16_t &port,
          const std::string &uri0, const std::string &uri1,
          const std::string &uri2)
      : mhaServer(), name(name), sharedQueue(queue), hdlsSymbol(),
        resultContainer(MemRef<float, 2>({MaxTokenLength, HiddenSize})),
        dataId(0) {
    /// 服务器初始化
    mhaServer.set_access_channels(websocketpp::log::alevel::none);
    mhaServer.clear_access_channels(websocketpp::log::alevel::all);
    mhaServer.init_asio();

    // mhaServer.set_close_handler([this](websocketpp::connection_hdl hdl) {
    //   std::lock_guard<std::mutex> lock(symbolMutex); // 加锁保护符号表
    //   auto it = connections.find(hdl);
    //   if (it != connections.end()) {
    //     std::string user_id = it->second;
    //     hdlsSymbol.erase(user_id);
    //     connections.erase(hdl);
    //   }
    // });

    mhaServer.set_message_handler(
        bind(&MHAMess::on_server_message, this, _1, _2));
    mhaServer.listen(port);
    mhaServer.set_reuse_addr(true);
    mhaServer.start_accept();

    /// 客户端初始化
    // 禁用客户端日志
    rmsClient.set_access_channels(websocketpp::log::alevel::none);
    rmsClient.clear_access_channels(websocketpp::log::alevel::all);
    // 初始化服务器并绑定ioService
    rmsClient.init_asio();
    // 设置服务器的消息回调
    rmsClient.set_open_handler([this, name](websocketpp::connection_hdl hdl) {
      rmsClient.send(hdl, name, websocketpp::frame::opcode::text);
    });
    rmsClient.set_message_handler(
        bind(&MHAMess::on_rmsClient_message, this, _1, _2));
    // 启动连接
    websocketpp::lib::error_code rmsec;
    auto rmscon = rmsClient.get_connection(uri1, rmsec);
    rmsClient.connect(rmscon);

    // 禁用客户端日志
    rmsClient0.set_access_channels(websocketpp::log::alevel::none);
    rmsClient0.clear_access_channels(websocketpp::log::alevel::all);
    // 初始化服务器并绑定ioService
    rmsClient0.init_asio();
    // 设置服务器的消息回调
    rmsClient0.set_open_handler([this, name](websocketpp::connection_hdl hdl) {
      rmsClient0.send(hdl, name, websocketpp::frame::opcode::text);
    });
    rmsClient0.set_message_handler(
        bind(&MHAMess::on_rmsClient0_message, this, _1, _2));
    // 启动连接
    websocketpp::lib::error_code rmsec0;
    auto rmscon0 = rmsClient0.get_connection(uri2, rmsec0);
    rmsClient0.connect(rmscon0);

    // 禁用客户端日志
    inputClient.set_access_channels(websocketpp::log::alevel::none);
    inputClient.clear_access_channels(websocketpp::log::alevel::all);
    // 初始化服务器并绑定ioService
    inputClient.init_asio();
    // 设置服务器的消息回调
    inputClient.set_open_handler([this, name](websocketpp::connection_hdl hdl) {
      rmsClient.send(hdl, name, websocketpp::frame::opcode::text);
    });
    inputClient.set_message_handler(
        bind(&MHAMess::on_inputClient_message, this, _1, _2));
    // 启动连接
    websocketpp::lib::error_code inputec;
    auto inputcon = inputClient.get_connection(uri0, inputec);
    inputClient.connect(inputcon);
  }

  void run() {
    std::thread inputClient_thread([this]() { inputClient.run(); });
    std::thread rmsClient_thread([this]() { rmsClient.run(); });
    std::thread rmsClient0_thread([this]() { rmsClient0.run(); });
    // 启动 WebSocket 服务器线程
    std::thread server_thread([this]() { mhaServer.run(); });

    // 新增：启动输出监听线程，向AddMess发送数据
    std::thread output_thread([this]() {
      while (true) {
        resultContainer = sharedQueue.pop_output();
        std::lock_guard<std::mutex> lock(symbolMutex); // 加锁保护符号表
        MemRef<float, 2> subResultContainer0({SubMaxTokenLength, HiddenSize});
        MemRef<float, 2> subResultContainer1({SubMaxTokenLength, HiddenSize});
        resultContainer.splitMemRef(std::move(resultContainer),
                                    subResultContainer0, subResultContainer1, 0,
                                    20);
        auto it = hdlsSymbol.find("AddMess0");
        if (it != hdlsSymbol.end()) {
          send_data(hdlsSymbol["AddMess0"], dataId++,
                    {subResultContainer0.getDataVector()});
          send_data(hdlsSymbol["AddMess1"], dataId++,
                    {subResultContainer1.getDataVector()});
          std::cout << "转发成功" << std::endl;
        } else {
          std::cout << "AddMess0未连接, 丢弃结果: " << "result" << std::endl;
        }
      }
    });
    inputClient_thread.join();
    rmsClient_thread.join();
    rmsClient0_thread.join();
    server_thread.join();
    output_thread.join();
  }

private:
  server mhaServer;
  client inputClient;
  client rmsClient;
  client rmsClient0;
  const std::string name;
  SharedQueue &sharedQueue;
  std::map<std::string, websocketpp::connection_hdl> hdlsSymbol;
  std::map<websocketpp::connection_hdl, std::string,
           std::owner_less<websocketpp::connection_hdl>>
      connections;
  std::mutex symbolMutex; // 保护 hdlsSymbol 的互斥锁
  MemRef<float, 2> resultContainer;
  //  确保对dataId的操作是​​原子​​的
  std::atomic<uint32_t> dataId;
  std::mutex dataMutex;
  std::mutex sequenceMutex;
  // 存储每个参数的shape
  std::vector<std::vector<size_t>> inputSizes = {
      {MaxTokenLength, HiddenSize1},
      {1, MaxTokenLength, HiddenSize0},
      {1, MaxTokenLength, HiddenSize0}};
  // 表示最近从其他服务器得到的数据块在数据组内的序号
  uint8_t currentSequence;

  void send_data(websocketpp::connection_hdl hdl, uint32_t dataId,
                 const std::vector<std::vector<float>> &data) {
    const uint8_t total = data.size();

    if (mhaServer.get_con_from_hdl(hdl)->get_state() !=
        websocketpp::session::state::open)
      return;

    for (uint8_t i = 0; i < total; ++i) {
      const auto &subdata = data[i];

      // 构造协议头
      std::vector<uint8_t> packet(10); // 4+1+1+4=10字节头
      memcpy(packet.data(), &dataId, 4);
      packet[4] = total;
      packet[5] = i;
      uint32_t num = subdata.size();
      memcpy(packet.data() + 6, &num, 4);

      // 添加浮点数据
      const uint8_t *binaryData =
          reinterpret_cast<const uint8_t *>(subdata.data());
      packet.insert(packet.end(), binaryData,
                    binaryData + subdata.size() * sizeof(float));

      mhaServer.send(hdl, packet.data(), packet.size(),
                     websocketpp::frame::opcode::binary);
    }
  }

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
    {
      std::lock_guard<std::mutex> lock(sequenceMutex);
      currentSequence = payload[5];
    }
    memcpy(&num_elements, payload.data() + 6, 4);

    // 验证分块序号有效性
    if (currentSequence >= totalChunks) {
      std::cerr << "错误：非法分块序号 " << (int)currentSequence
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

  void on_server_message(websocketpp::connection_hdl hdl,
                         server::message_ptr msg) {
    std::string payload = msg->get_payload();
    if (payload.find("AddMess") != std::string::npos) {
      std::lock_guard<std::mutex> lock(symbolMutex); // 加锁保护符号表
      hdlsSymbol[payload] = hdl;
      connections[hdl] = payload;
      std::cout << payload << " 已连接" << std::endl;
      return;
    }
  }

  void on_rmsClient_message(websocketpp::connection_hdl,
                            client::message_ptr msg) {
    std::lock_guard<std::mutex> lock(dataMutex);
    auto chunk = getFloatData(msg);
    intptr_t sizes[3] = {1, SubMaxTokenLength, HiddenSize};
    MemRef<float, 3> subResultContainer(chunk.data(), sizes);
    sharedQueue.push_input0(subResultContainer);
    std::cout << "接收到RMSMess数据" << std::endl;
  }

  void on_rmsClient0_message(websocketpp::connection_hdl,
                             client::message_ptr msg) {
    std::lock_guard<std::mutex> lock(dataMutex);
    auto chunk = getFloatData(msg);
    intptr_t sizes[3] = {1, SubMaxTokenLength, HiddenSize};
    MemRef<float, 3> subResultContainer(chunk.data(), sizes);
    sharedQueue.push_input1(subResultContainer);
    std::cout << "接收到RMSMess数据" << std::endl;
  }

  void on_inputClient_message(websocketpp::connection_hdl,
                              client::message_ptr msg) {
    std::lock_guard<std::mutex> lock(dataMutex);
    auto chunk = getFloatData(msg);
    int sequence = (int)currentSequence;

    // 构造 MemRef 并推入队列
    if (sequence == 0) {
      intptr_t sizes[2] = {MaxTokenLength, HiddenSize1};
      MemRef<float, 2> subResultContainer(chunk.data(), sizes);
      sharedQueue.push_input(subResultContainer);
    } else {
      MemRef<float, 3> subResultContainer(chunk.data(), inputSizes[sequence]);
      sharedQueue.push_input(subResultContainer);
    }
    std::cout << "接收到InputMess数据" << std::endl;
  }
};

//--------------------- Comp (子线程) ---------------------
class Comp {
public:
  Comp(SharedQueue &queue, const std::string splitNum = "0")
      : sharedQueue(queue), splitNum(splitNum),
        currentInput1(MemRef<float, 2>({MaxTokenLength, HiddenSize1})),
        currentInput2(MemRef<float, 3>({1, MaxTokenLength, HiddenSize0})),
        currentInput3(MemRef<float, 3>({1, MaxTokenLength, HiddenSize0})) {}

  void init() { loadAllParameters(); }

  void run() {
    while (true) {
      // 非阻塞检查是否有新输入
      if (!index) {
        updateParams(); // 原子更新三个参数
      }
      std::lock_guard<std::mutex> lock(inputMutex);
      MemRef<float, 3> rmsInput0 = sharedQueue.pop_input0();
      MemRef<float, 3> rmsInput1 = sharedQueue.pop_input1();
      MemRef<float, 3> input0({1, MaxTokenLength, HiddenSize});
      input0.concatenateMemRefs(rmsInput0, rmsInput1, input0, 1);
      MemRef<float, 2> resultContainer({MaxTokenLength, HiddenSize}); 
      _mlir_ciface_forward2(&resultContainer, &paramsContainers[index], &input0,
                            &currentInput2, &currentInput3, &currentInput1);
      std::cout << "第" << index << "次forward2 computed." << std::endl;
      sharedQueue.push_output(resultContainer);
      index = (index + 1) % 32;
    }
  }

private:
  SharedQueue &sharedQueue;
  std::vector<MemRef<float, 1>> paramsContainers;
  uint32_t index = 0;
  const std::string splitNum;
  MemRef<float, 2> currentInput1;
  MemRef<float, 3> currentInput2;
  MemRef<float, 3> currentInput3;
  std::mutex inputMutex; // 保护参数更新

  void loadAllParameters() {
    constexpr size_t paramSize_group[] = {
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
    size_t group_len = sizeof(paramSize_group) / sizeof(paramSize_group[0]);
    BaseDisModel::getParameters(paramSize_group, group_len, 33554432,
                                 splitNum, paramsContainers);
  }

  void updateParams() {
    std::lock_guard<std::mutex> lock(inputMutex);
    currentInput1 = std::any_cast<MemRef<float, 2>>(sharedQueue.pop_input());
    currentInput2 = std::any_cast<MemRef<float, 3>>(sharedQueue.pop_input());
    currentInput3 = std::any_cast<MemRef<float, 3>>(sharedQueue.pop_input());
    std::cout << "额外参数已更新" << std::endl;
  }
};

#endif // LLAMAMHA_H
