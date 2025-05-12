#ifndef LLAMARMS_H // 作用：防止llamaRMS.h被重复引用
#define LLAMARMS_H
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

/// Declare LLaMA forward function.
extern "C" void _mlir_ciface_forward1(MemRef<float, 3> *, MemRef<float, 1> *,
                                      MemRef<float, 3> *);

/// Print [Log] label in bold blue format.
void printLogLabel() { std::cout << "\033[34;1m[Log] \033[0m"; }

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

  void push_output(const std::any &data) {
    std::lock_guard<std::mutex> lock(outputMutex);
    outputQueue.push(data);
    output_cv.notify_one();
  }

  std::any pop_output() {
    std::unique_lock<std::mutex> lock(outputMutex);
    output_cv.wait(lock, [this] { return !outputQueue.empty(); });
    auto data = outputQueue.front();
    outputQueue.pop();
    return data;
  }

private:
  std::queue<std::any> inputQueue;
  std::queue<std::any> outputQueue;
  std::mutex inputMutex;
  std::mutex outputMutex;
  std::condition_variable input_cv;
  std::condition_variable output_cv;
};

//--------------------- RMSMess (主线程) ---------------------
class RMSMess {
public:
  RMSMess(const std::string name, SharedQueue &queue, const uint16_t &port,
          const std::string &uri)
      : rmsServer(), name(name), inputClient(), sharedQueue(queue),
        hdlsSymbol(),
        resultContainer(MemRef<float, 3>({1, SubMaxTokenLength, HiddenSize})),
        dataId(0) {
    /// 服务器初始化
    rmsServer.set_access_channels(websocketpp::log::alevel::none);
    rmsServer.clear_access_channels(websocketpp::log::alevel::all);
    rmsServer.init_asio();

    // rmsServer.set_close_handler([this](websocketpp::connection_hdl hdl) {
    //   std::lock_guard<std::mutex> lock(symbolMutex); // 加锁保护符号表
    //   auto it = connections.find(hdl);
    //   if (it != connections.end()) {
    //     std::string user_id = it->second;
    //     hdlsSymbol.erase(user_id);
    //     connections.erase(hdl);
    //   }
    // });

    rmsServer.set_message_handler(
        bind(&RMSMess::on_server_message, this, _1, _2));
    rmsServer.listen(port);
    rmsServer.set_reuse_addr(true);
    rmsServer.start_accept();

    /// 客户端初始化
    // 禁用客户端日志
    inputClient.set_access_channels(websocketpp::log::alevel::none);
    inputClient.clear_access_channels(websocketpp::log::alevel::all);
    // 初始化服务器并绑定ioService
    inputClient.init_asio();
    // 设置服务器的消息回调
    inputClient.set_open_handler([this, name](websocketpp::connection_hdl hdl) {
      inputClient.send(hdl, name, websocketpp::frame::opcode::text);
    });
    inputClient.set_message_handler(
        bind(&RMSMess::on_client_message, this, _1, _2));
    // 启动连接
    websocketpp::lib::error_code ec;
    auto con = inputClient.get_connection(uri, ec);
    inputClient.connect(con);
  }

  void run() {
    std::thread client_thread([this]() { inputClient.run(); });
    // 启动 WebSocket 服务器线程
    std::thread server_thread([this]() { rmsServer.run(); });

    // 新增：启动输出监听线程，向MHAMess或MLPMess发送数据
    std::thread output_thread([this]() {
      while (true) {
        resultContainer = std::any_cast<MemRef<float, 3>>(sharedQueue.pop_output());
        std::lock_guard<std::mutex> lock(symbolMutex); // 加锁保护符号表
        if (hdlsSymbol.find("MHAMess0") != hdlsSymbol.end()) {
          send_data(hdlsSymbol["MHAMess0"], dataId++,
                    {resultContainer.getDataVector()});
          send_data(hdlsSymbol["MHAMess1"], dataId++,
                    {resultContainer.getDataVector()});
          std::cout << name << "转发" << "MHAMess" << "成功" << std::endl;
        } else if (hdlsSymbol.find("MLPMess0") != hdlsSymbol.end()) {
          send_data(hdlsSymbol["MLPMess0"], dataId++,
                    {resultContainer.getDataVector()});
          send_data(hdlsSymbol["MLPMess1"], dataId++,
                    {resultContainer.getDataVector()});
          std::cout << name << "转发" << "MLPMess" << "成功" << std::endl;
        } else {
          std::cout << "MHAMess0或MLPMess0未连接, 丢弃结果: " << "result"
                    << std::endl;
        }
      }
    });

    client_thread.join();
    server_thread.join();
    output_thread.join();
  }

private:
  server rmsServer;
  client inputClient;
  const std::string name;
  SharedQueue &sharedQueue;
  std::map<std::string, websocketpp::connection_hdl> hdlsSymbol;
  std::map<websocketpp::connection_hdl, std::string,
           std::owner_less<websocketpp::connection_hdl>>
      connections;
  websocketpp::connection_hdl lastAddHdl;
  std::mutex symbolMutex; // 保护 hdlsSymbol 的互斥锁
  std::mutex sequenceMutex;
  MemRef<float, 3> resultContainer;
  //  确保对dataId的操作是​​原子​​的
  std::atomic<uint32_t> dataId;
  std::mutex dataMutex;
  // 存储每个参数的shape
  std::vector<intptr_t> inputSizes = {{1, SubMaxTokenLength, HiddenSize}};
  // 是否是第一个rms模块
  bool isFirst;

  void send_data(websocketpp::connection_hdl hdl, uint32_t dataId,
                 const std::vector<std::vector<float>> &data) {
    const uint8_t total = data.size();

    if (rmsServer.get_con_from_hdl(hdl)->get_state() !=
        websocketpp::session::state::open)
      return;

    for (uint8_t i = 0; i < total; ++i) {
      const auto &subdata = data[i];

      // 构造协议头
      std::vector<uint8_t> packet(10); // 4+1+1+2=8字节头
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

      rmsServer.send(hdl, packet.data(), packet.size(),
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

  void on_server_message(websocketpp::connection_hdl hdl,
                         server::message_ptr msg) {
    if (msg->get_opcode() == websocketpp::frame::opcode::text) {
      std::string payload = msg->get_payload();
      if (payload.find("AddMess") != std::string::npos) {
        std::lock_guard<std::mutex> lock(symbolMutex); // 加锁保护符号表
        hdlsSymbol["AddMess"] = hdl;
        connections[hdl] = payload;
        std::cout << payload << "已连接" << std::endl;
      } else if (payload.find("MHAMess") != std::string::npos ||
                 payload.find("MLPMess") != std::string::npos) {
        std::lock_guard<std::mutex> lock(symbolMutex); // 加锁保护符号表
        hdlsSymbol[payload] = hdl;
        connections[hdl] = payload;
        std::cout << payload << "已连接" << std::endl;
      } else if (payload.find("LastAdd") != std::string::npos) {
        std::lock_guard<std::mutex> lock(symbolMutex); // 加锁保护符号表
        connections[hdl] = payload;
        std::cout << payload << "已连接" << std::endl;
      }
      return;
    }

    if (msg->get_opcode() == websocketpp::frame::opcode::binary) {
      auto chunk = getFloatData(msg);
      intptr_t sizes[3] = {1, SubMaxTokenLength, HiddenSize};
      MemRef<float, 3> subResultContainer(chunk.data(), sizes);

      std::cout << "接收到AddMess数据" << std::endl;
      {
        std::lock_guard<std::mutex> lockMutex(symbolMutex); // 加锁保护符号表
        auto it = hdlsSymbol.find("AddMess");
        if (it != hdlsSymbol.end()) {
          send_data(hdlsSymbol["AddMess"], dataId++,
                    {subResultContainer.getDataVector()});
          std::cout << name << "转发AddMess成功." << std::endl;
        } else {
          std::cout << "AddMess未连接, 丢弃结果." << std::endl;
        }
      }
      sharedQueue.push_input(subResultContainer);
    }
  }

  void on_client_message(websocketpp::connection_hdl hdl,
                         client::message_ptr msg) {
    auto chunk = getFloatData(msg);
    intptr_t sizes[3] = {1, SubMaxTokenLength, HiddenSize};
    MemRef<float, 3> subResultContainer(chunk.data(), sizes);

    std::cout << "接收到InputMess数据" << std::endl;
    {
      std::lock_guard<std::mutex> lockMutex(symbolMutex); // 加锁保护符号表
      auto it = hdlsSymbol.find("AddMess");
      if (it != hdlsSymbol.end()) {
        send_data(hdlsSymbol["AddMess"], dataId++,
                  {subResultContainer.getDataVector()});
        std::cout << name << "转发AddMess成功." << std::endl;
      } else {
        std::cout << "AddMess未连接, 丢弃结果." << std::endl;
      }
    }
    sharedQueue.push_input(subResultContainer);
  }
};

//----------------------Comp (子线程)----------------------------------
//  -rmsNum: 标志当前RMS模块是第几组RMS模块（1标志前一组模块，0标志后一组模块）
//---------------------------------------------------------------------
class Comp {
public:
  Comp(SharedQueue &queue, const int rmsNum) : sharedQueue(queue) {
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
    /// Define directories of vacabulary and parameter file.
    std::string llamaBuildDir = LLAMA_EXAMPLE_BUILD_PATH;

    for (int i = 0; i < 194; i++) { // N 为需要生成的数量
      if (paramSize_group[i] == 4096 && i % 2 == rmsNum) {
        std::string paramsDir =
            llamaBuildDir + "/subgraph" + std::to_string(i) + "_arg0.data";
        MemRef<float, 1> paramsContainer({paramSize_group[i]});
        loadParameters(paramsDir, paramsContainer);
        paramsContainers.push_back(std::move(paramsContainer));
      }
    }
  }

  void run() {
    while (true) {
      auto input = std::any_cast<MemRef<float, 3>>(sharedQueue.pop_input());
      MemRef<float, 3> resultContainer({1, SubMaxTokenLength, HiddenSize});
      _mlir_ciface_forward1(&resultContainer, &paramsContainers[index], &input);
      std::cout << "第" << index << "次forward1 computed." << std::endl;
      sharedQueue.push_output(resultContainer);
      index = (index + 1) % 32;
    }
  }

private:
  SharedQueue &sharedQueue;
  std::vector<MemRef<float, 1>> paramsContainers;
  uint32_t index = 0;
};

#endif // LLAMARMS_H
