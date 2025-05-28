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
#include "SharedQueueTemp.h"

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
constexpr size_t ParamSize = 131072064;

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

/// Declare LLaMA forward function.
extern "C" void _mlir_ciface_forward0(MemRefContainer *, MemRef<float, 1> *,
                                      Text<size_t, 2> *);

/// Capture input message.
void getUserInput(std::string &inputStr) {
  std::cout << "\nPlease send a message:" << std::endl;
  std::cout << ">>> ";
  getline(std::cin, inputStr);
  std::cout << std::endl;
}

/// Print [Log] label in bold blue format.
void printLogLabel() { std::cout << "\033[34;1m[Log] \033[0m"; }


//--------------------- InputMess (主线程) ---------------------
class InputQueue : public SharedQueueTemp {
public:
  InputQueue() : SharedQueueTemp({"input", "output"}) {}
};


class InputMess {
public:
  InputMess(InputQueue &queue, MemRefContainer *resultContainerPtr)
      : inputServer(), shared_queue(queue), hdlsSymbol(), inputContainer(),
        resultContainerPtr(resultContainerPtr),
        memRef3D0(MemRef<float, 3>({1, MaxTokenLength, HiddenSize})),
        memRef2D(MemRef<float, 2>({MaxTokenLength, HiddenSize1})),
        memRef3D1(MemRef<float, 3>({1, MaxTokenLength, HiddenSize0})),
        memRef3D2(MemRef<float, 3>({1, MaxTokenLength, HiddenSize0})),
        subResultContainer0(
            MemRef<float, 3>({1, SubMaxTokenLength, HiddenSize})),
        subResultContainer1(
            MemRef<float, 3>({1, SubMaxTokenLength, HiddenSize})),
        dataId(0) {
    inputServer.set_access_channels(websocketpp::log::alevel::none);
    inputServer.clear_access_channels(websocketpp::log::alevel::all);
    inputServer.init_asio();

    inputServer.set_message_handler(
        bind(&InputMess::on_server_message, this, _1, _2));
    inputServer.set_reuse_addr(true);
    inputServer.listen(9001);
    inputServer.start_accept();
  }

  void run() {
    // 启动 WebSocket 服务器线程
    std::thread server_thread([this]() { inputServer.run(); });

    // 新增：启动输出监听线程，向RMSMess发送数据
    std::thread output_thread([this]() {
      while (true) {
        resultContainerPtr = shared_queue.pop<MemRefContainer*>("output");
        memRef3D0 = resultContainerPtr->memRef3D0;
        memRef2D = resultContainerPtr->memRef2D;
        memRef3D1 = resultContainerPtr->memRef3D1;
        memRef3D2 = resultContainerPtr->memRef3D2;
        memRef3D0.splitMemRef(std::move(memRef3D0), subResultContainer0,
                              subResultContainer1, 1, 20);

        std::lock_guard<std::mutex> lock(symbolMutex); // 加锁保护符号表
        auto it = hdlsSymbol.find("RMSMess0");
        if (it != hdlsSymbol.end()) {
          try {
            send_data(hdlsSymbol["RMSMess0"], dataId++,
                      {subResultContainer0.getDataVector()});
            send_data(hdlsSymbol["RMSMess1"], dataId++,
                      {subResultContainer1.getDataVector()});
            std::cout << "成功向RMSMess发送数据" << std::endl;
            send_data(hdlsSymbol["MHAMess0"], dataId++,
                      {memRef2D.getDataVector(), memRef3D1.getDataVector(),
                       memRef3D2.getDataVector()});
            send_data(hdlsSymbol["MHAMess1"], dataId++,
                      {memRef2D.getDataVector(), memRef3D1.getDataVector(),
                       memRef3D2.getDataVector()});
            std::cout << "成功向MHAMess发送数据" << std::endl;
          } catch (const websocketpp::exception &e) {
            std::cout << "转发失败: " << e.what() << std::endl;
          }
        } else {
          std::cout << "RMSMess未连接, 丢弃结果: " << "result" << std::endl;
        }
      }
    });

    server_thread.join();
    output_thread.join();
  }

private:
  server inputServer;
  InputQueue &shared_queue;
  std::map<std::string, websocketpp::connection_hdl> hdlsSymbol;
  std::map<websocketpp::connection_hdl, std::string,
           std::owner_less<websocketpp::connection_hdl>>
      connections;
  std::mutex symbolMutex; // 保护 hdlsSymbol 的互斥锁
  std::mutex hdlMutex;
  Text<size_t, 2> inputContainer;
  MemRef<float, 3> memRef3D0;
  MemRef<float, 2> memRef2D;
  MemRef<float, 3> memRef3D1;
  MemRef<float, 3> memRef3D2;
  MemRefContainer *resultContainerPtr;
  MemRef<float, 3> subResultContainer0;
  MemRef<float, 3> subResultContainer1;

  /// Define directories of vacabulary and file.
  std::string llamaDir = LLAMA_SPLIT_EXAMPLE_PATH;
  const std::string vocabDir = llamaDir + "/vocab.txt";

  //  确保对dataId的操作是​​原子​​的
  std::atomic<uint32_t> dataId;

  void send_data(websocketpp::connection_hdl hdl, uint32_t dataId,
                 const std::vector<std::vector<float>> &data) {
    const uint8_t total = data.size();

    if (inputServer.get_con_from_hdl(hdl)->get_state() !=
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

      inputServer.send(hdl, packet.data(), packet.size(),
                       websocketpp::frame::opcode::binary);
    }
  }

  void on_server_message(websocketpp::connection_hdl hdl,
                         server::message_ptr msg) {
    std::string payload = msg->get_payload();
    if (payload.find("RMSMess") != std::string::npos ||
        payload.find("MHAMess") != std::string::npos) {
      std::lock_guard<std::mutex> lock(symbolMutex); // 加锁保护符号表
      hdlsSymbol[payload] = hdl;
      connections[hdl] = payload;
      std::cout << payload << " 已连接" << std::endl;
    } else if (payload == "OutputMess") {
      std::lock_guard<std::mutex> lock(symbolMutex); // 加锁保护符号表
      hdlsSymbol[payload] = hdl;
      connections[hdl] = payload;
      std::cout << payload << " 已连接" << std::endl;

      // 获取用户输入
      std::string inputStr;
      getUserInput(inputStr);
      // 创建并tokenize输入容器
      inputContainer = Text<size_t, 2>(inputStr);
      BaseDisModel::tokenizeInput(vocabDir, inputContainer, MaxTokenLength);
      // 将输入压入队列
      shared_queue.push("input", inputContainer);
      // 发送 token 数量到 Output 模块
      int tokenCnt = inputContainer.getTokenCnt();
      inputServer.send(hdl, std::to_string(tokenCnt),
                       websocketpp::frame::opcode::text);

    } else {
      BaseDisModel::appendToken(inputContainer, payload);
      shared_queue.push("input", inputContainer);
    }
  }
};

//--------------------- Comp (子线程) ---------------------
class Comp {
public:
  Comp(InputQueue &queue, MemRefContainer *resultContainerPtr)
      : shared_queue(queue), resultContainerPtr(resultContainerPtr),
        paramsContainer({ParamSize}) {
    
  }
  void init() { loadAllParameters(); }
  void run() {
    while (true) {
      auto input = shared_queue.pop<Text<size_t, 2>>("input");
      _mlir_ciface_forward0(resultContainerPtr, &paramsContainer, &input);
      shared_queue.push("output", resultContainerPtr);
      std::cout << "forward0 computed." << std::endl;
    }
  }

private:
  InputQueue &shared_queue;
  MemRefContainer *resultContainerPtr;
  MemRef<float, 1> paramsContainer;

  void loadAllParameters() {
    /// Define directories of parameter file.
    std::string llamaBuildDir = LLAMA_EXAMPLE_BUILD_PATH;
    std::string paramsDir =
        llamaBuildDir + "/subgraph0_arg0.data"; // 权重文件路径
    BaseDisModel::loadParameters(paramsDir, paramsContainer);
  }
};
