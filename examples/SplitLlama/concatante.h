#ifndef CONCATNATE_H//作用：防止concatnate.h被重复引用
#define CONCATNATE_H
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

//--------------------- Concatenate (主线程) ---------------------
class Concatenate {
public:
  Concatenate(const uint16_t& port, const std::string& uri, const std::string& uri0)
      : concateServer(), hdlsSymbol(),
        subResultContainer0(
            MemRef<float, 3>({1, SubMaxTokenLength, HiddenSize})),
        dataId(0) {
    /// 服务器初始化
    concateServer.set_access_channels(websocketpp::log::alevel::none);
    concateServer.clear_access_channels(websocketpp::log::alevel::all);
    concateServer.init_asio();

    concateServer.set_message_handler(
        bind(&Concatenate::on_server_message, this, _1, _2));
    concateServer.listen(port);
    concateServer.set_reuse_addr(true);
    concateServer.start_accept();

    /// 客户端初始化
    // 禁用客户端日志
    rmsClient.set_access_channels(websocketpp::log::alevel::none);
    rmsClient.clear_access_channels(websocketpp::log::alevel::all);
    rmsClient.init_asio(); 

    rmsClient.set_open_handler([this](websocketpp::connection_hdl hdl) {
      rmsClient.send(hdl, "Concatenate", websocketpp::frame::opcode::text);
    });
    rmsClient.set_message_handler(bind(&Concatenate::on_client_message, this, _1, _2));
    // 启动连接
    websocketpp::lib::error_code ec;
    auto con = rmsClient.get_connection(uri, ec);
    rmsClient.connect(con);

    // 禁用客户端日志
    rmsClient0.set_access_channels(websocketpp::log::alevel::none);
    rmsClient0.clear_access_channels(websocketpp::log::alevel::all);
    rmsClient0.init_asio(); 

    rmsClient0.set_open_handler([this](websocketpp::connection_hdl hdl) {
      rmsClient0.send(hdl, "Concatenate", websocketpp::frame::opcode::text);
    });
    rmsClient0.set_message_handler(bind(&Concatenate::on_client_message, this, _1, _2));
    // 启动连接
    websocketpp::lib::error_code ec0;
    auto con = rmsClient0.get_connection(uri0, ec0);
    rmsClient0.connect(con);
  }

  void run() {
    std::thread client_thread([this]() { rmsClient.run(); });
    std::thread client0_thread([this]() { rmsClient0.run(); });
    // 启动 WebSocket 服务器线程
    std::thread server_thread([this]() { concateServer.run(); });

    client_thread.join();
    client0_thread.join();
    server_thread.join();
  }

private:
  server concateServer;
  client rmsClient;
  client rmsClient0;
  std::map<std::string, std::vector<websocketpp::connection_hdl>> hdlsSymbol;
  std::map<websocketpp::connection_hdl, std::string,
           std::owner_less<websocketpp::connection_hdl>>
      connections;
  std::mutex symbolMutex; // 保护 hdlsSymbol 的互斥锁
  MemRef<float, 3> subResultContainer0;
  //  确保对dataId的操作是​​原子​​的
  std::atomic<uint32_t> dataId;
  std::vector<std::vector<float>> received_data;
  std::mutex data_mutex;
  // 存储每个参数的shape
  std::vector<intptr_t> inputSizes = {{1, SubMaxTokenLength, HiddenSize}};

  void send_data(websocketpp::connection_hdl hdl, uint32_t dataId, const std::vector<std::vector<float>> &data) {
    const uint8_t total = data.size();

    for (auto mhaHdl :hdlsSymbol["MHAMess"]){
      if (concateServer.get_con_from_hdl(mhaHdl)->get_state() !=
        websocketpp::session::state::open)
      return;
    }

    for (uint8_t i = 0; i < total; ++i) {
      const auto &subdata = data[i];

      // 构造协议头
      std::vector<uint8_t> packet(8); // 4+1+1+2=8字节头
      memcpy(packet.data(), &dataId, 4);
      packet[4] = total;
      packet[5] = i;
      uint16_t num = subdata.size();
      memcpy(packet.data() + 6, &num, 2);

      // 添加浮点数据
      const uint8_t *binaryData =
          reinterpret_cast<const uint8_t *>(subdata.data());
      packet.insert(packet.end(), binaryData,
                    binaryData + subdata.size() * sizeof(float));

      concateServer.send(hdl, packet.data(), packet.size(),
                       websocketpp::frame::opcode::binary);
    }
  }

  void on_server_message(websocketpp::connection_hdl hdl,
                         server::message_ptr msg) {
    std::string payload = msg->get_payload();
    if (payload == "MHAMess" || payload == "AddMess") {
      std::lock_guard<std::mutex> lock(symbolMutex); // 加锁保护符号表
      auto hdlsVec = hdlsSymbol[payload];
      hdlsVec.push_back(hdl);
      hdlsSymbol[payload] = hdlsVec;
      connections[hdl] = payload;
      std::cout << payload << " 已连接" << std::endl;
      return;
    }
  }

  void on_client_message(websocketpp::connection_hdl, client::message_ptr msg) {
    try {
        if (msg->get_opcode() != websocketpp::frame::opcode::binary) {
            std::cout << "忽略非二进制消息" << std::endl;
            return;
        }

        const std::string& payload = msg->get_payload();
        if (payload.size() < 8) {
            std::cerr << "错误: 协议头不完整(需要至少8字节)" << std::endl;
            return;
        }

        // 解析协议头
        uint32_t batch_id;
        uint8_t total_chunks, chunkSeq;
        uint16_t num_elements;

        memcpy(&batch_id, payload.data(), 4);
        total_chunks = payload[4];
        chunkSeq = payload[5];
        memcpy(&num_elements, payload.data() + 6, 2);

        // 验证分块序号有效性
        if (chunkSeq >= total_chunks) {
            std::cerr << "错误：非法分块序号 " << (int)chunkSeq 
                      << " (总块数=" << (int)total_chunks << ")" << std::endl;
            return;
        }

        // 验证数据长度
        const size_t expectedSize = 8 + num_elements * sizeof(float);
        if (payload.size() != expectedSize) {
            std::cerr << "错误：数据长度不匹配(预期=" << expectedSize 
                      << " 实际=" << payload.size() << ")" << std::endl;
            return;
        }

        // 提取浮点数据
        const float* float_data = reinterpret_cast<const float*>(payload.data() + 8);
        std::vector<float> chunk(float_data, float_data + num_elements);

        // 存储数据
        {
          std::lock_guard<std::mutex> lock(data_mutex);
          received_data.push_back(chunk);
        }
        intptr_t sizes[3] = {inputSizes[chunkSeq]};
        MemRef<float, 3> subResultContainer(chunk.data(), sizes);
        shared_queue.push_input(subResultContainer);
        send_data(hdlsSymbol["AddMess"][0], dataId++, {chunk});
    } catch (const std::exception& e) {
        std::cerr << "处理消息错误: " << e.what() << std::endl;
    }
}
};

#endif// CONCATNATE_H
