// -*- coding: utf-8 -*-
// @Time : 2025-06-25 10:00:00
// @Author : gaoxu
// @Project : buddy compiler llama

#include "llamaMHA.h"
// -----------------------------------------------------------------------------
// LLaMA Inference Main Entry
// -----------------------------------------------------------------------------

int main() {
  MHAQueue shared_queue;
  MHAMess mahMess("MHAMess1", shared_queue, 9005, "ws://localhost:9001", "ws://localhost:9002", "ws://localhost:9003");
  Comp comp(shared_queue, "1");

  std::thread mha_thread([&mahMess] { mahMess.run(); });
  std::this_thread::sleep_for(std::chrono::seconds(1));
  comp.init();
  std::thread comp_thread([&comp] { comp.run(); });

  mha_thread.join();
  comp_thread.join();

  return 0;
}
