// -*- coding: utf-8 -*-
// @Time : 2025-06-25 10:00:00
// @Author : gaoxu
// @Project : buddy compiler llama

#include "llamaMLP.h"
// -----------------------------------------------------------------------------
// LLaMA Inference Main Entry
// -----------------------------------------------------------------------------

int main() {

  MLPQueue shared_queue;
  MLPMess mlpMess("MLPMess0", shared_queue, 9010, "ws://localhost:9008", "ws://localhost:9009");
  Comp comp(shared_queue, "0");

  std::thread mlp_thread([&mlpMess] { mlpMess.run(); });

  std::this_thread::sleep_for(std::chrono::seconds(1));
  comp.init();
  std::thread comp_thread([&comp] { comp.run(); });

  mlp_thread.join();
  comp_thread.join();

  return 0;
}
