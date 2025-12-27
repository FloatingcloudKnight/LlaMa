// -*- coding: utf-8 -*-
// @Time : 2025-06-25 10:00:00
// @Author : gaoxu
// @Project : buddy compiler llama

#include "llamaRMS.h"
// -----------------------------------------------------------------------------
// LLaMA Inference Main Entry
// -----------------------------------------------------------------------------

int main() {
  RMSQueue shared_queue;
  RMSMess rmsMess("RMSMess0", shared_queue, 9002, "ws://localhost:9001");
  Comp comp(shared_queue, 1);

  std::thread rms_thread([&rmsMess] { rmsMess.run(); });
  std::this_thread::sleep_for(std::chrono::seconds(1));
  comp.init();
  std::thread comp_thread([&comp] { comp.run(); });

  rms_thread.join();
  comp_thread.join();

  return 0;
}
