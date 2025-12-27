// -*- coding: utf-8 -*-
// @Time : 2025-06-25 10:00:00
// @Author : gaoxu
// @Project : buddy compiler llama

#include "llamaOutput.h"
// -----------------------------------------------------------------------------
// LLaMA Inference Main Entry
// -----------------------------------------------------------------------------

int main() {
  OutputQueue shared_queue;
  OutputMess outputMess(shared_queue, "ws://localhost:9012", "ws://localhost:9013", "ws://localhost:9001");
  Comp comp(shared_queue);

  std::thread output_thread([&outputMess] { outputMess.run(); });
  std::this_thread::sleep_for(std::chrono::seconds(1));
  comp.init();
  std::thread comp_thread([&comp] { comp.run(); });

  output_thread.join();
  comp_thread.join();

  return 0;
}
