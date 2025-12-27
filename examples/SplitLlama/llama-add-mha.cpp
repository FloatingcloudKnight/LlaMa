// -*- coding: utf-8 -*-
// @Time : 2025-06-25 10:00:00
// @Author : gaoxu
// @Project : buddy compiler llama

#include "llamaAdd.h"
// -----------------------------------------------------------------------------
// LLaMA Inference Main Entry
// -----------------------------------------------------------------------------

int main() {
  AddQueue shared_queue;
  AddMess addMess("AddMess0", 0, shared_queue, 9006, "ws://localhost:9002",
                  "ws://localhost:9004", "ws://localhost:9005", "ws://localhost:9002");
  Comp comp(shared_queue);

  std::thread add_thread([&addMess] { addMess.run(); });
  std::thread comp_thread([&comp] { comp.run(); });

  add_thread.join();
  comp_thread.join();

  return 0;
}
