// -*- coding: utf-8 -*-
// @Time : 2025-06-25 10:00:00
// @Author : gaoxu
// @Project : buddy compiler llama

#include "llamaInput.h"
// -----------------------------------------------------------------------------
// LLaMA Inference Main Entry
// -----------------------------------------------------------------------------

int main() {
  /// Print the title of this example.
  const std::string title = "LLaMA 2 Inference Powered by Buddy Compiler";
  std::cout << "\033[33;1m" << title << "\033[0m" << std::endl;

  /// Initialize data containers
  //  - Input container.
  //  - Result container
  //  - Parameters container.
  MemRef<float, 3> myMemRef1({1, MaxTokenLength, HiddenSize});
  MemRef<float, 2> myMemRef2({MaxTokenLength, HiddenSize1});
  MemRef<float, 3> myMemRef3({1, MaxTokenLength, HiddenSize0});
  MemRef<float, 3> myMemRef4({1, MaxTokenLength, HiddenSize0});
  MemRefContainer resultContainer(myMemRef1, myMemRef2, myMemRef3, myMemRef4);
  MemRefContainer *resultContainerPtr = &resultContainer;

  /// Fill data into containers
  //  - Input: register vocabulary and tokenize the input string.
  //  - Output: register vocabulary.
  //  - Parameters: load parameters from the `arg0` file into the container.

  // SharedQueue shared_queue;
  InputQueue shared_queue;
  InputMess inputMess(shared_queue, resultContainerPtr);
  Comp comp(shared_queue, resultContainerPtr);

  std::thread input_thread([&inputMess] { inputMess.run(); });
  std::this_thread::sleep_for(std::chrono::seconds(1));
  comp.init();
  std::thread comp_thread([&comp] { comp.run(); });

  input_thread.join();
  comp_thread.join();

  // std::cout << "\n\033[33;1m[Input]\033[0m " << inputStr << std::endl;

  return 0;
}
