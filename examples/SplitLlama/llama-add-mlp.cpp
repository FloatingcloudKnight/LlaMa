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
#include "llamaAdd.h"
// -----------------------------------------------------------------------------
// LLaMA Inference Main Entry
// -----------------------------------------------------------------------------

int main() {
  AddQueue shared_queue;
  AddMess addMess("AddMess0", 1, shared_queue, 9012, "ws://localhost:9008",
                  "ws://localhost:9010", "ws://localhost:9011", "ws://localhost:9002");
  Comp comp(shared_queue);

  std::thread add_thread([&addMess] { addMess.run(); });
  std::thread comp_thread([&comp] { comp.run(); });

  add_thread.join();
  comp_thread.join();

  return 0;
}
