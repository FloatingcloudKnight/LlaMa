# ===- buddy-lenet-import.py ---------------------------------------------------
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ===---------------------------------------------------------------------------
#
# This is the LeNet model AOT importer.
#
# ===---------------------------------------------------------------------------

import os
from pathlib import Path
import argparse

import numpy as np
import torch
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.type import DeviceType
from buddy.compiler.graph.transform import simply_fuse, apply_classic_fusion
from buddy.compiler.ops import tosa
from buddy.compiler.graph.operation import *
from model import LeNet

# Parse command-line arguments.
parser = argparse.ArgumentParser(description="LeNet model AOT importer")
parser.add_argument(
    "--output-dir", 
    type=str, 
    default="./", 
    help="Directory to save output files."
)
args = parser.parse_args()

# Ensure output directory exists.
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

# Retrieve the LeNet model path.
model_path = os.path.dirname(os.path.abspath(__file__))

model = LeNet()
model = torch.load(model_path + "/lenet-model.pth", weights_only=False)
model = model.eval()

# Initialize Dynamo Compiler with specific configurations as an importer.
dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
)

data = torch.randn([1, 1, 28, 28])
# Import the model into MLIR module and parameters.
with torch.no_grad():
    graphs = dynamo_compiler.importer(model, data)

graph = graphs[0]
params = dynamo_compiler.imported_params[graph]
group = []
for i, op in enumerate(graph._body):
    if isinstance(op, PlaceholderOp) or isinstance(op, OutputOp) or i == 25:
        continue
    group.append(op)
    subgraph_name = "subgraph0"
    graph.group_map_device[subgraph_name] = DeviceType.CPU
    graph.op_groups[subgraph_name] = group
new_group = [graph._body[25]]
subgraph_name = "subgraph1"
graph.group_map_device[subgraph_name] = DeviceType.GPU
graph.op_groups[subgraph_name] = new_group

# pattern_list = [simply_fuse]
# graphs[0].fuse_ops(pattern_list)
driver = GraphDriver(graphs[0])
for i in range(len(driver.subgraphs)):
    driver.subgraphs[i].lower_to_top_level_ir()
# driver.subgraphs[0].lower_to_top_level_ir()
driver.construct_main_graph(True)
# Save the generated files to the specified output directory.
for i in range(len(driver.modules)): 
    with open(os.path.join(output_dir, f"subgraph{i}.mlir"), "w") as module_file:
        print(driver.subgraphs[i]._imported_module, file=module_file)
    with open(os.path.join(output_dir, f"forward{i}.mlir"), "w") as module_file:
        print(driver.modules[i], file=module_file)
    # 从 GraphDriver 中获取该子图收集到的参数索引列表
    param_indices = driver.subgraph_param_indices[i]

     # 根据参数索引从 loaded_params 中提取参数，并拼接为一维数组
    selected_arrays = []
    js = 0
    for idx in param_indices:
        # 注意：loaded_params 中的每个参数都是一个 tensor
        arr = params[idx].detach().cpu().numpy().reshape(-1)
        print(len(arr))
        js = js + len(arr)
        selected_arrays.append(arr)
    print(js)
    if selected_arrays:
        concat_arr = np.concatenate(selected_arrays)
    else:
        concat_arr = np.array([])

    # 定义输出文件名，数字 i 与子图对应
    filename = os.path.join(output_dir, f"arg{i}.data")
    concat_arr.tofile(filename)
# with open(output_dir / "subgraph0.mlir", "w") as module_file:
#     print(driver.subgraphs[0]._imported_module, file=module_file)
# with open(output_dir / "forward.mlir", "w") as module_file:
#     print(driver.construct_main_graph(True), file=module_file)

# params = dynamo_compiler.imported_params[graph]

# float32_param = np.concatenate(
#     [param.detach().numpy().reshape([-1]) for param in params]
# )

# float32_param.tofile(output_dir / "arg0.data")
