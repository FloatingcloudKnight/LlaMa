# ===- strategy.py ---------------------------------------------------------
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
# This is the graph splite stategy to splite the graph:
#
# ===---------------------------------------------------------------------------

from typing import Dict, Optional, Tuple, List, Union
import numpy as np

from .operation import *
from .type import *
from ..graph import Graph

from .device import *

class Strategy:
  def __init__(self, graph: Graph, parallelism: int = 1,
               devices: Dict[str, Device] = None, 
               max_memory: float = 16.0):
    self._devices = devices
    self._graph = graph
    self._parallelism = parallelism
    self._assignment_strategy: Dict[str, str] = {}
    self.op_groups: Dict[str, List[Op]] = {}
    self.group_map_device: Dict[str, DeviceType] = {}
    self.max_memory: float = max_memory # GB
    self._subgraphs_inputs: Dict[str, List[Op]] = {}
    self._subgraphs_outputs: Dict[str, List[Op]] = {}
    (
      self._subgraphs_inputs,
      self._subgraphs_outputs,
    ) = self.get_split_strategy()
    self._subgraph_dependencies = {}

  def calculate_node_memory(self, node:Op, subgraph: List[Op]) -> float:
    """
    Calculates the memory resources required when adding a node to the graph.
    """
    # TODO: 对复杂算子进行资源占用估算（flash_attention等）

    node_total_params : int = 0
    for arg in node.args:
      if arg not in subgraph:
        # 递归计算子算子的内存
        arg_shape = list(arg.tensor_meta["shape"])
        for arg_dim in arg_shape:
          if isinstance(arg_dim, int):
            node_total_params *= arg_dim

    node_shape = list(node.tensor_meta["shape"])
    for dim in node_shape:
      if isinstance(dim, int):
        node_total_params *= dim
    
    # TODO: 支持更多数据类型，目前默认数据类型是f32
    node_memory = node_total_params * 4 / (1024 * 1024)  # Convert to MB
    return node_memory
  
  def calculate_graph_memory(self, subgraph: List[Op]) -> float:
    """
    Calculates the memory resources required by the graph based on its operations.
    """
    allNode = self._subgraphs_inputs + subgraph
    total_params : int = 0
    for node in allNode:
      node_shape = list(node.tensor_meta["shape"])
      for dim in node_shape:
        if isinstance(dim, int):
          total_params *= dim

    # TODO: 支持更多数据类型，目前默认数据类型是f32
    memory_requirement = total_params * 4 / (1024 * 1024)  # Convert to MB
    return memory_requirement
  
  def calculate_compute_resources(self) -> float:
    """
    Calculates the compute resources required by the module based on its operations.
    """
    # TODO: Implement compute resource calculation based on operations
    return 0.0  
  
  # 当输出数据的shape中有-1时, 根据输入数据的shape推断输出数据的shape,
  def _infer_new_shape(self, old_shape, new_shape):
    total_size = 1
    for dim_siz in old_shape:
      total_size *= dim_siz

    neg_one_cnt = 0
    rest_size = 1
    for dim_siz in new_shape:
      if dim_siz == -1:
        neg_one_cnt += 1
        continue
      rest_size *= dim_siz

    if neg_one_cnt != 0:
      if neg_one_cnt > 1 or total_size % rest_size != 0:
        raise ValueError("Can not infer the new shape!")
      infer_dim_size = total_size // rest_size
      for i, _ in enumerate(new_shape):
        if new_shape[i] == -1:
          new_shape[i] = infer_dim_size
    return new_shape
  
  def infer_subgraph_inputs(self, op_group) -> List[Op]:
    inputs = []
    for op in op_group:
      for parent in op._parents:
        op_parent = self._graph.node_table[parent]
        if (op_parent not in op_group) and (op_parent not in inputs):
          inputs.append(op_parent)
    return inputs

  def infer_subgraph_outputs(self, op_group, output_node) -> List[Op]:
    outputs = []
    for op in op_group:
      for key in self._subgraphs_inputs.keys():
        if op in self._subgraphs_inputs[key]:
          if(op not in outputs):
            outputs.append(op)
      if (op.name in output_node) and (
        op not in outputs):
        outputs.append(op)
    return outputs
  
  def update_subgraph_shape(self):
    """
    Update the shape of op in the subgraph according to the parallelism.
    
    Returns:
    - None
    """
    # 更新每个子图中与需要拆分的权重矩阵相关的节点的shape信息，对PermuteOp, MatmulOp, AddOp, 
    # SubOp, MulOp, DivOp, ViewOp, CatOp等会根据输入数据的shape而改变输出数据的shape的
    # 算子进行特殊处理
    if self._parallelism == 1:
      return 
    for subgraph_name in self.op_groups.keys():
      new_shape = []
      for node in self.op_groups[subgraph_name]:
        if isinstance(node, PermuteOp) and node != self.op_groups[subgraph_name][-1]:
          if node.args[0] in self._paral_op_shape.keys():
            old_shape = self._paral_op_shape[node.args[0]]
            new_shape = [old_shape[index] for index in node.args[1]]
            if node != self.op_groups[subgraph_name][-1]:
              self._add_paral_op_shape(node.name, new_shape)
        elif isinstance(node, MatmulOp) and node != self.op_groups[subgraph_name][-1]:
          assert len(node.args) == 2
          # 由于MatmulOp的输入参数是其他op的结果，所以无法通过两个参数的shape来预测出结果shape
          for op_arg in node.args:
            if op_arg in self._paral_op_shape.keys():
              if(node.args[0] in self._paral_op_shape.keys()):
                input1_shape = self._paral_op_shape[node.args[0]]
              else:
                input1 = self._graph.node_table[node.args[0]]
                input1_shape = list(input1.tensor_meta["shape"])
              if(node.args[1] in self._paral_op_shape.keys()):
                input2_shape = self._paral_op_shape[node.args[1]]
              else:
                input2 = self._graph.node_table[node.args[1]]
                input2_shape = list(input2.tensor_meta["shape"])
              new_shape = input1_shape
              new_shape[-1] = input2_shape[-1]
              self._add_paral_op_shape(node.name, new_shape)
              break
        elif isinstance(node, AddOp | SubOp | MulOp | DivOp) and node != self.op_groups[subgraph_name][-1]:
          assert len(node.args) == 2
          # 由于MatmulOp的输入参数是其他op的结果，所以无法通过两个参数的shape来预测出结果shape
          for i, op_arg in enumerate(node.args):
            if op_arg in self._paral_op_shape.keys():
              broadcasted_result_shp = []
              if isinstance(node.args[1-i], int | float):
                broadcasted_result_shp = self._paral_op_shape[op_arg]
              else:
                if(node.args[0] in self._paral_op_shape.keys()):
                    input1_shape = self._paral_op_shape[node.args[0]]
                else:
                    input1 = self._graph.node_table[node.args[0]]
                    input1_shape = list(input1.tensor_meta["shape"])
                if(node.args[1] in self._paral_op_shape.keys()):
                    input2_shape = self._paral_op_shape[node.args[1]]
                else:
                    input2 = self._graph.node_table[node.args[1]]
                    input2_shape = list(input2.tensor_meta["shape"])
                norm_input1_shape, norm_input2_shape = self._normalize_binary_operator_shape(
                    input1_shape, input2_shape
                )
                for dim1, dim2 in zip(norm_input1_shape, norm_input2_shape):
                    broadcasted_result_shp.append(max(dim1, dim2))
              self._add_paral_op_shape(node.name, broadcasted_result_shp)
          del i
        elif isinstance(node, ViewOp) and node != self.op_groups[subgraph_name][-1]:
          if node.args[0] in self._paral_op_shape.keys():
            old_shape = self._paral_op_shape[node.args[0]]
            old_len = len(old_shape)
            tmp_old_shape = []
            for i in range(old_len):
              if old_shape[i] != 1:
                tmp_old_shape.append(old_shape[i])
            new_shape = list(node.args[1])
            new_len = len(new_shape)
            tmp_new_shape = []
            for i in range(new_len):
              if new_shape[i] != 1:
                tmp_new_shape.append(new_shape[i])
            if len(tmp_old_shape) == len(tmp_new_shape):
              # todo: 待优化，当前处理方式只考虑<MxNx...> <--> <1xMxNx...>的情况
              if old_len < new_len:
                for i in range(old_len):
                  new_shape[i+1] = old_shape[i]
              else:
                for i in range(new_len):
                  new_shape[i] = old_shape[i+1]
            else:
              # todo: 待优化，当前处理方式只考虑<...xMxN> <--> <...xMxYxZ>(其中N=YxZ)的情况
              if old_len < new_len:
                new_shape[-2] = old_shape[-1] // new_shape[-1]
              else:
                new_shape = self._infer_new_shape_with_neg_one(old_shape, new_shape)
            self._add_paral_op_shape(node.name, new_shape)
        elif isinstance(node, CatOp) and node != self.op_groups[subgraph_name][-1]:
          for op_arg in node.args[0]:
            op_arg = str(op_arg)
            if op_arg in self._paral_op_shape.keys():
              new_shape = self._paral_op_shape[op_arg]
              self._add_paral_op_shape(node.name, new_shape)
              break
        else:
          if node != self.op_groups[subgraph_name][-1]:
          # 默认不属于上述的操作的算子算子都与被切分的算子的shape相同
            for i, op_arg in enumerate(node.args):
              if op_arg in self._paral_op_shape.keys():
                new_shape = self._paral_op_shape[op_arg]
                self._add_paral_op_shape(node.name, new_shape)
                break
    
    return 
  
  def split_graph(self, subgraph: List[Op], subgraph_name) -> Dict[str, List[Op]]:
    new_subgraph = []
    recent_min_outputs = float('inf')
    leastest_selection = []
    subsubgraph_num = 0
    subsubgraphs:Dict[str, List[Op]] = {}
    
    for node in subgraph:  
      if self.calculate_node_memory(node, new_subgraph) > self.max_memory :
        subsubgraph_name = str(subgraph_name).format(subsubgraph_num)
        subsubgraphs[subsubgraph_name] = leastest_selection
        subsubgraph_num += 1
        new_subgraph = []
        continue

      new_subgraph.append(node)

      if self.calculate_graph_memory(node, new_subgraph) > (2/3) * self.max_memory :
        outputs = self.infer_subgraph_outputs(new_subgraph)
        if(len(outputs) <= recent_min_outputs):
          leastest_selection = new_subgraph

      return subsubgraphs

  def get_split_strategy(self):
    """
    Group ops based on the computational graph in terms of subgraphs.
    
    Analyse the inputs and outputs of each subgraph.

    Update the shape information of the nodes in each subgraph 
    associated with the weight matrix to be split.

    Returns:
    - None
    """
    if self._parallelism < 1:
      raise ValueError(
          "Parallelism must be greater than or equal to 0")
    elif self._parallelism > 1:
      self.op_groups = {}
      self.group_map_device = {}
      # 对计算图的op进行分组，分组策略为：以PowOp为间隔放在一个subgraph中，忽略PlaceholderOp和OutputOp
      submodel_count = 0
      ops_count = [6, 50, 2, 6, 14, 2]
      pow_count = 0
      tsf_count = 0
      for i, op in enumerate(self._graph._body):
        if isinstance(op, PlaceholderOp) or isinstance(op, OutputOp):
          continue
        
        if "subgraph{}".format(submodel_count) not in self.op_groups.keys():
          subgraph_name = "subgraph{}".format(submodel_count)
          self.group_map_device[subgraph_name] = DeviceType.CPU
          self.op_groups[subgraph_name] = [op]
          continue
        
        # todo: Added handling of more complex embedding cases
        if isinstance(op, PowOp):
          pow_count += 1
          submodel_count += 1
          tsf_count = 1
          subgraph_name = "subgraph{}".format(submodel_count)
          self.group_map_device[subgraph_name] = DeviceType.CPU
          self.op_groups[subgraph_name] = [op]
          continue
        
        if pow_count > 0 and pow_count < 65:
          if tsf_count == ops_count[(submodel_count-1)%6]:
            tsf_count = 1
            submodel_count += 1
            subgraph_name = "subgraph{}".format(submodel_count)
            self.group_map_device[subgraph_name] = DeviceType.CPU
            self.op_groups[subgraph_name] = [op]
            continue
          else:
            tsf_count += 1

        subgraph_name = "subgraph{}".format(submodel_count)
        group = self.op_groups[subgraph_name]
        group.append(op)
        self.op_groups[subgraph_name] = group
        
    self._subgraph_dependencies = {
        subgraph_name: set()
        for subgraph_name in list(self.op_groups.keys())
    }

    # 初始化所有子图的存储需求信息
    memory_requirements = {}
    for name, subgraph in self.op_groups.items():
      memory_requirements[name] = self.calculate_graph_memory(subgraph)

      # 对内存需求较大的子图进行模型分割
      if memory_requirements[name] > self.max_memory:
        subsubgraphs = self.split_graph(subgraph)
        self.op_groups.pop(name)
        self.op_groups = subsubgraphs + self.op_groups

    # 识别每个子图的输入节点，并将这些输入节点存储在subgraphs_inputs中
    # 每个子图的输入节点是那些不属于当前子图但与当前子图中的操作有依赖关系的节点。
    subgraphs_inputs = {}

    # 预置每个权重矩阵的分割位置
    paral_pos0 = [-1, -1, -1]
    paral_pos1 = [[1, -1], [0, -1, 0, 0, -1, -1, -1, 1], [0, 1], [1, -1], [0, -1, 0, 1], [0, 1]]
    
    # Identify inputs for each subgraph
    for i, subgraph_name in enumerate(self.op_groups.keys()):
      subgraphs_inputs[subgraph_name] = []
      if(i == 0 or i == 193):
        paral_pos = paral_pos0
      else:
        paral_pos = paral_pos1[(i-1)%6]
      input_count = 0
      for op in self.op_groups[subgraph_name]:
        for parent in op._parents:
          op_parent = self._graph.node_table[parent]
          if ( op_parent not in self.op_groups[subgraph_name]
          ) and (op_parent not in subgraphs_inputs[subgraph_name]):
            subgraphs_inputs[subgraph_name].append(op_parent)
            if self._parallelism > 1:
              op_parent_shape = list(op_parent.tensor_meta["shape"])
              pos = paral_pos[input_count]
              input_count += 1 
              if(pos != -1 and pos < len(op_parent_shape)):
                op_parent_shape[pos] = op_parent_shape[pos] // self._parallelism
                self._add_paral_op_shape(parent, op_parent_shape)

    subgraphs_outputs = {}
    output_node = []
    
    # 识别整个图的输出节点，并将这些输出节点存储在output_node列表中，收集整个图的所有输出节点
    # Identify output nodes of the entire graph
    for node in self._graph.body:
      if isinstance(node, OutputOp):
        for arg in node.args:
          if(arg not in output_node):
            output_node.append(arg)

    # 识别每个子图的输出节点，并建立子图之间的依赖关系。
    # Identify outputs for each subgraph and build dependencies between subgraphs
    for subgraph_name in self.op_groups.keys():
      subgraphs_outputs[subgraph_name] = []
      for op in self.op_groups[subgraph_name]:
        for key in subgraphs_inputs.keys():
          if op in subgraphs_inputs[key]:
            if(op not in subgraphs_outputs[subgraph_name]):
              subgraphs_outputs[subgraph_name].append(op)
            self._subgraph_dependencies[subgraph_name].add(key)
        if (op.name in output_node) and (
          op not in subgraphs_outputs[subgraph_name]):
          subgraphs_outputs[subgraph_name].append(op)

    return subgraphs_inputs, subgraphs_outputs
