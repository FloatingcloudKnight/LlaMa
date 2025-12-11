# ===- graph_driver.py ---------------------------------------------------------
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
# This is the graph driver to drive the input graph:
#     1. Split the input graph into subgraphs.
#     2. Construct a main graph to call subgraphs with right order.
#
# ===---------------------------------------------------------------------------

import os      #########
import functools  #########
import numpy as np  #########
from mlir import ir
import torch  #########
from collections import deque, defaultdict

from .graph import Graph, GraphImporter, TensorMeta

from .operation import * #########
from .type import * #########

class GraphDriver:
    """
    Class responsible for managing and driving the execution of a computational
    graph.

    Attributes:
    - _graph (Graph): The computational graph associated with this driver.
    - _subgraphs (dict): A dictionary mapping subgraph names to their
    corresponding subgraphs.
    - _subgraphs_inputs (dict): A dictionary mapping subgraph names to their
    input placeholders.
    - _subgraphs_outputs (dict): A dictionary mapping subgraph names to their
    output op's result.
    """

    def __init__(self, graph: Graph, parallelism: int = 1) -> None:   #########
        """
        Initialize the GraphDriver object with a given computational graph.

        Args:
        - graph (Graph): The computational graph to be associated with this
        driver.

        Returns:
        - None
        """
        self._graph = graph
        self._parallelism = parallelism    #######
        # 对原图的操作分组
        self.op_groups = self._graph.op_groups  #########
        self.group_map_device = self._graph.group_map_device #########
        self._subgraph_dependencies = {}   #########更改了
        self._paral_op_shape: Dict[str, List[int]] = {}   #########
        (
            self._subgraphs_inputs,
            self._subgraphs_outputs,
        ) = self.get_split_strategy()
        self._call_table = {}  
        (
            self._subgraphs
        ) = self.build_subgraph_by_group()
        self._maingraphs = {}
        self._modules = {}
        # 新增：子图参数索引表 {子图名: 参数索引列表}
        self._subgraph_param_info = defaultdict(dict)

    @property
    def subgraphs(self):
        return list(self._subgraphs.values())

#############################################################
    @property
    def maingraphs(self):
        return list(self._maingraphs.values())
    
    @property
    def modules(self):
        return list(self._modules.values())
    
    @property
    def subgraph_param_indices(self):
       return list(self._subgraph_param_indices.values())
    
    def _add_paral_op_shape(self, op_name, shape):
        if op_name not in self._paral_op_shape.keys():
            self._paral_op_shape[op_name] = shape

    # 统一两个有不同维度的数据的维度, 低维数据扩展到高维
    def _normalize_binary_operator_shape(self, shp1, shp2):
        """Normalize the shape of two input tensors according to the broadcasting
        rule"""
        shp1 = list(shp1)
        shp2 = list(shp2)
        while len(shp1) < len(shp2):
            shp1.insert(0, 1)
        while len(shp2) < len(shp1):
            shp2.insert(0, 1)

        return shp1, shp2
    
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
    
    # 获取参数的总大小
    def get_pack_params_size(self, tensors_meta: list[TensorMeta]) -> int:
        param_total_size = 0
        for tensor_meta in tensors_meta:
            param_total_size += functools.reduce(
                lambda x, y: x * y, list(tensor_meta.shape), 1
            )
        return param_total_size
    
    # 生成模型的分割策略，并根据这个策略推断出每个子模块的输入和输出参数
    def get_split_strategy(self):
      """
      Group ops based on the computational graph in terms of subgraphs.
      
      Analyse the inputs and outputs of each subgraph.

      Update the shape information of the nodes in each subgraph 
      associated with the weight matrix to be split.

      Returns:
      - None
      """
      # 对是否需要分割两种情况进行处理
      if self._parallelism < 1:
        raise ValueError(
            "Parallelism must be greater than or equal to 0")
      elif self._parallelism > 1:
        self.op_groups = {}
        self.group_map_device = {}
        # 对计算图的op进行分组，分组策略为：以PowOp为间隔放在一个subgraph中，忽略PlaceholderOp和OutputOp
        submodel_count = 0
        # prefill
        # ops_count = [6, 65, 2, 6, 14, 2] 
        
        # decode
        ops_count = [6, 65, 2, 6, 14, 2] 
        
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
          
          if pow_count > 0 and pow_count < 57:
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

      # 识别每个子图的输入节点，并将这些输入节点存储在subgraphs_inputs中
      # 每个子图的输入节点是那些不属于当前子图但与当前子图中的操作有依赖关系的节点。
      subgraphs_inputs = {}

      # 预置每个权重矩阵的分割位置 
      paral_pos0 = [-1, -1, -1, -1]
      paral_pos2 = [-1, -1, -1]
    #   paral_pos0 = [-1, -1, -1]
      #decode
      paral_pos1 = [[-1, -1], [-1, 0, 0, 0, 0, 0, 0, -1, -1, 1, -1, 1, -1, 1], [-1, -1], [-1, -1], [0, -1, 0, 1], [-1, -1]]
    #   prefill
    #   paral_pos1 = [[1, -1], [-1, 0, 0, 0, 0, 0, 0, -1, -1, -1, 1], [0, 1], [1, -1], [0, -1, 0, 1], [0, 1]]
#  [0, -1, 0, 0, -1, -1, -1, 1] 权重矩阵分列 原始数据不分 权重 权重 位置编码不分 位置 mask不分 o行分
# 原始数据不分 权重分列 偏置分列 权重1分列 偏置分列 权重1分列 偏置分列 位置 位置 mask 权重o(最后的结果：kv cache 1 1 1024 128 (分了) 1024x1536)
# 
      # Identify inputs for each subgraph
      for i, subgraph_name in enumerate(self.op_groups.keys()):
          subgraphs_inputs[subgraph_name] = []
        #   if(i == 0 or i == 169 ):
          if(i == 0):
              paral_pos = paral_pos0
          elif(i==169):
              paral_pos = paral_pos2
          else:
              paral_pos = paral_pos1[(i-1)%6]
          input_count = 0
          for op in self.op_groups[subgraph_name]:
                # 构造一个临时的依赖列表。
                # 默认包含 op._parents，但会额外扫描 args 里的 list 结构，
                # 把漏掉的张量名（如 arg58_1）加进来。
            #   current_dependencies = list(op._parents)
              current_dependencies = []
              for p in op._parents:
                # _parents 里可能是 Node 对象，也可能是字符串名
                if isinstance(p, str):
                    current_dependencies.append(p)
                elif hasattr(p, "name"):
                    current_dependencies.append(p.name)
                else:
                    current_dependencies.append(str(p))
            
              def _get_node_name(obj):
                    if isinstance(obj, str):
                        return obj
                    if hasattr(obj, "name"): # 如果是 Node 对象
                        return obj.name
                    return str(obj) #
                
                # 遍历 args，寻找隐藏在 list 中的依赖 (针对 IndexPut 算子)
              for arg in op.args:
                    if isinstance(arg, list):
                        for item in arg:
                            if item is not None:
                                # 获取潜在的名字
                                potential_name = _get_node_name(item)
                                # 只有当这个名字是图里的一个节点，且还没在依赖列表里时，才添加
                                if (potential_name in self._graph.node_table) and \
                                   (potential_name not in current_dependencies):
                                    current_dependencies.append(potential_name)
                                    # print(f"[Fix Success] 在 {op.name} 的 args 列表中发现了隐藏依赖: {potential_name} (类型: {type(item)})")
                            # # 如果 list 里是字符串（张量名），且没在 _parents 里，手动加上
                            # if isinstance(item, str) and (item not in current_dependencies):
                            #     # 排除 None
                            #     current_dependencies.append(item)
                                # print(f"[Fix] 手动添加漏掉的依赖: {item} -> {op.name}") 
                # === [新增修复代码 END] ===
            #           # === DEBUG START: 专门针对 index_put 的调试信息 ===
                      
            #   is_target_op = "index_put"
            #   if op.name == is_target_op:
            #         print(f"\n[DEBUG-START] Processing {op.name} in {subgraph_name}")
            #         print(f"  -> 原始依赖: {op._parents}")
            #         print(f"  -> 修正后依赖: {current_dependencies}")
            #             # === DEBUG END ===
              for parent in current_dependencies:
                #   if parent not in self._graph.node_table:
                #         if op.name == is_target_op:
                #             print(f"  [WARNING] 父节点 {parent} 不在 graph.node_table 中！")
                #         continue
                  op_parent = self._graph.node_table[parent]
                  is_internal = op_parent in self.op_groups[subgraph_name]
                  is_added = op_parent in subgraphs_inputs[subgraph_name]
                #   # === DEBUG START ===
                #   if op.name == is_target_op:
                #         print(f"  检查父节点: {parent}")
                #         print(f"    -> 是否在当前子图内 (Internal): {is_internal}")
                #         print(f"    -> 是否已添加为输入 (Added): {is_added}")
                #     # === DEBUG END ===
                  if (not is_internal) and (not is_added):
                      subgraphs_inputs[subgraph_name].append(op_parent)
                        # === DEBUG START ===
                        # if op.name == is_target_op:
                        #     print(f"    -> [ACTION] 将 {parent} 添加为 {subgraph_name} 的外部输入 (Placeholder)")
                        # === DEBUG END ===
                #   if ( op_parent not in self.op_groups[subgraph_name]
                #   ) and (op_parent not in subgraphs_inputs[subgraph_name]):
                #       subgraphs_inputs[subgraph_name].append(op_parent)
                      if self._parallelism > 1:
                        op_parent_shape = list(op_parent.tensor_meta["shape"])
                        if input_count >= len(paral_pos):
                            print(f"[DEBUG] subgraph={subgraph_name}, len(paral_pos)={len(paral_pos)}, input_count={input_count}")
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
                  op not in subgraphs_outputs[subgraph_name]
              ):
                  subgraphs_outputs[subgraph_name].append(op)
      
      # 更新每个子图中与需要拆分的权重矩阵相关的节点的shape信息，对PermuteOp, MatmulOp, AddOp, 
      # SubOp, MulOp, DivOp, ViewOp, CatOp等会根据输入数据的shape而改变输出数据的shape的
      # 算子进行特殊处理
      if self._parallelism == 1:
        return subgraphs_inputs, subgraphs_outputs
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
               
              elif isinstance(node, AddMMOp) and node != self.op_groups[subgraph_name][-1]:
                  # addmm 有三个输入参数：
                  #   args[0] = bias (tensor<MxN> or broadcastable)
                  #   args[1] = input (tensor<MxK>)
                  #   args[2] = weight (tensor<KxN>)
                  assert len(node.args) == 3, f"AddmmOp expects 3 arguments, got {len(node.args)}"

                  # 尝试从缓存或 graph 中获取输入张量形状
                  def get_shape(arg_name):
                      if arg_name in self._paral_op_shape:
                          return self._paral_op_shape[arg_name]
                      elif arg_name in self._graph.node_table:
                          return list(self._graph.node_table[arg_name].tensor_meta["shape"])
                      else:
                          return None

                  bias_shape = get_shape(node.args[0])
                  input_shape = get_shape(node.args[1])
                  weight_shape = get_shape(node.args[2])

                  # 仅当输入和权重可用时才能推导输出形状
                  if input_shape is not None and weight_shape is not None:
                      M = input_shape[-2] if len(input_shape) >= 2 else input_shape[0]
                      N = weight_shape[-1]
                      new_shape = [M, N]

                      # 偏置可能是 [N] 或 [M,N]，可以广播
                      if bias_shape is not None:
                          if bias_shape == [N]:
                              pass  # 单行偏置广播
                          elif bias_shape == [M, N]:
                              pass  # 完全匹配
                          else:
                              print(f"[Warning] Unexpected bias shape for AddmmOp: {bias_shape}")

                      self._add_paral_op_shape(node.name, new_shape)
                    #   print(f"[Fix] Inferred shape for AddmmOp {node.name}: {new_shape}")
                  else:
                      print(f"[Warning] Cannot infer AddmmOp shape, missing input/weight shapes for {node.name}")
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
                  debug_target = "view_14"
                #   if node.name == debug_target:
                #     print("\n" + "="*100)
                #     print(f"[DEBUG-STEP] Processing ViewOp: {node.name}")
                #     print(f"Args: {node.args}")
                #     # print(f"New shape (from node.args[1]): {list(node.args[1])}")
                #     # print(f"All current _paral_op_shape keys: {list(self._paral_op_shape.keys())}")
                #     print("="*100)
                  parent = node.args[0]
                  if parent in self._paral_op_shape.keys():
                      old_shape = self._paral_op_shape[node.args[0]]
                      old_len = len(old_shape)
                      tmp_old_shape = [d for d in old_shape if d != 1]
                      new_shape = list(node.args[1])
                      new_len = len(new_shape)
                      tmp_new_shape = [d for d in new_shape if d != 1]
                      
                      old_total = 1
                      for x in old_shape:
                        old_total *= x

                      new_total = 1
                      for x in new_shape:
                        new_total *= x

                    # 当 new_shape 与 old_shape 的元素数不一致 →  head 合并
                      need_head_fix = (old_total != new_total)
                      
                    #   if node.name == debug_target:
                    #     print(f"[STEP 1] old_shape: {old_shape}")
                    #     print(f"[STEP 2] new_shape(before): {new_shape}")
                    #     print(f"[STEP 3] tmp_old_shape: {tmp_old_shape}")
                    #     print(f"[STEP 4] tmp_new_shape: {tmp_new_shape}")
                    #     print(f"[STEP 5] old_len={old_len}, new_len={new_len}")
                    #     # === DEBUG ADD START ===
                    #     print("\n" + "="*80)
                    #     print(f"[DEBUG-ERROR] ViewOp={node.name}")
                    #     print(f"  parent op: {node.args[0]}")
                    #     print(f"  old_shape(from _paral_op_shape): {old_shape}")
                    #     print(f"  new_shape(before): {new_shape}")
                    #     print(f"  total paral_op_shape entries: {len(self._paral_op_shape)}")
                    #     print("  === Dump of _paral_op_shape ===")
                    #     for k, v in self._paral_op_shape.items():
                    #         print(f"    {k:30s} : {v}")
                    #     print("="*80 + "\n")
                        # === DEBUG ADD END ===
                      if len(tmp_old_shape) == len(tmp_new_shape):
                          # todo: 待优化，当前处理方式只考虑<MxNx...> <--> <1xMxNx...>的情况
                          if old_len < new_len:
                              for i in range(old_len):
                                  new_shape[i+1] = old_shape[i]
                                  if node.name == debug_target:
                                    print(f"[STEP 6a] old_len<new_len, assign new_shape[{i+1}]={old_shape[i]}")
                          else:
                              for i in range(new_len):
                                  if node.name == debug_target:
                                    print(f"[STEP 6b] old_len>=new_len, assign new_shape[{i}]={old_shape[i+1]}")
                                #   print(f"[DEBUG] ViewOp={node.name}, old_shape={old_shape}, new_shape(before)={new_shape}")
                                  new_shape[i] = old_shape[i+1]
                      else:
                          # todo: 待优化，当前处理方式只考虑<...xMxN> <--> <...xMxYxZ>(其中N=YxZ)的情况
                          if old_len < new_len:
                              new_shape[-2] = old_shape[-1] // new_shape[-1]
                              if node.name == debug_target:
                                print(f"[STEP 7a] adjusted new_shape[-2] -> {new_shape[-2]}")

                          else:
                              if node.name == debug_target:
                                print(f"[STEP 7b] calling _infer_new_shape(old_shape, new_shape)")
                              new_shape = self._infer_new_shape(old_shape, new_shape)
                    #   else:
                    #     # 计算缺失的 head 数
                    #     missing_dim = old_total // (new_total if new_total != 0 else 1)

                    #     # 找到“可能的 head 维度”
                    #     # 规则：第一个非1维，且不是空间维
                    #     for i in range(1, new_len):
                    #         if new_shape[i] != 1:
                    #             new_shape[i] = missing_dim
                    #             break
                      if node.name == debug_target:
                        print(f"[STEP 8] final new_shape(after): {new_shape}")
                        print(f"[STEP 9] writing to _paral_op_shape[{node.name}] = {new_shape}")
                      self._add_paral_op_shape(node.name, new_shape)
                      if node.name == debug_target:
                         print(f"[STEP 10] current _paral_op_shape entry for {node.name}: {self._paral_op_shape[node.name]}")
                         print("="*100 + "\n")
              elif isinstance(node, CatOp) and node != self.op_groups[subgraph_name][-1]:
                  for op_arg in node.args[0]:
                      op_arg = str(op_arg)
                      if op_arg in self._paral_op_shape.keys():
                          new_shape = self._paral_op_shape[op_arg]
                          self._add_paral_op_shape(node.name, new_shape)
                          break
              elif isinstance(node, IndexPutOp) and node != self.op_groups[subgraph_name][-1]:
                  # IndexPutOp 的输出形状严格等于它第一个参数(目标张量)的形状
                target_arg = str(node.args[0])

                # 1. 尝试从已计算的并行形状表中获取 (如果前序节点被分割过)
                if target_arg in self._paral_op_shape:
                    new_shape = self._paral_op_shape[target_arg]
                    # print("11111111111111111111111UYGVJHBKNUIYTFDDXFCGVHBJ")
                    
                
                # 2. 如果表中没有，说明前序节点未被改变，直接读取原始 Graph 元数据
                else:
                    input1_node = self._graph.node_table[target_arg]
                    new_shape = list(input1_node.tensor_meta["shape"])
                    # print("222222222222222UYGVJHBKNUIYTFDDXFCGVHBJ")
                    
                
                # 3. 注册当前节点的形状
                # print("33333333333333333333UYGVJHBKNUIYTFDDXFCGVHBJ")
                # print(new_shape)
                # print(node.name)
                node.tensor_meta["shape"] = new_shape
                self._add_paral_op_shape(node.name, new_shape)
                
                # Debug (可选)
                # print(f"[IndexPutOp] Propagated shape from {target_arg} to {node.name}: {new_shape}")
                  
              elif isinstance(node, ExpandOp) and node != self.op_groups[subgraph_name][-1]:
                    op_arg = str(node.args[0])               # expand 的输入 tensor
                    if op_arg in self._paral_op_shape:       # 分割后的 shape 存在
                        new_shape = self._paral_op_shape[op_arg]   # e.g. [1,1,1024,128]

                        # ===== 原始 expand 目标 =====
                        # node.args[1] 原来是 [1,2,6,1024,128]
                        old_new_size = node.args[1]

                        # ===== 更新规则 =====
                        # expanded shape = [1, new_group, head_per_group, seq, dim]
                        # new_group 就是分割后的 shape 中第二维
                        new_group_dim = new_shape[1]

                        # 维度映射：
                        #  old_new_size = [B,  G_old, H,  S,   D]
                        #  new_new_size = [B,  G_new, H,  S,   D]
                        new_new_size = old_new_size.copy()
                        new_new_size[1] = new_group_dim      # 将 2 → 1

                        # ===== 写回算子 =====
                        node.args[1] = new_new_size
                        self._add_paral_op_shape(node.name, new_new_size)

                        # ====== Debug 输出 ======
                        # print(f"[UPDATE EXPAND] {node.name}: {old_new_size} -> {new_new_size}")

              else:
                  if node != self.op_groups[subgraph_name][-1]:
                    # === 调试 AddmmOp ===
                    if node.name == "erstdtyrfy":
                        print("\n" + "="*100)
                        print(f"[DEBUG-ADDMM-GENERIC] Processing node: {node.name}")
                        print(f"Args: {node.args}")

                        matched_input = None
                        for i, op_arg in enumerate(node.args):
                            if op_arg in self._paral_op_shape.keys():
                                matched_input = op_arg
                                new_shape = self._paral_op_shape[op_arg]
                                print(f"  -> Found shape source: arg[{i}] = {op_arg}")
                                print(f"     shape = {new_shape}")
                                self._add_paral_op_shape(node.name, new_shape)
                                break

                        if matched_input is None:
                            print("  ⚠️ No input of addmm found in _paral_op_shape! Using empty shape.")
                            self._add_paral_op_shape(node.name, [])
                            new_shape = []

                        print(f"  ==> Recorded _paral_op_shape[{node.name}] = {new_shape}")
                        print("="*100 + "\n")
                        # print("  === Dump of _paral_op_shape ===")
                        # for k, v in self._paral_op_shape.items():
                        #     print(f"    {k:30s} : {v}")
                        # print("="*80 + "\n")
                        # sys.exit()
                        
                        

                    else:
                        # === 默认处理分支 ===
                        for i, op_arg in enumerate(node.args):
                            if op_arg in self._paral_op_shape.keys():
                                new_shape = self._paral_op_shape[op_arg]
                                self._add_paral_op_shape(node.name, new_shape)
                                break
      
    #   print("  === Dump of _paral_op_shape ===")
    #   for k, v in self._paral_op_shape.items():
    #         print(f"    {k:30s} : {v}")
    #   print("="*80 + "\n")
      return subgraphs_inputs, subgraphs_outputs
   ################################################ 
    # 根据分组信息构建子图，为每个子图创建一个新的Graph对象，并将相关的操作和输入输出节点添加到该子图中。
    def build_subgraph_by_group(self):
        """
        Builds subgraphs from a given graph based on groups.

        Args:
        - graph (Graph): The graph from which subgraphs are constructed.

        Returns:
        - tuple: A tuple containing dictionaries of subgraphs, subgraph inputs,
        and subgraph outputs.
        """
        subgraphs = {}

        # Construct each subgraph
        for subgraph_name in self.op_groups.keys():
            subgraph_input = []
            subgraph_body = {}
            # 设备信息
            subgraph_device = self.group_map_device[subgraph_name]

            # Construct input placeholder nodes
            for node in self._subgraphs_inputs[subgraph_name]:
                if node.name in self._paral_op_shape.keys():
                    node_shape = self._paral_op_shape[node.name]
                else:
                    node_shape = node.tensor_meta["shape"]
                node_dtype = node.tensor_meta["dtype"]
                input_tensor_meta = TensorMeta(node_shape, node_dtype)
                subgraph_input.append(input_tensor_meta)
                placeholder_node = PlaceholderOp()
                placeholder_node.name = node.name
                placeholder_node.tensor_meta = input_tensor_meta
                for op in self.op_groups[subgraph_name]:
                    # if node.name in node._parents:
                    if node.name in (op._parents if isinstance(op._parents, (list, tuple)) else []):
                        placeholder_node.add_children(op.name)
                subgraph_body[placeholder_node.name] = placeholder_node

            # Add operations to subgraph body
            for op in self.op_groups[subgraph_name]:
                # 遍历当前子图的操作，切分与权重文件相关的操作
                # 与权重文件相关的操作指参数中包含权重矩阵或参数根据权重矩阵计算获得的操作

                # ReshapeOp会改变shape,需要更新shape参数列表
                if isinstance(op, ViewOp) and self._parallelism > 1:
                    if op.args[0] in self._paral_op_shape.keys():
                        op._newshape = self._paral_op_shape[op.name]
                subgraph_body[op.name] = op
                # if op.name == "unsqueeze_7":
                #     print("Node: " + op.name)
                #     print("Type: " + str(op._op_type))
                #     print("Arguments: " + str(op.args))
                #     print("Parents: " + str(op._parents))
                #     print("Children: " + str(op._children))
                
            # Construct output node
            output_node = OutputOp()
            output_node.name = "output"
            for output in self._subgraphs_outputs[subgraph_name]:
                output_node.add_argument(output.name)
                output_node.add_parent(output.name)
            subgraph_body[output_node.name] = output_node

            # Create subgraph and add it to the dictionary
            subgraph = Graph(
                subgraph_input,
                [],
                self._graph._ops_registry,
                subgraph_name,
                subgraph_device,
                verbose=self._graph._verbose,
            )
            subgraph.body = subgraph_body.values()
            for op in subgraph_body.values():
                subgraph.node_table[op.name] = op
            # subgraph._output = output_node
            subgraphs[subgraph_name] = subgraph

        return subgraphs
    
    def topological_sort_subgraph(self):
        """
        Performs topological sorting on the subgraphs based on their dependencies.
        Args:
        - graph (Graph): The graph from which subgraphs are constructed.
        Returns:
        - list: A list of subgraph names in topological order if the graph is acyclic; otherwise, None.
        """
        # Calculate in degree of each subgraph
        in_degree = {
            subgraph_name: 0 for subgraph_name in list(self._subgraphs.keys())
        }
        for src, dests in self._subgraph_dependencies.items():
            for dest in dests:
                in_degree[dest] += 1
        # Topological sorting
        queue = deque([node for node in in_degree if in_degree[node] == 0])
        topo_order = []
        while queue:
            node = queue.popleft()
            topo_order.append(node)
            for child in self._subgraph_dependencies[node]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
        # TODO: If the custom subgraph partitioning is illegal, further partition the subgraph to make it valid.
        return (
            topo_order
            if len(topo_order) == len(list(self._subgraphs.keys()))
            else None
        )

    # 构建主图，包含子图的调用和占位符操作，根据并行度修改各个参数的shape信息.
    # 每个子图对应一个主图，主图中取权重参数并调用子图函数.
    def construct_main_graph(self, do_param_pack=False):
        """
        Constructs the main computational graph by incorporating subgraphs' call
        and placeholder operations.

        Args:
        - do_param_pack (bool): Flag indicating whether parameter packing should
        be performed. Defaults to False.

        Returns:
        - Graph: The main computational graph constructed.

        Note: The actual call sequence and topology analysis are pending
        implementation.

        """
        # Analysis topology order to sort subgraph call.
        topo_order = self.topological_sort_subgraph()
        if topo_order == None:
            print("Error : Graph Partitioning is illegal!")
            return None

        # 为每个子图创建一个FuncOp节点，并将这些节点添加到主图中。
        # Adding FuncOp nodes for each subgraph
        inputs0 = self._graph._inputs
        split_group = []
        param_size_group = []
        # for i, subgraph_name in [(2, list(self._subgraphs.keys())[2])]:
        for i, subgraph_name in enumerate(self._subgraphs.keys()):
            main_graph_name = "forward{}".format(i)
            current_param_info = {} # 存储参数索引和分割方式
            if self._parallelism > 1:  
              main_graph = Graph(
                  [],
                  [],
                  self._graph._ops_registry,
                  main_graph_name,
                  self._graph._verbose,
              )
            else:
              main_graph = Graph(
                self._graph._inputs,
                self._graph._fake_params,
                self._graph._ops_registry,
                self._graph._func_name,
                self._graph._verbose,
              )
            # 为每个子图创建一个FuncOp节点，并将这些节点添加到对应主图中。
            # FuncOp节点代表每个子图，用于主图对子图的调用
            func_node = FuncOp()
            func_node.name = subgraph_name
            func_node.tensor_meta = {"shape": [], "dtype": []}
            for inp in self._subgraphs[subgraph_name]._inputs:
                func_node.add_argument(inp)
            
            outputs = self._subgraphs[subgraph_name]._outputs
            if outputs is None or self._parallelism == 1:
                for output in self._subgraphs_outputs[subgraph_name]:
                    func_node.tensor_meta["shape"].append(
                        self._graph.node_table[output.name].tensor_meta["shape"]
                    )
                    func_node.tensor_meta["dtype"].append(
                        self._graph.node_table[output.name].tensor_meta["dtype"]
                    )
            else:
                for out_node in outputs:
                    out_type = ir.RankedTensorType(out_node.type)
                    output_shape = list(out_type.shape)
                    func_node.tensor_meta["shape"].append(torch.Size(output_shape))
                for output in self._subgraphs_outputs[subgraph_name]:
                    func_node.tensor_meta["dtype"].append(
                        self._graph.node_table[output.name].tensor_meta["dtype"]
                    )
            main_graph.add_node(func_node)
            
            # Adding placeholder operations from the original graph
            ph_count : int = 0 #原图 PlaceholderOp 的序号
            # 记录子图中是否有权重矩阵被分割
            issplit = False
            current_param_info["params"] = []
            current_param_info["total_partitions"] = 1
            split_group.append(1)
            current_subgraph_input_names = set(n.name for n in self._subgraphs_inputs[subgraph_name])
            # print("HERE!!!!!!!!!!!!")
            # print(current_subgraph_input_names)
            # print("HERE!!!!!!!!!!!!")
            #遍历原图所有 PlaceholderOp（按出现顺序 ph_count）， 找到哪些是当前子图需要的，然后复制成主图里的参数 placeholder。
            # 处理从原本子图中就能得到的输入 
            #新增是子图输入但不从权重文件中读取的部分（decode子图从prefill拿到的kv cache）
            maingraph_input = list(inputs0) # 初始=self._graph._inputs
            for node in self._graph.body:
                if isinstance(node, PlaceholderOp):
                    # if node in self._subgraphs_inputs[subgraph_name]:
                    if node.name in current_subgraph_input_names:
                        # print(f"1"+node.name)
                        # print(f"len(self._graph._fake_params)")
                        # print(+len(self._graph._fake_params))
                        # print(f"ph_count")
                        # print(ph_count)
                        if(len(self._graph._fake_params) > (ph_count) and self._parallelism > 1):
                            # print(f"2"+node.name)
                            main_graph._fake_params.append(self._graph._fake_params[ph_count])
                            if node.name in self._paral_op_shape.keys():
                                # print(f"3"+node.name)
                                node._newshape = self._paral_op_shape[node.name]
                                main_graph._fake_params[-1]['shape'] = torch.Size(node._newshape)
                                current_param_info["params"].append(
                                    {"index": ph_count, "split_degree": node._newshape}
                                )
                                issplit = True
                            else: 
                                # print(f"4"+node.name)
                                current_param_info["params"].append(
                                    {"index": ph_count, "split_degree": []}
                                )
                        elif(self._parallelism > 1):
                            # current_param_info["params"].append(
                            #     {"index": ph_count, "split_degree": []}
                            # )
                            # if i > 0:
                            #     node_shape = node.tensor_meta["shape"]
                            #     node_dtype = node.tensor_meta["dtype"]
                            #     input_tensor_meta = TensorMeta(node_shape, node_dtype)
                            #     maingraph_input.append(input_tensor_meta)
                            if node.name in self._paral_op_shape:
                                node_shape = self._paral_op_shape[node.name]
                            else:
                                node_shape = node.tensor_meta["shape"]
                            node_dtype = node.tensor_meta["dtype"]
                            input_tensor_meta = TensorMeta(node_shape, node_dtype)
                            maingraph_input.append(input_tensor_meta)
                        # print(f"5"+node.name)
                        main_graph.add_node(node)


                    ph_count += 1
            param_size_group.append(self.get_pack_params_size(main_graph._fake_params))
            # print("drtryughbjgyuojikb vcfgyuiojlk")
            # # print(main_graph)
            
            # for i, op in enumerate(main_graph._body):
            # # #     if op.name == "clone_4" or op.name == "expand_6" or op.name == "view_14":
            # # #         print(f"[DEBUG] {op.name}: inferred shape = {op.tensor_meta}")

            # #     # print("=" * 20 + "Graph Node" + "=" * 20)
            #     print(f"{i}  ")
            #     # print(op)
            #     print("Node: " + op.name)
            #     print("Type: " + str(op._op_type))
            #     print("Arguments: " + str(op.args))
            #     print("Parents: " + str(op._parents))
            #     print("Children: " + str(op._children))
            # print("drtryughbjgyuojikb vcfgyuiojlk")
            

            if issplit: 
                current_param_info["total_partitions"] = self._parallelism
                split_group[-1] = self._parallelism
            self._subgraph_param_info[subgraph_name] = current_param_info

            #处理来自其他子图的输入节点
            # maingraph_input = list(inputs0) # 初始=self._graph._inputs
            # Identify inputs for each subgraph
            if self._parallelism > 1:
              for node in self._subgraphs_inputs[subgraph_name]:
                if (node.name not in main_graph.node_table.keys()):
                  if node.name in self._paral_op_shape.keys():
                    node_shape = self._paral_op_shape[node.name]
                    # issplit = True
                  else:
                    node_shape = node.tensor_meta["shape"]
                  node_dtype = node.tensor_meta["dtype"]
                  input_tensor_meta = TensorMeta(node_shape, node_dtype)
                  maingraph_input.append(input_tensor_meta)
                  placeholder_node = PlaceholderOp()
                  placeholder_node.name = node.name
                  placeholder_node.tensor_meta = input_tensor_meta
                  main_graph.add_node(placeholder_node)

            # for i, op in enumerate(main_graph._body):
            # # #     if op.name == "clone_4" or op.name == "expand_6" or op.name == "view_14":
            # # #         print(f"[DEBUG] {op.name}: inferred shape = {op.tensor_meta}")

            # #     # print("=" * 20 + "Graph Node" + "=" * 20)
            #     print(f"{i}  ")
            #     # print(op)
            #     print("Node: " + op.name)
            #     print("Type: " + str(op._op_type))
            #     print("Arguments: " + str(op.args))
            #     print("Parents: " + str(op._parents))
            #     print("Children: " + str(op._children))
            # print("drtryughbjgyuojikb vcfgyuiojlk")

            
            # Adding CallOp to invoke the single subgraph
            call_node = CallOp()
            call_node.name = "call{}".format(i)
            call_node.call_func_name = subgraph_name
            call_node.tensor_meta = {"shape": [], "dtype": []}
            for node in self._subgraphs_inputs[subgraph_name]:
                if node.name in self._graph.node_table:
                    call_node.add_argument(node.name)
                    continue
                for key, value in self._subgraphs_outputs.items():
                    if node in value:
                        call_node.add_argument(
                            arg=self._call_table[key].name,
                            arg_index=value.index(node.name),
                        )
                        break
            outputs = self._subgraphs[subgraph_name]._outputs
            if outputs is None or self._parallelism == 1:
                for output in self._subgraphs_outputs[subgraph_name]:
                    call_node.tensor_meta["shape"].append(
                        self._graph.node_table[output.name].tensor_meta["shape"]
                    )
                    call_node.tensor_meta["dtype"].append(
                        self._graph.node_table[output.name].tensor_meta["dtype"]
                    )
            else:
                for out_node in outputs:
                    out_type = ir.RankedTensorType(out_node.type)
                    output_shape = list(out_type.shape)
                    call_node.tensor_meta["shape"].append(torch.Size(output_shape))
                for output in self._subgraphs_outputs[subgraph_name]:
                    call_node.tensor_meta["dtype"].append(
                        self._graph.node_table[output.name].tensor_meta["dtype"]
                    )
            self._call_table[subgraph_name] = call_node
            main_graph.add_node(call_node)

            # Adding GetItemOps to retrieve individual output tensors
            output_node = OutputOp()
            for m, output in enumerate(self._subgraphs_outputs[subgraph_name]):
                getitem_node = GetItemOp()
                getitem_node.add_argument(call_node.name)
                getitem_node.add_argument(m)
                getitem_node.name = "getitem{}".format(m)
                output_node.add_argument(getitem_node.name)
                main_graph.add_node(getitem_node)
            # Marking the final output of the main graph
            output_node.name = "output"
            main_graph.add_node(output_node)
            self._maingraphs[main_graph_name] = main_graph

            # Importing the main graph
            with ir.Location.unknown(ir.Context()):
                main_importer = GraphImporter(
                    main_graph.body,
                    main_graph._fake_params,
                    maingraph_input,
                    main_graph._func_name,
                    main_graph._ops_registry,
                    do_param_pack,
                )
                if self._parallelism == 1:
                    return main_importer.import_main_graph()
                # print("=== Debug main_graph before import ===")
                # print("main_graph_name:", main_graph_name)
                # print("len(maingraph_input):", len(maingraph_input))
                # print("maingraph_input shapes/dtypes:", [ (m.shape, m.dtype) for m in maingraph_input ])
                # print("len(main_graph._fake_params):", len(main_graph._fake_params))
                # print("fake_params shapes:", [p['shape'] if isinstance(p, dict) and 'shape' in p else None for p in main_graph._fake_params])
                # print("nodes in main_graph node_table:", list(main_graph.node_table.keys()))
                # print("main_graph.body node count:", len(main_graph.body))
                self._modules[main_graph_name] = main_importer.import_main_graph()
                inputs0 = [] 
        
        print(f"split_group: {split_group}")
        print(f"param_size_group: {param_size_group}")
    
    # 根据计算图分割的结果，构建子权重矩阵并打包为新权重文件
    def construct_sub_params(self, params, subgraph_entry, output_dir):
        """
        处理参数并根据 subgraph 的配置生成多个权重文件。
        
        参数:
        params: 分离出的全部参数，由 params = dynamo_compiler.imported_params[graph]获得
        subgraph: 包含 'params'（参数配置列表） 和 'total_partitions' 键的字典，
                    其中每个参数配置包括:
                    - "index": 在 params 中的索引
                    - "split_degree": 分片形状
        output_dir: 输出目录，将在此目录中生成 arg0.data, arg1.data, ... 文件
        """
        subgraph_name, subgraph = subgraph_entry
        total_partitions = subgraph["total_partitions"]

        # 为每个分区建立列表，存放各个参数（切分后的部分）的 flattened 数组
        partition_data = [[] for _ in range(total_partitions)]
        
        # 按 subgraph["params"] 中的顺序处理每个参数
        for param_info in subgraph["params"]:
            idx = param_info["index"]
            split_degree = param_info["split_degree"]
            
            # 从参数列表中获取 tensor
            tensor = params[idx]
            
            # 将 tensor 转为 NumPy 数组
            np_tensor = tensor.detach().cpu().numpy()
            orig_shape = np_tensor.shape
            
            if not split_degree:
                # 不切分，完整 tensor 复制到每个权重矩阵
                flat = np_tensor.reshape(-1)
                for part in range(total_partitions):
                    partition_data[part].append(flat)
            
            else:
                # split_degree 给出每个切片的形状
                slice_shape = tuple(split_degree)
                if len(orig_shape) != len(slice_shape):
                    raise ValueError(
                        f"参数索引 {idx} 的原始形状 {orig_shape} 与 split degree {slice_shape} 维度不匹配"
                    )
                # 确定切分轴：slice_shape[axis] * total_partitions == orig_shape[axis]
                axis = None
                for dim in range(len(orig_shape)):
                    if slice_shape[dim] * total_partitions == orig_shape[dim] and \
                    all(slice_shape[d] == orig_shape[d] for d in range(len(orig_shape)) if d != dim):
                        axis = dim
                        break
                if axis is None:
                    raise ValueError(
                        f"参数索引 {idx} 的 split degree {slice_shape} 无法与原始形状 {orig_shape} 匹配 (分区数={total_partitions})"
                    )
                # 按轴切分
                for part in range(total_partitions):
                    start = part * slice_shape[axis]
                    end = (part + 1) * slice_shape[axis]
                    slicer = [slice(None)] * len(orig_shape)
                    slicer[axis] = slice(start, end)
                    sliced = np_tensor[tuple(slicer)]
                    partition_data[part].append(sliced.reshape(-1))
        
        # 为每个分区将所有切分后的参数拼接，并写入输出文件
        for part in range(total_partitions):
            # 若当前分区没有数据，也生成一个空文件
            if partition_data[part]:
                concat_arr = np.concatenate(partition_data[part])
            else:
                concat_arr = np.array([])
            filename = os.path.join(output_dir, f"{subgraph_name}_arg{part}.data")
            concat_arr.tofile(filename)
            
            # # 输出调试信息
            # print(f"保存分区 {part} 权重到 {filename}")
            # print(f"总元素数: {concat_arr.size}")
            # print(f"内存占用: {concat_arr.nbytes/1024**2:.2f} MB\n")
