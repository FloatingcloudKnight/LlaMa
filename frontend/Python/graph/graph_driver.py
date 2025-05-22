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

import os
import functools
import numpy as np
from mlir import ir
import torch
from collections import deque, defaultdict

from .graph import Graph, GraphImporter, TensorMeta

from .operation import *
from .type import *

from .operation import *
from .type import *


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
    def __init__(self, graph: Graph, parallelism: int = 2) -> None:
        """
        Initialize the GraphDriver object with a given computational graph.

        Args:
        - graph (Graph): The computational graph to be associated with this
        driver.

        Returns:
        - None
        """
        self._graph = graph
        self._parallelism = parallelism
        # 对原图的操作分组
        self.op_groups: Dict[str, List[Op]] = {}
        self.group_map_device: Dict[str, DeviceType] = {}
        self._subgraph_dependencies = {}
        self._paral_op_shape: Dict[str, List[int]] = {}
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
            # print(op_name, shape)
            self._paral_op_shape[op_name] = shape

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
    
    def _infer_new_shape_with_neg_one(self, old_shape, new_shape):
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
    
    def get_pack_params_size(self, tensors_meta: list[TensorMeta]) -> int:
        param_total_size = 0
        for tensor_meta in tensors_meta:
            param_total_size += functools.reduce(
                lambda x, y: x * y, list(tensor_meta.shape), 1
            )
        return param_total_size
    def get_split_strategy(self):
        """
        Group ops based on the computational graph in terms of subgraphs.
        
        Analyse the inputs and outputs of each subgraph.

        Update the shape information of the nodes in each subgraph 
        associated with the weight matrix to be split.

        Returns:
        - None
        """
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
                    op not in subgraphs_outputs[subgraph_name]
                ):
                    subgraphs_outputs[subgraph_name].append(op)
        
        # 更新每个子图中与需要拆分的权重矩阵相关的节点的shape信息
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
        
        return subgraphs_inputs, subgraphs_outputs
    
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
            # print(f"-----------------------subgraph{m}------------------------------")
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
                    if node.name in node._parents:
                        placeholder_node.add_children(op.name)
                subgraph_body[placeholder_node.name] = placeholder_node

            # Add operations to subgraph body
            for op in self.op_groups[subgraph_name]:
                # 遍历当前子图的操作，切分与权重文件相关的操作
                # 与权重文件相关的操作指参数中包含权重矩阵或参数根据权重矩阵计算获得的操作

                # ReshapeOp会改变shape,需要更新shape参数列表
                if isinstance(op, ViewOp):
                    if op.args[0] in self._paral_op_shape.keys():
                        op._newshape = self._paral_op_shape[op.name]
                subgraph_body[op.name] = op
                
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

        # Analysis topology order to sort subgraph call.
        topo_order = self.topological_sort_subgraph()
        if topo_order == None:
            print("Error : Graph Partitioning is illegal!")
            return None
        
        # fake_params_offsets = []
        # current_fake_param_offset = 0
        # for tensorMeta in self._graph._fake_params:
        #     fake_params_offsets.append(current_fake_param_offset)
        #     current_fake_param_offset += functools.reduce(
        #         lambda x, y: x * y, list(tensorMeta.shape), 1
        #     )

        # 为每个子图创建一个FuncOp节点，并将这些节点添加到主图中。
        # Adding FuncOp nodes for each subgraph
        inputs0 = self._graph._inputs
        split_group = []
        param_size_group = []
        for i, subgraph_name in enumerate(self._subgraphs.keys()):
            main_graph_name = "forward{}".format(i)
            current_param_info = {} # 存储参数索引和分割方式
            main_graph = Graph(
                [],
                [],
                self._graph._ops_registry,
                main_graph_name,
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
            if outputs is None:
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
            ph_count : int = 0
            # 记录子图中是否有权重矩阵被分割
            issplit = False
            current_param_info["params"] = []
            current_param_info["total_partitions"] = 1
            split_group.append(1)
            for node in self._graph.body:
                if isinstance(node, PlaceholderOp) :
                    if node in self._subgraphs_inputs[subgraph_name]:
                        if(len(self._graph._fake_params) > (ph_count)):
                            main_graph._fake_params.append(self._graph._fake_params[ph_count])
                            if node.name in self._paral_op_shape.keys():
                                node._newshape = self._paral_op_shape[node.name]
                                main_graph._fake_params[-1]['shape'] = torch.Size(node._newshape)
                                current_param_info["params"].append(
                                    {"index": ph_count, "split_degree": node._newshape}
                                )
                                issplit = True
                            else: 
                                current_param_info["params"].append(
                                    {"index": ph_count, "split_degree": []}
                                )
                        main_graph.add_node(node)
                    ph_count += 1
            param_size_group.append(self.get_pack_params_size(main_graph._fake_params))

            if issplit: 
                current_param_info["total_partitions"] = self._parallelism
                split_group[-1] = self._parallelism
            self._subgraph_param_info[subgraph_name] = current_param_info

            # Identify inputs for each subgraph
            maingraph_input = inputs0
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
                    main_graph._body.append(placeholder_node)
            # if issplit: 
            #     current_param_info["total_partitions"] = self._parallelism
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
            if outputs is None:
                for output in self._subgraphs_outputs[subgraph_name]:
                    call_node.tensor_meta["dtype"].append(
                        self._graph.node_table[output.name].tensor_meta["dtype"]
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
