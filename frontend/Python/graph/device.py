# ===- device.py ---------------------------------------------------------------
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
# Device class that encapsulates memory and compute resource states.
#
# ===---------------------------------------------------------------------------

from typing import Dict, Optional, Tuple, List, Union
import numpy as np

class Device:
  """
  Stores information about the device, records the current resource usage of the device, and performs resource allocation and resource recovery.
  """
  def __init__(self, device_id: str, memory_size: float, compute_size: float):
    self.deviceId = device_id
    self.memoryAvailable = memory_size
    self.computeAvailable = compute_size
    # module_id -> { "memory": float, "compute": float}
    self.allocatedModule: Dict[str, Dict[str, float]] = {}

  def canAssigned(self, module_id: str, memory: float, compute: float) -> bool:
    """
    检查当前设备能否满足最小的资源需求
    """    
    return (
            self.memory.available >= memory and
            self.compute.available >= compute
        )

  def allocate_memory(self, module_id: str, needMemory: float, needCompute: float) -> bool:
    """ 分配指定模块的内存和计算资源
    如果当前设备无法满足最小的资源需求则返回False
    """
    
    if not self.canAssigned(needMemory, needCompute):
      return False
    
    self.memoryAvailable -= needMemory
    self.computeAvailable -= needCompute
    self.allocatedModule[module_id] = {"memory": needMemory, "compute": needCompute}

    return True

  def deallocate_memory(self, module_id: str) -> None:
    """
    释放指定模块的内存和计算资源
    """
    if module_id in self.allocatedModule:
      allocated = self.allocatedModule.pop(module_id)
      self.memoryAvailable += allocated["memory"]
      self.computeAvailable += allocated["compute"]

  def record_usage(self) -> Dict[str, Union[str, float, List[str]]]:
    """
    记录当前设备的资源使用情况
    """
    allocatedModules = self.allocatedModule.keys()
    return {
        "deviceId": self.deviceId,
        "memoryAvailable": self.memoryAvailable,
        "computeAvailable": self.computeAvailable,
        "allocatedModule": allocatedModules
    }