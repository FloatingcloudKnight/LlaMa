module {
  func.func @subgraph194(%arg0: tensor<1x10x4096xf32>, %arg1: tensor<1x10x4096xf32>, %arg2: tensor<1x10x4096xf32>, %arg3: tensor<1x10x4096xf32>) -> tensor<1x40x4096xf32> {
    %0 = tensor.empty() : tensor<1x40x4096xf32>
    %inserted_slice = tensor.insert_slice %arg0 into %0[0, 0, 0] [1, 10, 4096] [1, 1, 1] : tensor<1x10x4096xf32> into tensor<1x40x4096xf32>
    %inserted_slice_0 = tensor.insert_slice %arg1 into %inserted_slice[0, 10, 0] [1, 10, 4096] [1, 1, 1] : tensor<1x10x4096xf32> into tensor<1x40x4096xf32>
    %inserted_slice_1 = tensor.insert_slice %arg2 into %inserted_slice_0[0, 20, 0] [1, 10, 4096] [1, 1, 1] : tensor<1x10x4096xf32> into tensor<1x40x4096xf32>
    %inserted_slice_2 = tensor.insert_slice %arg3 into %inserted_slice_1[0, 30, 0] [1, 10, 4096] [1, 1, 1] : tensor<1x10x4096xf32> into tensor<1x40x4096xf32>
    return %inserted_slice_2 : tensor<1x40x4096xf32>
  }
}