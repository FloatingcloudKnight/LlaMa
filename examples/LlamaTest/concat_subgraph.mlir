module {
  func.func @subgraph194(%arg0: tensor<1x5x4096xf32>, %arg1: tensor<1x5x4096xf32>, %arg2: tensor<1x5x4096xf32>, %arg3: tensor<1x5x4096xf32>, %arg4: tensor<1x5x4096xf32>, %arg5: tensor<1x5x4096xf32>, %arg6: tensor<1x5x4096xf32>, %arg7: tensor<1x5x4096xf32>) -> tensor<1x40x4096xf32> {
    %0 = tensor.empty() : tensor<1x40x4096xf32>
    %inserted_slice = tensor.insert_slice %arg0 into %0[0, 0, 0] [1, 5, 4096] [1, 1, 1] : tensor<1x5x4096xf32> into tensor<1x40x4096xf32>
    %inserted_slice_0 = tensor.insert_slice %arg1 into %inserted_slice[0, 5, 0] [1, 5, 4096] [1, 1, 1] : tensor<1x5x4096xf32> into tensor<1x40x4096xf32>
    %inserted_slice_1 = tensor.insert_slice %arg2 into %inserted_slice_0[0, 10, 0] [1, 5, 4096] [1, 1, 1] : tensor<1x5x4096xf32> into tensor<1x40x4096xf32>
    %inserted_slice_2 = tensor.insert_slice %arg3 into %inserted_slice_1[0, 15, 0] [1, 5, 4096] [1, 1, 1] : tensor<1x5x4096xf32> into tensor<1x40x4096xf32>
    %inserted_slice_3 = tensor.insert_slice %arg4 into %inserted_slice_2[0, 20, 0] [1, 5, 4096] [1, 1, 1] : tensor<1x5x4096xf32> into tensor<1x40x4096xf32>
    %inserted_slice_4 = tensor.insert_slice %arg5 into %inserted_slice_3[0, 25, 0] [1, 5, 4096] [1, 1, 1] : tensor<1x5x4096xf32> into tensor<1x40x4096xf32>
    %inserted_slice_5 = tensor.insert_slice %arg6 into %inserted_slice_4[0, 30, 0] [1, 5, 4096] [1, 1, 1] : tensor<1x5x4096xf32> into tensor<1x40x4096xf32>
    %inserted_slice_6 = tensor.insert_slice %arg7 into %inserted_slice_5[0, 35, 0] [1, 5, 4096] [1, 1, 1] : tensor<1x5x4096xf32> into tensor<1x40x4096xf32>
    return %inserted_slice_6 : tensor<1x40x4096xf32>
  }
}