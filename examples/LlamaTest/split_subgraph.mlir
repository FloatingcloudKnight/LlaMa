module {
  func.func @subgraph196(%arg0: tensor<1x40x4096xf32>) -> (tensor<1x5x4096xf32>, tensor<1x5x4096xf32>, tensor<1x5x4096xf32>, tensor<1x5x4096xf32>, tensor<1x5x4096xf32>, tensor<1x5x4096xf32>, tensor<1x5x4096xf32>, tensor<1x5x4096xf32>) {
    %extract_slice = tensor.extract_slice %arg0[0, 0, 0] [1, 5, 4096] [1, 1, 1] : tensor<1x40x4096xf32> to tensor<1x5x4096xf32>
    %extract_slice_0 = tensor.extract_slice %arg0[0, 5, 0] [1, 5, 4096] [1, 1, 1] : tensor<1x40x4096xf32> to tensor<1x5x4096xf32>
    %extract_slice_1 = tensor.extract_slice %arg0[0, 10, 0] [1, 5, 4096] [1, 1, 1] : tensor<1x40x4096xf32> to tensor<1x5x4096xf32>
    %extract_slice_2 = tensor.extract_slice %arg0[0, 15, 0] [1, 5, 4096] [1, 1, 1] : tensor<1x40x4096xf32> to tensor<1x5x4096xf32>
    %extract_slice_3 = tensor.extract_slice %arg0[0, 20, 0] [1, 5, 4096] [1, 1, 1] : tensor<1x40x4096xf32> to tensor<1x5x4096xf32>
    %extract_slice_4 = tensor.extract_slice %arg0[0, 25, 0] [1, 5, 4096] [1, 1, 1] : tensor<1x40x4096xf32> to tensor<1x5x4096xf32>
    %extract_slice_5 = tensor.extract_slice %arg0[0, 30, 0] [1, 5, 4096] [1, 1, 1] : tensor<1x40x4096xf32> to tensor<1x5x4096xf32>
    %extract_slice_6 = tensor.extract_slice %arg0[0, 35, 0] [1, 5, 4096] [1, 1, 1] : tensor<1x40x4096xf32> to tensor<1x5x4096xf32>
    return %extract_slice, %extract_slice_0, %extract_slice_1, %extract_slice_2, %extract_slice_3, %extract_slice_4, %extract_slice_5, %extract_slice_6 : tensor<1x5x4096xf32>, tensor<1x5x4096xf32>, tensor<1x5x4096xf32>, tensor<1x5x4096xf32>, tensor<1x5x4096xf32>, tensor<1x5x4096xf32>, tensor<1x5x4096xf32>, tensor<1x5x4096xf32>
  }
}