module {
  func.func @subgraph195(%arg0: tensor<40x4096xf32>, %arg1: tensor<40x4096xf32>, %arg2: tensor<40x4096xf32>, %arg3: tensor<40x4096xf32>) -> (tensor<10x4096xf32>, tensor<10x4096xf32>, tensor<10x4096xf32>, tensor<10x4096xf32>) {
    %0 = tosa.add %arg0, %arg1 : (tensor<40x4096xf32>, tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1 = tosa.add %arg2, %arg3 : (tensor<40x4096xf32>, tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2 = tosa.add %0, %1 : (tensor<40x4096xf32>, tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %extract_slice = tensor.extract_slice %2[0, 0] [10, 4096] [1, 1] : tensor<40x4096xf32> to tensor<10x4096xf32>
    %extract_slice_0 = tensor.extract_slice %2[10, 0] [10, 4096] [1, 1] : tensor<40x4096xf32> to tensor<10x4096xf32>
    %extract_slice_1 = tensor.extract_slice %2[20, 0] [10, 4096] [1, 1] : tensor<40x4096xf32> to tensor<10x4096xf32>
    %extract_slice_2 = tensor.extract_slice %2[30, 0] [10, 4096] [1, 1] : tensor<40x4096xf32> to tensor<10x4096xf32>
    return %extract_slice, %extract_slice_0, %extract_slice_1, %extract_slice_2 : tensor<10x4096xf32>, tensor<10x4096xf32>, tensor<10x4096xf32>, tensor<10x4096xf32>
  }
}