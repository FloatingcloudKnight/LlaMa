module {
  func.func @subgraph195(%arg0: tensor<40x4096xf32>, %arg1: tensor<40x4096xf32>, %arg2: tensor<40x4096xf32>, %arg3: tensor<40x4096xf32>, %arg4: tensor<40x4096xf32>, %arg5: tensor<40x4096xf32>, %arg6: tensor<40x4096xf32>, %arg7: tensor<40x4096xf32>) -> (tensor<5x4096xf32>, tensor<5x4096xf32>, tensor<5x4096xf32>, tensor<5x4096xf32>, tensor<5x4096xf32>, tensor<5x4096xf32>, tensor<5x4096xf32>, tensor<5x4096xf32>) {
    %0 = tosa.add %arg0, %arg1 : (tensor<40x4096xf32>, tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1 = tosa.add %arg2, %arg3 : (tensor<40x4096xf32>, tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2 = tosa.add %arg4, %arg5 : (tensor<40x4096xf32>, tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3 = tosa.add %arg6, %arg7 : (tensor<40x4096xf32>, tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %4 = tosa.add %0, %1 : (tensor<40x4096xf32>, tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %5 = tosa.add %2, %3 : (tensor<40x4096xf32>, tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %6 = tosa.add %4, %5 : (tensor<40x4096xf32>, tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %extract_slice = tensor.extract_slice %6[0, 0] [5, 4096] [1, 1] : tensor<40x4096xf32> to tensor<5x4096xf32>
    %extract_slice_0 = tensor.extract_slice %6[5, 0] [5, 4096] [1, 1] : tensor<40x4096xf32> to tensor<5x4096xf32>
    %extract_slice_1 = tensor.extract_slice %6[10, 0] [5, 4096] [1, 1] : tensor<40x4096xf32> to tensor<5x4096xf32>
    %extract_slice_2 = tensor.extract_slice %6[15, 0] [5, 4096] [1, 1] : tensor<40x4096xf32> to tensor<5x4096xf32>
    %extract_slice_3 = tensor.extract_slice %6[20, 0] [5, 4096] [1, 1] : tensor<40x4096xf32> to tensor<5x4096xf32>
    %extract_slice_4 = tensor.extract_slice %6[25, 0] [5, 4096] [1, 1] : tensor<40x4096xf32> to tensor<5x4096xf32>
    %extract_slice_5 = tensor.extract_slice %6[30, 0] [5, 4096] [1, 1] : tensor<40x4096xf32> to tensor<5x4096xf32>
    %extract_slice_6 = tensor.extract_slice %6[35, 0] [5, 4096] [1, 1] : tensor<40x4096xf32> to tensor<5x4096xf32>
    return %extract_slice, %extract_slice_0, %extract_slice_1, %extract_slice_2, %extract_slice_3, %extract_slice_4, %extract_slice_5, %extract_slice_6 : tensor<5x4096xf32>, tensor<5x4096xf32>, tensor<5x4096xf32>, tensor<5x4096xf32>, tensor<5x4096xf32>, tensor<5x4096xf32>, tensor<5x4096xf32>, tensor<5x4096xf32>
  }
}