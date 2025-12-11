module {
  func.func @subgraph3_prefill(%arg0: tensor<512x1536xf32>, %arg1: tensor<1x512x1536xf32>) -> tensor<1x512x1536xf32> {
    %0 = tosa.reshape %arg0 {new_shape = array<i64: 1, 512, 1536>} : (tensor<512x1536xf32>) -> tensor<1x512x1536xf32>
    %1 = tosa.add %arg1, %0 : (tensor<1x512x1536xf32>, tensor<1x512x1536xf32>) -> tensor<1x512x1536xf32>
    return %1 : tensor<1x512x1536xf32>
  }
}

