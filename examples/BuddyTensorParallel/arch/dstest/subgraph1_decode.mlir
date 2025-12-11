#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module {
  func.func @subgraph1_decode(%arg0: tensor<1x1x1536xf32>, %arg1: tensor<1536xf32>) -> tensor<1x1x1536xf32> {
    %0 = tensor.empty() : tensor<1x1x1536xf32>
    %c2_i32 = arith.constant 2 : i32
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x1x1536xf32>) outs(%0 : tensor<1x1x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %13 = math.fpowi %in, %c2_i32 : f32, i32
      linalg.yield %13 : f32
    } -> tensor<1x1x1536xf32>
    %2 = tosa.reduce_sum %1 {axis = 2 : i32} : (tensor<1x1x1536xf32>) -> tensor<1x1x1xf32>
    %3 = "tosa.const"() <{value = dense<1.536000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %4 = tosa.reciprocal %3 : (tensor<1xf32>) -> tensor<1xf32>
    %5 = tosa.reshape %4 {new_shape = array<i64: 1, 1, 1>} : (tensor<1xf32>) -> tensor<1x1x1xf32>
    %6 = tosa.mul %5, %2 : (tensor<1x1x1xf32>, tensor<1x1x1xf32>) -> tensor<1x1x1xf32>
    %7 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x1x1xf32>}> : () -> tensor<1x1x1xf32>
    %8 = tosa.add %6, %7 : (tensor<1x1x1xf32>, tensor<1x1x1xf32>) -> tensor<1x1x1xf32>
    %9 = tosa.rsqrt %8 : (tensor<1x1x1xf32>) -> tensor<1x1x1xf32>
    %10 = tosa.mul %arg0, %9 : (tensor<1x1x1536xf32>, tensor<1x1x1xf32>) -> tensor<1x1x1536xf32>
    %11 = tosa.reshape %arg1 {new_shape = array<i64: 1, 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1x1536xf32>
    %12 = tosa.mul %11, %10 : (tensor<1x1x1536xf32>, tensor<1x1x1536xf32>) -> tensor<1x1x1536xf32>
    return %12 : tensor<1x1x1536xf32>
  }
}

