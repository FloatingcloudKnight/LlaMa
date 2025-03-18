#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module {
  // subgraph1的输入: X--->%arg0
  // X1 = RMSNorm(X)
  func.func @subgraph1(%arg0: tensor<1x40x4096xf32>, %arg3: tensor<4096xf32>) -> tensor<1x40x4096xf32> {
    %48 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32 = arith.constant 2 : i32
    %49 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x40x4096xf32>) outs(%48 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3745 = math.fpowi %in, %c2_i32 : f32, i32
      linalg.yield %3745 : f32
    } -> tensor<1x40x4096xf32>
    %50 = tosa.reduce_sum %49 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %51 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %52 = tosa.reciprocal %51 : (tensor<1xf32>) -> tensor<1xf32>
    %53 = tosa.mul %52, %50 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %54 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %55 = tosa.add %53, %54 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %56 = tosa.rsqrt %55 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %57 = tosa.mul %arg0, %56 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %58 = tosa.reshape %arg3 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %59 = tosa.mul %58, %57 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    return %59 : tensor<1x40x4096xf32>
  }
}