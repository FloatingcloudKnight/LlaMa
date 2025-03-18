#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module {
  // subgraph4的输入: Y--->%arg0
  // Y1 = RMSNorm(Y)
  func.func @subgraph4(%arg0: tensor<1x40x4096xf32>, %arg1: tensor<1x40x4096xf32>, %arg8: tensor<4096xf32>) -> tensor<1x40x4096xf32> {
    %132 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_27 = arith.constant 2 : i32
    %133 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x40x4096xf32>) outs(%132 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3745 = math.fpowi %in, %c2_i32_27 : f32, i32
      linalg.yield %3745 : f32
    } -> tensor<1x40x4096xf32>
    %134 = tosa.reduce_sum %133 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %135 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %136 = tosa.reciprocal %135 : (tensor<1xf32>) -> tensor<1xf32>
    %137 = tosa.mul %136, %134 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %138 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %139 = tosa.add %137, %138 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %140 = tosa.rsqrt %139 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %141 = tosa.mul %arg0, %140 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %142 = tosa.reshape %arg8 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %143 = tosa.mul %142, %141 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    return %143 : tensor<1x40x4096xf32>
  }
}