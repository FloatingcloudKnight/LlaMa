#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module {
  // subgraph5的输入:Y1, Y--->%arg0, %arg1
  // O = Y + MLP(Y1)
  func.func @subgraph5(%arg0: tensor<1x40x4096xf32>, %arg1: tensor<1x40x4096xf32>, %arg9: tensor<11008x4096xf32>, %arg10: tensor<11008x4096xf32>, %arg11: tensor<4096x11008xf32>) -> tensor<1x40x4096xf32> {
    %144 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %145 = tosa.transpose %arg9, %144 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %146 = tosa.reshape %arg0 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_28 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %147 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%146, %145 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_28 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %148 = tosa.reshape %147 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %149 = tosa.sigmoid %148 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %150 = tosa.mul %148, %149 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %151 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %152 = tosa.transpose %arg10, %151 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %153 = tosa.reshape %arg0 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_29 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %154 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%153, %152 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_29 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %155 = tosa.reshape %154 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %156 = tosa.mul %150, %155 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %157 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %158 = tosa.transpose %arg11, %157 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %159 = tosa.reshape %156 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_30 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %160 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%159, %158 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_30 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %161 = tosa.reshape %160 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %162 = tosa.add %arg1, %161 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    return %162 : tensor<1x40x4096xf32>
  }
}