module {
  func.func @subgraph5_decode(%arg0: tensor<4480x1536xf32>, %arg1: tensor<1x1x1536xf32>, %arg2: tensor<4480x1536xf32>, %arg3: tensor<1536x4480xf32>) -> tensor<1x1536xf32> {
    %0 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1 = tosa.transpose %arg0, %0 : (tensor<4480x1536xf32>, tensor<2xi32>) -> tensor<1536x4480xf32>
    %2 = tosa.reshape %arg1 {new_shape = array<i64: 1, 1536>} : (tensor<1x1x1536xf32>) -> tensor<1x1536xf32>
    %cst = arith.constant dense<0.000000e+00> : tensor<1x4480xf32>
    %3 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2, %1 : tensor<1x1536xf32>, tensor<1536x4480xf32>) outs(%cst : tensor<1x4480xf32>) -> tensor<1x4480xf32>
    %4 = tosa.reshape %3 {new_shape = array<i64: 1, 1, 4480>} : (tensor<1x4480xf32>) -> tensor<1x1x4480xf32>
    %5 = tosa.sigmoid %4 : (tensor<1x1x4480xf32>) -> tensor<1x1x4480xf32>
    %6 = tosa.mul %4, %5 : (tensor<1x1x4480xf32>, tensor<1x1x4480xf32>) -> tensor<1x1x4480xf32>
    %7 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %8 = tosa.transpose %arg2, %7 : (tensor<4480x1536xf32>, tensor<2xi32>) -> tensor<1536x4480xf32>
    %9 = tosa.reshape %arg1 {new_shape = array<i64: 1, 1536>} : (tensor<1x1x1536xf32>) -> tensor<1x1536xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<1x4480xf32>
    %10 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%9, %8 : tensor<1x1536xf32>, tensor<1536x4480xf32>) outs(%cst_0 : tensor<1x4480xf32>) -> tensor<1x4480xf32>
    %11 = tosa.reshape %10 {new_shape = array<i64: 1, 1, 4480>} : (tensor<1x4480xf32>) -> tensor<1x1x4480xf32>
    %12 = tosa.mul %6, %11 : (tensor<1x1x4480xf32>, tensor<1x1x4480xf32>) -> tensor<1x1x4480xf32>
    %13 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %14 = tosa.transpose %arg3, %13 : (tensor<1536x4480xf32>, tensor<2xi32>) -> tensor<4480x1536xf32>
    %15 = tosa.reshape %12 {new_shape = array<i64: 1, 4480>} : (tensor<1x1x4480xf32>) -> tensor<1x4480xf32>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<1x1536xf32>
    %16 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%15, %14 : tensor<1x4480xf32>, tensor<4480x1536xf32>) outs(%cst_1 : tensor<1x1536xf32>) -> tensor<1x1536xf32>
    return %16 : tensor<1x1536xf32>
  }
}

