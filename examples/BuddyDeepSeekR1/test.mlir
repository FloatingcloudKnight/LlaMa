#map = affine_map<(d0, d1, d2) -> (d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module {
  func.func @subgraph0(%3: tensor<1x1024x1536xf32>, %63: tensor<1x1024x1xf32>, %arg4: tensor<1536xf32>, %arg5: tensor<1536x1536xf32>, %arg6: tensor<1536xf32>) -> tensor<1024x1536xf32> {    
    %64 = tosa.mul %3, %63 : (tensor<1x1024x1536xf32>, tensor<1x1024x1xf32>) -> tensor<1x1024x1536xf32>
    %65 = tosa.reshape %arg4 {new_shape = array<i64: 1, 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1x1536xf32>
    %66 = tosa.mul %65, %64 : (tensor<1x1x1536xf32>, tensor<1x1024x1536xf32>) -> tensor<1x1024x1536xf32>
    %67 = tosa.reshape %66 {new_shape = array<i64: 1024, 1536>} : (tensor<1x1024x1536xf32>) -> tensor<1024x1536xf32>
    %68 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %69 = tosa.transpose %arg5, %68 : (tensor<1536x1536xf32>, tensor<2xi32>) -> tensor<1536x1536xf32>
    %cst_27 = arith.constant dense<0.000000e+00> : tensor<1024x1536xf32>
    %70 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%67, %69 : tensor<1024x1536xf32>, tensor<1536x1536xf32>) outs(%cst_27 : tensor<1024x1536xf32>) -> tensor<1024x1536xf32>
    %71 = tosa.reshape %arg6 {new_shape = array<i64: 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1536xf32>
    %72 = tosa.add %71, %70 : (tensor<1x1536xf32>, tensor<1024x1536xf32>) -> tensor<1024x1536xf32>
    %73 = tosa.reshape %72 {new_shape = array<i64: 1, 1024, 1536>} : (tensor<1024x1536xf32>) -> tensor<1x1024x1536xf32>
    %74 = tosa.reshape %66 {new_shape = array<i64: 1024, 1536>} : (tensor<1x1024x1536xf32>) -> tensor<1024x1536xf32>
    return %74 : tensor<1024x1536xf32>
  }
}