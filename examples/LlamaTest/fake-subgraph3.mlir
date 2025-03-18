#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module {
  // subgraph3的输入: X, MHA(X1)--->%arg0, %arg1
  // Y = X + MHA(X1)
  func.func @subgraph3(%arg0: tensor<1x40x4096xf32>, %arg1: tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32> {
    %131 = tosa.add %arg0, %arg1 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    return %131 : tensor<1x40x4096xf32>
  }
}