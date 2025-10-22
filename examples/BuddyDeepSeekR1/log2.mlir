#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, 0)>
#map2 = affine_map<(d0, d1, d2) -> (d0, 0, d2)>
#map3 = affine_map<(d0, d1) -> (0, d1)>
#map4 = affine_map<(d0, d1) -> (d0, d1)>
module {
  memref.global "private" constant @__constant_1024x1536xf32 : memref<1024x1536xf32> = dense<0.000000e+00> {alignment = 64 : i64}
  memref.global "private" constant @__constant_2xi32 : memref<2xi32> = dense<[1, 0]> {alignment = 64 : i64}
  func.func @subgraph0(%arg0: memref<1x1024x1536xf32, strided<[?, ?, ?], offset: ?>>, %arg1: memref<1x1024x1xf32, strided<[?, ?, ?], offset: ?>>, %arg2: memref<1536xf32, strided<[?], offset: ?>>, %arg3: memref<1536x1536xf32, strided<[?, ?], offset: ?>>, %arg4: memref<1536xf32, strided<[?], offset: ?>>) -> memref<1024x1536xf32> {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1024x1536xf32>
    linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : memref<1x1024x1536xf32, strided<[?, ?, ?], offset: ?>>, memref<1x1024x1xf32, strided<[?, ?, ?], offset: ?>>) outs(%alloc : memref<1x1024x1536xf32>) {
    ^bb0(%in: f32, %in_7: f32, %out: f32):
      %2 = arith.mulf %in, %in_7 : f32
      linalg.yield %2 : f32
    }
    %expand_shape = memref.expand_shape %arg2 [[0, 1, 2]] output_shape [1, 1, 1536] : memref<1536xf32, strided<[?], offset: ?>> into memref<1x1x1536xf32, strided<[?, ?, ?], offset: ?>>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<1x1024x1536xf32>
    linalg.generic {indexing_maps = [#map2, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expand_shape, %alloc : memref<1x1x1536xf32, strided<[?, ?, ?], offset: ?>>, memref<1x1024x1536xf32>) outs(%alloc_0 : memref<1x1024x1536xf32>) {
    ^bb0(%in: f32, %in_7: f32, %out: f32):
      %2 = arith.mulf %in, %in_7 : f32
      linalg.yield %2 : f32
    }
    %collapse_shape = memref.collapse_shape %alloc_0 [[0, 1], [2]] : memref<1x1024x1536xf32> into memref<1024x1536xf32>
    %0 = memref.get_global @__constant_2xi32 : memref<2xi32>
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<1536x1536xf32>
    linalg.transpose ins(%arg3 : memref<1536x1536xf32, strided<[?, ?], offset: ?>>) outs(%alloc_1 : memref<1536x1536xf32>) permutation = [1, 0] 
    %1 = memref.get_global @__constant_1024x1536xf32 : memref<1024x1536xf32>
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<1024x1536xf32>
    memref.copy %1, %alloc_2 : memref<1024x1536xf32> to memref<1024x1536xf32>
    linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%collapse_shape, %alloc_1 : memref<1024x1536xf32>, memref<1536x1536xf32>) outs(%alloc_2 : memref<1024x1536xf32>)
    %expand_shape_3 = memref.expand_shape %arg4 [[0, 1]] output_shape [1, 1536] : memref<1536xf32, strided<[?], offset: ?>> into memref<1x1536xf32, strided<[?, ?], offset: ?>>
    %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<1024x1536xf32>
    linalg.generic {indexing_maps = [#map3, #map4, #map4], iterator_types = ["parallel", "parallel"]} ins(%expand_shape_3, %alloc_2 : memref<1x1536xf32, strided<[?, ?], offset: ?>>, memref<1024x1536xf32>) outs(%alloc_4 : memref<1024x1536xf32>) {
    ^bb0(%in: f32, %in_7: f32, %out: f32):
      %2 = arith.addf %in, %in_7 : f32
      linalg.yield %2 : f32
    }
    %expand_shape_5 = memref.expand_shape %alloc_4 [[0, 1], [2]] output_shape [1, 1024, 1536] : memref<1024x1536xf32> into memref<1x1024x1536xf32>
    %collapse_shape_6 = memref.collapse_shape %alloc_0 [[0, 1], [2]] : memref<1x1024x1536xf32> into memref<1024x1536xf32>
    %cast = memref.cast %collapse_shape_6 : memref<1024x1536xf32> to memref<1024x1536xf32, strided<[?, ?], offset: ?>>
    return %collapse_shape_6 : memref<1024x1536xf32>
  }
}

