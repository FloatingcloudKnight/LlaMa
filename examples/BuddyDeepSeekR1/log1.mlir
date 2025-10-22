#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, 0)>
#map2 = affine_map<(d0, d1, d2) -> (d0, 0, d2)>
module {
  func.func @subgraph0(%arg0: memref<1x1024x1536xf32, strided<[?, ?, ?], offset: ?>>, %arg1: memref<1x1024x1xf32, strided<[?, ?, ?], offset: ?>>, %arg2: memref<1536xf32, strided<[?], offset: ?>>, %arg3: memref<1536x1536xf32, strided<[?, ?], offset: ?>>, %arg4: memref<1536xf32, strided<[?], offset: ?>>) -> memref<1024x1536xf32> {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1024x1536xf32>
    linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : memref<1x1024x1536xf32, strided<[?, ?, ?], offset: ?>>, memref<1x1024x1xf32, strided<[?, ?, ?], offset: ?>>) outs(%alloc : memref<1x1024x1536xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %0 = arith.mulf %in, %in_1 : f32
      linalg.yield %0 : f32
    }
    %expand_shape = memref.expand_shape %arg2 [[0, 1, 2]] output_shape [1, 1, 1536] : memref<1536xf32, strided<[?], offset: ?>> into memref<1x1x1536xf32, strided<[?, ?, ?], offset: ?>>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<1x1024x1536xf32>
    linalg.generic {indexing_maps = [#map2, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expand_shape, %alloc : memref<1x1x1536xf32, strided<[?, ?, ?], offset: ?>>, memref<1x1024x1536xf32>) outs(%alloc_0 : memref<1x1024x1536xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %0 = arith.mulf %in, %in_1 : f32
      linalg.yield %0 : f32
    }
    %collapse_shape = memref.collapse_shape %alloc_0 [[0, 1], [2]] : memref<1x1024x1536xf32> into memref<1024x1536xf32>
    %cast = memref.cast %collapse_shape : memref<1024x1536xf32> to memref<1024x1536xf32, strided<[?, ?], offset: ?>>
    return %collapse_shape : memref<1024x1536xf32>
  }
}

