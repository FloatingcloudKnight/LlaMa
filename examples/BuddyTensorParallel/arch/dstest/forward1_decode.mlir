module {
  func.func private @subgraph1_decode(memref<1x1x1536xf32, strided<[1536, 1536, 1], offset: ?>>, memref<1536xf32, strided<[1], offset: ?>>) -> memref<1x1x1536xf32>
  func.func @forward1_decode(%arg0: memref<1536xf32>, %arg1: memref<1x1x1536xf32>) -> memref<1x1x1536xf32> {
    %subview = memref.subview %arg0[0] [1536] [1] : memref<1536xf32> to memref<1536xf32>
    %cast = memref.cast %arg1 : memref<1x1x1536xf32> to memref<1x1x1536xf32, strided<[1536, 1536, 1], offset: ?>>
    %cast_0 = memref.cast %subview : memref<1536xf32> to memref<1536xf32, strided<[1], offset: ?>>
    %0 = call @subgraph1_decode(%cast, %cast_0) : (memref<1x1x1536xf32, strided<[1536, 1536, 1], offset: ?>>, memref<1536xf32, strided<[1], offset: ?>>) -> memref<1x1x1536xf32>
    return %0 : memref<1x1x1536xf32>
  }
}

