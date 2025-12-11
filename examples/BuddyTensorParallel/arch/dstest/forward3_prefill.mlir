module {
  func.func private @subgraph3_prefill(memref<512x1536xf32, strided<[1536, 1], offset: ?>>, memref<1x512x1536xf32, strided<[786432, 1536, 1], offset: ?>>) -> memref<1x512x1536xf32>
  func.func @forward3_prefill(%arg0: memref<512x1536xf32>, %arg1: memref<1x512x1536xf32>) -> memref<1x512x1536xf32> {
    %cast = memref.cast %arg0 : memref<512x1536xf32> to memref<512x1536xf32, strided<[1536, 1], offset: ?>>
    %cast_0 = memref.cast %arg1 : memref<1x512x1536xf32> to memref<1x512x1536xf32, strided<[786432, 1536, 1], offset: ?>>
    %0 = call @subgraph3_prefill(%cast, %cast_0) : (memref<512x1536xf32, strided<[1536, 1], offset: ?>>, memref<1x512x1536xf32, strided<[786432, 1536, 1], offset: ?>>) -> memref<1x512x1536xf32>
    return %0 : memref<1x512x1536xf32>
  }
}

