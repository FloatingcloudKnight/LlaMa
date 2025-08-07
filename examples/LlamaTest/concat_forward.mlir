module {
  func.func private @subgraph194(memref<1x10x4096xf32, strided<[40960, 4096, 1], offset: ?>>, memref<1x10x4096xf32, strided<[40960, 4096, 1], offset: ?>>, memref<1x10x4096xf32, strided<[40960, 4096, 1], offset: ?>>, memref<1x10x4096xf32, strided<[40960, 4096, 1], offset: ?>>) -> memref<1x40x4096xf32>
  func.func @forward194(%arg0: memref<1x10x4096xf32>, %arg1: memref<1x10x4096xf32>, %arg2: memref<1x10x4096xf32>, %arg3: memref<1x10x4096xf32>) -> memref<1x40x4096xf32> {
    %cast = memref.cast %arg0 : memref<1x10x4096xf32> to memref<1x10x4096xf32, strided<[40960, 4096, 1], offset: ?>>
    %cast_0 = memref.cast %arg1 : memref<1x10x4096xf32> to memref<1x10x4096xf32, strided<[40960, 4096, 1], offset: ?>>
    %cast_1 = memref.cast %arg0 : memref<1x10x4096xf32> to memref<1x10x4096xf32, strided<[40960, 4096, 1], offset: ?>>
    %cast_2 = memref.cast %arg1 : memref<1x10x4096xf32> to memref<1x10x4096xf32, strided<[40960, 4096, 1], offset: ?>>
    %0 = call @subgraph194(%cast, %cast_0, %cast_1, %cast_2) : (memref<1x10x4096xf32, strided<[40960, 4096, 1], offset: ?>>, memref<1x10x4096xf32, strided<[40960, 4096, 1], offset: ?>>, memref<1x10x4096xf32, strided<[40960, 4096, 1], offset: ?>>, memref<1x10x4096xf32, strided<[40960, 4096, 1], offset: ?>>) -> memref<1x40x4096xf32>
    return %0 : memref<1x40x4096xf32>
  }
}