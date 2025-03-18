module {
  func.func private @subgraph3(memref<1x40x4096xf32, strided<[163840, 4096, 1], offset: ?>>, memref<1x40x4096xf32, strided<[163840, 4096, 1], offset: ?>>) -> memref<1x40x4096xf32>
  func.func @forward3(%arg0: memref<6738415680xf32>, %arg1: memref<1x40x4096xf32>, %arg2: memref<1x40x4096xf32>) -> memref<1x40x4096xf32> {
    %cast = memref.cast %arg1 : memref<1x40x4096xf32> to memref<1x40x4096xf32, strided<[163840, 4096, 1], offset: ?>>
    %cast0 = memref.cast %arg2 : memref<1x40x4096xf32> to memref<1x40x4096xf32, strided<[163840, 4096, 1], offset: ?>>
    %0 = call @subgraph3(%cast, %cast0) : (memref<1x40x4096xf32, strided<[163840, 4096, 1], offset: ?>>, memref<1x40x4096xf32, strided<[163840, 4096, 1], offset: ?>>) -> memref<1x40x4096xf32>
    return %0 : memref<1x40x4096xf32>
  }
}

