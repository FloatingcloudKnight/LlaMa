module {
  func.func private @subgraph196(memref<1x40x4096xf32, strided<[163840, 4096, 1], offset: ?>>) -> (memref<1x10x4096xf32>, memref<1x10x4096xf32>, memref<1x10x4096xf32>, memref<1x10x4096xf32>)
  func.func @forward196(%arg0: memref<1x40x4096xf32>) -> (memref<1x10x4096xf32>, memref<1x10x4096xf32>, memref<1x10x4096xf32>, memref<1x10x4096xf32>) {
    %cast = memref.cast %arg0 : memref<1x40x4096xf32> to memref<1x40x4096xf32, strided<[163840, 4096, 1], offset: ?>>
    %0:4 = call @subgraph196(%cast) : (memref<1x40x4096xf32, strided<[163840, 4096, 1], offset: ?>>) -> (memref<1x10x4096xf32>, memref<1x10x4096xf32>, memref<1x10x4096xf32>, memref<1x10x4096xf32>)
    return %0#0, %0#1, %0#2, %0#3 : memref<1x10x4096xf32>, memref<1x10x4096xf32>, memref<1x10x4096xf32>, memref<1x10x4096xf32>
  }
}