module {
  func.func private @subgraph0(memref<32000x4096xf32, strided<[4096, 1], offset: ?>>, memref<1x40xi64, strided<[40, 1], offset: ?>>) -> memref<1x40x4096xf32>
  func.func @forward0(%arg0: memref<6738415680xf32>, %arg1: memref<1x40xi64>) -> memref<1x40x4096xf32> {
    %subview = memref.subview %arg0[0] [131072000] [1] : memref<6738415680xf32> to memref<131072000xf32>
    %expand_shape = memref.expand_shape %subview [[0, 1]] : memref<131072000xf32> into memref<32000x4096xf32>
    %cast = memref.cast %expand_shape : memref<32000x4096xf32> to memref<32000x4096xf32, strided<[4096, 1], offset: ?>>
    %cast_516 = memref.cast %arg1 : memref<1x40xi64> to memref<1x40xi64, strided<[40, 1], offset: ?>>
    %0 = call @subgraph0(%cast, %cast_516) : (memref<32000x4096xf32, strided<[4096, 1], offset: ?>>, memref<1x40xi64, strided<[40, 1], offset: ?>>) -> memref<1x40x4096xf32>
    return %0 : memref<1x40x4096xf32>
  }
}

