module {
  func.func private @subgraph1(memref<1x40x4096xf32, strided<[163840, 4096, 1], offset: ?>>, memref<4096xf32, strided<[1], offset: ?>>) -> memref<1x40x4096xf32>
  func.func @forward1(%arg0: memref<6738415680xf32>, %arg1: memref<1x40x4096xf32>) -> memref<1x40x4096xf32> {
    %cast = memref.cast %arg1 : memref<1x40x4096xf32> to memref<1x40x4096xf32, strided<[163840, 4096, 1], offset: ?>>
    %subview_0 = memref.subview %arg0[131072000] [4096] [1] : memref<6738415680xf32> to memref<4096xf32, strided<[1], offset: 131072000>>
    %cast_518 = memref.cast %subview_0 : memref<4096xf32, strided<[1], offset: 131072000>> to memref<4096xf32, strided<[1], offset: ?>>
    %0 = call @subgraph1(%cast, %cast_518) : (memref<1x40x4096xf32, strided<[163840, 4096, 1], offset: ?>>, memref<4096xf32, strided<[1], offset: ?>>) -> memref<1x40x4096xf32>
    return %0 : memref<1x40x4096xf32>
  }
}

