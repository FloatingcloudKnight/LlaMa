module {
  func.func private @subgraph4(memref<1x40x4096xf32, strided<[163840, 4096, 1], offset: ?>>, memref<1x40x4096xf32, strided<[163840, 4096, 1], offset: ?>>, memref<4096xf32, strided<[1], offset: ?>>) -> memref<1x40x4096xf32>
  func.func @forward4(%arg0: memref<6738415680xf32>, %arg1: memref<1x40x4096xf32>, %arg2: memref<1x40x4096xf32>) -> memref<1x40x4096xf32> {
    %subview_9 = memref.subview %arg0[198184960] [4096] [1] : memref<6738415680xf32> to memref<4096xf32, strided<[1], offset: 198184960>>
    %cast = memref.cast %arg1 : memref<1x40x4096xf32> to memref<1x40x4096xf32, strided<[163840, 4096, 1], offset: ?>>
    %cast0 = memref.cast %arg2 : memref<1x40x4096xf32> to memref<1x40x4096xf32, strided<[163840, 4096, 1], offset: ?>>
    %cast_523 = memref.cast %subview_9 : memref<4096xf32, strided<[1], offset: 198184960>> to memref<4096xf32, strided<[1], offset: ?>>
    %0 = call @subgraph4(%cast, %cast0, %cast_523) : (memref<1x40x4096xf32, strided<[163840, 4096, 1], offset: ?>>, memref<1x40x4096xf32, strided<[163840, 4096, 1], offset: ?>>, memref<4096xf32, strided<[1], offset: ?>>) -> memref<1x40x4096xf32>
    return %0 : memref<1x40x4096xf32>
  }
}

