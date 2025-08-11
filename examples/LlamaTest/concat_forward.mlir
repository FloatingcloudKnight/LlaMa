module {
  func.func private @subgraph194(memref<1x5x4096xf32, strided<[20480, 4096, 1], offset: ?>>, memref<1x5x4096xf32, strided<[20480, 4096, 1], offset: ?>>, memref<1x5x4096xf32, strided<[20480, 4096, 1], offset: ?>>, memref<1x5x4096xf32, strided<[20480, 4096, 1], offset: ?>>, memref<1x5x4096xf32, strided<[20480, 4096, 1], offset: ?>>, memref<1x5x4096xf32, strided<[20480, 4096, 1], offset: ?>>, memref<1x5x4096xf32, strided<[20480, 4096, 1], offset: ?>>, memref<1x5x4096xf32, strided<[20480, 4096, 1], offset: ?>>) -> memref<1x40x4096xf32>
  func.func @forward194(%arg0: memref<1x5x4096xf32>, %arg1: memref<1x5x4096xf32>, %arg2: memref<1x5x4096xf32>, %arg3: memref<1x5x4096xf32>, %arg4: memref<1x5x4096xf32>, %arg5: memref<1x5x4096xf32>, %arg6: memref<1x5x4096xf32>, %arg7: memref<1x5x4096xf32>) -> memref<1x40x4096xf32> {
    %cast = memref.cast %arg0 : memref<1x5x4096xf32> to memref<1x5x4096xf32, strided<[20480, 4096, 1], offset: ?>>
    %cast_0 = memref.cast %arg1 : memref<1x5x4096xf32> to memref<1x5x4096xf32, strided<[20480, 4096, 1], offset: ?>>
    %cast_1 = memref.cast %arg2 : memref<1x5x4096xf32> to memref<1x5x4096xf32, strided<[20480, 4096, 1], offset: ?>>
    %cast_2 = memref.cast %arg3 : memref<1x5x4096xf32> to memref<1x5x4096xf32, strided<[20480, 4096, 1], offset: ?>>
    %cast_3 = memref.cast %arg4 : memref<1x5x4096xf32> to memref<1x5x4096xf32, strided<[20480, 4096, 1], offset: ?>>
    %cast_4 = memref.cast %arg5 : memref<1x5x4096xf32> to memref<1x5x4096xf32, strided<[20480, 4096, 1], offset: ?>>
    %cast_5 = memref.cast %arg6 : memref<1x5x4096xf32> to memref<1x5x4096xf32, strided<[20480, 4096, 1], offset: ?>>
    %cast_6 = memref.cast %arg7 : memref<1x5x4096xf32> to memref<1x5x4096xf32, strided<[20480, 4096, 1], offset: ?>>
    %0 = call @subgraph194(%cast, %cast_0, %cast_1, %cast_2, %cast_3, %cast_4, %cast_5, %cast_6) : (memref<1x5x4096xf32, strided<[20480, 4096, 1], offset: ?>>, memref<1x5x4096xf32, strided<[20480, 4096, 1], offset: ?>>, memref<1x5x4096xf32, strided<[20480, 4096, 1], offset: ?>>, memref<1x5x4096xf32, strided<[20480, 4096, 1], offset: ?>>, memref<1x5x4096xf32, strided<[20480, 4096, 1], offset: ?>>, memref<1x5x4096xf32, strided<[20480, 4096, 1], offset: ?>>, memref<1x5x4096xf32, strided<[20480, 4096, 1], offset: ?>>, memref<1x5x4096xf32, strided<[20480, 4096, 1], offset: ?>>) -> memref<1x40x4096xf32>
    return %0 : memref<1x40x4096xf32>
  }
}