module {
  func.func private @subgraph195(memref<40x4096xf32, strided<[4096, 1], offset: ?>>, memref<40x4096xf32, strided<[4096, 1], offset: ?>>, memref<40x4096xf32, strided<[4096, 1], offset: ?>>, memref<40x4096xf32, strided<[4096, 1], offset: ?>>) -> (memref<10x4096xf32>, memref<10x4096xf32>, memref<10x4096xf32>, memref<10x4096xf32>)
  func.func @forward195(%arg0: memref<40x4096xf32>, %arg1: memref<40x4096xf32>, %arg2: memref<40x4096xf32>, %arg3: memref<40x4096xf32>) -> (memref<10x4096xf32>, memref<10x4096xf32>, memref<10x4096xf32>, memref<10x4096xf32>) {
    %cast = memref.cast %arg0 : memref<40x4096xf32> to memref<40x4096xf32, strided<[4096, 1], offset: ?>>
    %cast_0 = memref.cast %arg1 : memref<40x4096xf32> to memref<40x4096xf32, strided<[4096, 1], offset: ?>>
    %cast_1 = memref.cast %arg2 : memref<40x4096xf32> to memref<40x4096xf32, strided<[4096, 1], offset: ?>>
    %cast_2 = memref.cast %arg3 : memref<40x4096xf32> to memref<40x4096xf32, strided<[4096, 1], offset: ?>>
    %0:4 = call @subgraph195(%cast, %cast_0, %cast_1, %cast_2) : (memref<40x4096xf32, strided<[4096, 1], offset: ?>>, memref<40x4096xf32, strided<[4096, 1], offset: ?>>, memref<40x4096xf32, strided<[4096, 1], offset: ?>>, memref<40x4096xf32, strided<[4096, 1], offset: ?>>) -> (memref<10x4096xf32>, memref<10x4096xf32>, memref<10x4096xf32>, memref<10x4096xf32>)
    return %0#0, %0#1, %0#0, %0#1 : memref<10x4096xf32>, memref<10x4096xf32>, memref<10x4096xf32>, memref<10x4096xf32>
  }
}