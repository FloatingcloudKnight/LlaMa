module {
  func.func private @subgraph169_decode(memref<1x1x1536xf32, strided<[1536, 1536, 1], offset: ?>>, memref<1536xf32, strided<[1], offset: ?>>, memref<151936x1536xf32, strided<[1536, 1], offset: ?>>) -> memref<1x1x151936xf32>
  func.func @forward169_decode(%arg0: memref<233375232xf32>, %arg1: memref<1x1x1536xf32>) -> memref<1x1x151936xf32> {
    %subview = memref.subview %arg0[0] [1536] [1] : memref<233375232xf32> to memref<1536xf32>
    %subview_0 = memref.subview %arg0[1536] [233373696] [1] : memref<233375232xf32> to memref<233373696xf32, strided<[1], offset: 1536>>
    %expand_shape = memref.expand_shape %subview_0 [[0, 1]] output_shape [151936, 1536] : memref<233373696xf32, strided<[1], offset: 1536>> into memref<151936x1536xf32, strided<[1536, 1], offset: 1536>>
    %cast = memref.cast %arg1 : memref<1x1x1536xf32> to memref<1x1x1536xf32, strided<[1536, 1536, 1], offset: ?>>
    %cast_1 = memref.cast %subview : memref<1536xf32> to memref<1536xf32, strided<[1], offset: ?>>
    %cast_2 = memref.cast %expand_shape : memref<151936x1536xf32, strided<[1536, 1], offset: 1536>> to memref<151936x1536xf32, strided<[1536, 1], offset: ?>>
    %0 = call @subgraph169_decode(%cast, %cast_1, %cast_2) : (memref<1x1x1536xf32, strided<[1536, 1536, 1], offset: ?>>, memref<1536xf32, strided<[1], offset: ?>>, memref<151936x1536xf32, strided<[1536, 1], offset: ?>>) -> memref<1x1x151936xf32>
    return %0 : memref<1x1x151936xf32>
  }
}

