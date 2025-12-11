module {
  func.func private @subgraph0_prefill(memref<151936x1536xf32, strided<[1536, 1], offset: ?>>, memref<1x1024xi64, strided<[1024, 1], offset: ?>>, memref<64xf32, strided<[1], offset: ?>>) -> (memref<1x1024x1536xf32>, memref<1024x1024xi1>, memref<1x1024x128xf32>, memref<1x1024x128xf32>)
  func.func @forward0_prefill(%arg0: memref<233373760xf32>, %arg1: memref<1x1024xi64>) -> (memref<1x1024x1536xf32>, memref<1024x1024xi1>, memref<1x1024x128xf32>, memref<1x1024x128xf32>) {
    %subview = memref.subview %arg0[0] [233373696] [1] : memref<233373760xf32> to memref<233373696xf32>
    %expand_shape = memref.expand_shape %subview [[0, 1]] output_shape [151936, 1536] : memref<233373696xf32> into memref<151936x1536xf32>
    %subview_0 = memref.subview %arg0[233373696] [64] [1] : memref<233373760xf32> to memref<64xf32, strided<[1], offset: 233373696>>
    %cast = memref.cast %expand_shape : memref<151936x1536xf32> to memref<151936x1536xf32, strided<[1536, 1], offset: ?>>
    %cast_1 = memref.cast %arg1 : memref<1x1024xi64> to memref<1x1024xi64, strided<[1024, 1], offset: ?>>
    %cast_2 = memref.cast %subview_0 : memref<64xf32, strided<[1], offset: 233373696>> to memref<64xf32, strided<[1], offset: ?>>
    %0:4 = call @subgraph0_prefill(%cast, %cast_1, %cast_2) : (memref<151936x1536xf32, strided<[1536, 1], offset: ?>>, memref<1x1024xi64, strided<[1024, 1], offset: ?>>, memref<64xf32, strided<[1], offset: ?>>) -> (memref<1x1024x1536xf32>, memref<1024x1024xi1>, memref<1x1024x128xf32>, memref<1x1024x128xf32>)
    return %0#0, %0#1, %0#2, %0#3 : memref<1x1024x1536xf32>, memref<1024x1024xi1>, memref<1x1024x128xf32>, memref<1x1024x128xf32>
  }
}

