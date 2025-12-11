module {
  func.func private @subgraph5_decode(memref<4480x1536xf32, strided<[1536, 1], offset: ?>>, memref<1x1x1536xf32, strided<[1536, 1536, 1], offset: ?>>, memref<4480x1536xf32, strided<[1536, 1], offset: ?>>, memref<1536x4480xf32, strided<[4480, 1], offset: ?>>) -> memref<1x1536xf32>
  func.func @forward5_decode(%arg0: memref<20643840xf32>, %arg1: memref<1x1x1536xf32>) -> memref<1x1536xf32> {
    %subview = memref.subview %arg0[0] [6881280] [1] : memref<20643840xf32> to memref<6881280xf32>
    %expand_shape = memref.expand_shape %subview [[0, 1]] output_shape [4480, 1536] : memref<6881280xf32> into memref<4480x1536xf32>
    %subview_0 = memref.subview %arg0[6881280] [6881280] [1] : memref<20643840xf32> to memref<6881280xf32, strided<[1], offset: 6881280>>
    %expand_shape_1 = memref.expand_shape %subview_0 [[0, 1]] output_shape [4480, 1536] : memref<6881280xf32, strided<[1], offset: 6881280>> into memref<4480x1536xf32, strided<[1536, 1], offset: 6881280>>
    %subview_2 = memref.subview %arg0[13762560] [6881280] [1] : memref<20643840xf32> to memref<6881280xf32, strided<[1], offset: 13762560>>
    %expand_shape_3 = memref.expand_shape %subview_2 [[0, 1]] output_shape [1536, 4480] : memref<6881280xf32, strided<[1], offset: 13762560>> into memref<1536x4480xf32, strided<[4480, 1], offset: 13762560>>
    %cast = memref.cast %expand_shape : memref<4480x1536xf32> to memref<4480x1536xf32, strided<[1536, 1], offset: ?>>
    %cast_4 = memref.cast %arg1 : memref<1x1x1536xf32> to memref<1x1x1536xf32, strided<[1536, 1536, 1], offset: ?>>
    %cast_5 = memref.cast %expand_shape_1 : memref<4480x1536xf32, strided<[1536, 1], offset: 6881280>> to memref<4480x1536xf32, strided<[1536, 1], offset: ?>>
    %cast_6 = memref.cast %expand_shape_3 : memref<1536x4480xf32, strided<[4480, 1], offset: 13762560>> to memref<1536x4480xf32, strided<[4480, 1], offset: ?>>
    %0 = call @subgraph5_decode(%cast, %cast_4, %cast_5, %cast_6) : (memref<4480x1536xf32, strided<[1536, 1], offset: ?>>, memref<1x1x1536xf32, strided<[1536, 1536, 1], offset: ?>>, memref<4480x1536xf32, strided<[1536, 1], offset: ?>>, memref<1536x4480xf32, strided<[4480, 1], offset: ?>>) -> memref<1x1536xf32>
    return %0 : memref<1x1536xf32>
  }
}

