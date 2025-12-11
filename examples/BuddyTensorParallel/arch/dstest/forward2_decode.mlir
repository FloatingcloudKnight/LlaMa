module {
  func.func private @subgraph2_decode(memref<1x1x1536xf32, strided<[1536, 1536, 1], offset: ?>>, memref<768x1536xf32, strided<[1536, 1], offset: ?>>, memref<768xf32, strided<[1], offset: ?>>, memref<128x1536xf32, strided<[1536, 1], offset: ?>>, memref<128xf32, strided<[1], offset: ?>>, memref<128x1536xf32, strided<[1536, 1], offset: ?>>, memref<128xf32, strided<[1], offset: ?>>, memref<1x1x128xf32, strided<[128, 128, 1], offset: ?>>, memref<1x1x128xf32, strided<[128, 128, 1], offset: ?>>, memref<1x1x1024x128xf32, strided<[131072, 131072, 128, 1], offset: ?>>, memref<1xi64, strided<[1], offset: ?>>, memref<1x1x1024x128xf32, strided<[131072, 131072, 128, 1], offset: ?>>, memref<1x1x1x1024xi1, strided<[1024, 1024, 1024, 1], offset: ?>>, memref<1536x768xf32, strided<[768, 1], offset: ?>>) -> (memref<1x1x1024x128xf32>, memref<1x1x1024x128xf32>, memref<1x1536xf32>)
  func.func @forward2_decode(%arg0: memref<2753536xf32>, %arg1: memref<1xi64>, %arg2: memref<1x1x1024x128xf32>, %arg3: memref<1x1x1024x128xf32>, %arg4: memref<1x1x1536xf32>, %arg5: memref<1x1x128xf32>, %arg6: memref<1x1x128xf32>, %arg7: memref<1x1x1x1024xi1>) -> (memref<1x1x1024x128xf32>, memref<1x1x1024x128xf32>, memref<1x1536xf32>) {
    %subview = memref.subview %arg0[0] [1179648] [1] : memref<2753536xf32> to memref<1179648xf32>
    %expand_shape = memref.expand_shape %subview [[0, 1]] output_shape [768, 1536] : memref<1179648xf32> into memref<768x1536xf32>
    %subview_0 = memref.subview %arg0[1179648] [768] [1] : memref<2753536xf32> to memref<768xf32, strided<[1], offset: 1179648>>
    %subview_1 = memref.subview %arg0[1180416] [196608] [1] : memref<2753536xf32> to memref<196608xf32, strided<[1], offset: 1180416>>
    %expand_shape_2 = memref.expand_shape %subview_1 [[0, 1]] output_shape [128, 1536] : memref<196608xf32, strided<[1], offset: 1180416>> into memref<128x1536xf32, strided<[1536, 1], offset: 1180416>>
    %subview_3 = memref.subview %arg0[1377024] [128] [1] : memref<2753536xf32> to memref<128xf32, strided<[1], offset: 1377024>>
    %subview_4 = memref.subview %arg0[1377152] [196608] [1] : memref<2753536xf32> to memref<196608xf32, strided<[1], offset: 1377152>>
    %expand_shape_5 = memref.expand_shape %subview_4 [[0, 1]] output_shape [128, 1536] : memref<196608xf32, strided<[1], offset: 1377152>> into memref<128x1536xf32, strided<[1536, 1], offset: 1377152>>
    %subview_6 = memref.subview %arg0[1573760] [128] [1] : memref<2753536xf32> to memref<128xf32, strided<[1], offset: 1573760>>
    %subview_7 = memref.subview %arg0[1573888] [1179648] [1] : memref<2753536xf32> to memref<1179648xf32, strided<[1], offset: 1573888>>
    %expand_shape_8 = memref.expand_shape %subview_7 [[0, 1]] output_shape [1536, 768] : memref<1179648xf32, strided<[1], offset: 1573888>> into memref<1536x768xf32, strided<[768, 1], offset: 1573888>>
    %cast = memref.cast %arg4 : memref<1x1x1536xf32> to memref<1x1x1536xf32, strided<[1536, 1536, 1], offset: ?>>
    %cast_9 = memref.cast %expand_shape : memref<768x1536xf32> to memref<768x1536xf32, strided<[1536, 1], offset: ?>>
    %cast_10 = memref.cast %subview_0 : memref<768xf32, strided<[1], offset: 1179648>> to memref<768xf32, strided<[1], offset: ?>>
    %cast_11 = memref.cast %expand_shape_2 : memref<128x1536xf32, strided<[1536, 1], offset: 1180416>> to memref<128x1536xf32, strided<[1536, 1], offset: ?>>
    %cast_12 = memref.cast %subview_3 : memref<128xf32, strided<[1], offset: 1377024>> to memref<128xf32, strided<[1], offset: ?>>
    %cast_13 = memref.cast %expand_shape_5 : memref<128x1536xf32, strided<[1536, 1], offset: 1377152>> to memref<128x1536xf32, strided<[1536, 1], offset: ?>>
    %cast_14 = memref.cast %subview_6 : memref<128xf32, strided<[1], offset: 1573760>> to memref<128xf32, strided<[1], offset: ?>>
    %cast_15 = memref.cast %arg5 : memref<1x1x128xf32> to memref<1x1x128xf32, strided<[128, 128, 1], offset: ?>>
    %cast_16 = memref.cast %arg6 : memref<1x1x128xf32> to memref<1x1x128xf32, strided<[128, 128, 1], offset: ?>>
    %cast_17 = memref.cast %arg2 : memref<1x1x1024x128xf32> to memref<1x1x1024x128xf32, strided<[131072, 131072, 128, 1], offset: ?>>
    %cast_18 = memref.cast %arg1 : memref<1xi64> to memref<1xi64, strided<[1], offset: ?>>
    %cast_19 = memref.cast %arg3 : memref<1x1x1024x128xf32> to memref<1x1x1024x128xf32, strided<[131072, 131072, 128, 1], offset: ?>>
    %cast_20 = memref.cast %arg7 : memref<1x1x1x1024xi1> to memref<1x1x1x1024xi1, strided<[1024, 1024, 1024, 1], offset: ?>>
    %cast_21 = memref.cast %expand_shape_8 : memref<1536x768xf32, strided<[768, 1], offset: 1573888>> to memref<1536x768xf32, strided<[768, 1], offset: ?>>
    %0:3 = call @subgraph2_decode(%cast, %cast_9, %cast_10, %cast_11, %cast_12, %cast_13, %cast_14, %cast_15, %cast_16, %cast_17, %cast_18, %cast_19, %cast_20, %cast_21) : (memref<1x1x1536xf32, strided<[1536, 1536, 1], offset: ?>>, memref<768x1536xf32, strided<[1536, 1], offset: ?>>, memref<768xf32, strided<[1], offset: ?>>, memref<128x1536xf32, strided<[1536, 1], offset: ?>>, memref<128xf32, strided<[1], offset: ?>>, memref<128x1536xf32, strided<[1536, 1], offset: ?>>, memref<128xf32, strided<[1], offset: ?>>, memref<1x1x128xf32, strided<[128, 128, 1], offset: ?>>, memref<1x1x128xf32, strided<[128, 128, 1], offset: ?>>, memref<1x1x1024x128xf32, strided<[131072, 131072, 128, 1], offset: ?>>, memref<1xi64, strided<[1], offset: ?>>, memref<1x1x1024x128xf32, strided<[131072, 131072, 128, 1], offset: ?>>, memref<1x1x1x1024xi1, strided<[1024, 1024, 1024, 1], offset: ?>>, memref<1536x768xf32, strided<[768, 1], offset: ?>>) -> (memref<1x1x1024x128xf32>, memref<1x1x1024x128xf32>, memref<1x1536xf32>)
    return %0#0, %0#1, %0#2 : memref<1x1x1024x128xf32>, memref<1x1x1024x128xf32>, memref<1x1536xf32>
  }
}

