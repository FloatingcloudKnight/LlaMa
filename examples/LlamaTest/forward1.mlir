module {
  func.func private @subgraph1(memref<1x40x4096xf32, strided<[163840, 4096, 1], offset: ?>>, memref<4096xf32, strided<[1], offset: ?>>, memref<4096x4096xf32, strided<[4096, 1], offset: ?>>, memref<4096x4096xf32, strided<[4096, 1], offset: ?>>, memref<4096x4096xf32, strided<[4096, 1], offset: ?>>, memref<1x40x128xf32, strided<[5120, 128, 1], offset: ?>>, memref<1x40x128xf32, strided<[5120, 128, 1], offset: ?>>, memref<40x41xf32, strided<[41, 1], offset: ?>>, memref<4096x4096xf32, strided<[4096, 1], offset: ?>>, memref<4096xf32, strided<[1], offset: ?>>, memref<11008x4096xf32, strided<[4096, 1], offset: ?>>, memref<11008x4096xf32, strided<[4096, 1], offset: ?>>, memref<4096x11008xf32, strided<[11008, 1], offset: ?>>) -> memref<1x40x4096xf32>
  func.func @forward1(%arg0: memref<202383360xf32>, %arg1: memref<1x40x4096xf32>, %arg2: memref<1x40x128xf32>, %arg3: memref<1x40x128xf32>, %arg4: memref<40x41xf32>) -> memref<1x40x4096xf32> {
    %subview = memref.subview %arg0[131072000] [4096] [1] : memref<202383360xf32> to memref<4096xf32, strided<[1], offset: 131072000>>
    %subview_0 = memref.subview %arg0[131076096] [16777216] [1] : memref<202383360xf32> to memref<16777216xf32, strided<[1], offset: 131076096>>
    %expand_shape = memref.expand_shape %subview_0 [[0, 1]] : memref<16777216xf32, strided<[1], offset: 131076096>> into memref<4096x4096xf32, strided<[4096, 1], offset: 131076096>>
    %subview_1 = memref.subview %arg0[147853312] [16777216] [1] : memref<202383360xf32> to memref<16777216xf32, strided<[1], offset: 147853312>>
    %expand_shape_2 = memref.expand_shape %subview_1 [[0, 1]] : memref<16777216xf32, strided<[1], offset: 147853312>> into memref<4096x4096xf32, strided<[4096, 1], offset: 147853312>>
    %subview_3 = memref.subview %arg0[164630528] [16777216] [1] : memref<202383360xf32> to memref<16777216xf32, strided<[1], offset: 164630528>>
    %expand_shape_4 = memref.expand_shape %subview_3 [[0, 1]] : memref<16777216xf32, strided<[1], offset: 164630528>> into memref<4096x4096xf32, strided<[4096, 1], offset: 164630528>>
    %subview_5 = memref.subview %arg0[181407744] [16777216] [1] : memref<202383360xf32> to memref<16777216xf32, strided<[1], offset: 181407744>>
    %expand_shape_6 = memref.expand_shape %subview_5 [[0, 1]] : memref<16777216xf32, strided<[1], offset: 181407744>> into memref<4096x4096xf32, strided<[4096, 1], offset: 181407744>>
    %subview_7 = memref.subview %arg0[198184960] [4096] [1] : memref<202383360xf32> to memref<4096xf32, strided<[1], offset: 198184960>>
    %subview_8 = memref.subview %arg0[198189056] [45088768] [1] : memref<202383360xf32> to memref<45088768xf32, strided<[1], offset: 198189056>>
    %expand_shape_9 = memref.expand_shape %subview_8 [[0, 1]] : memref<45088768xf32, strided<[1], offset: 198189056>> into memref<11008x4096xf32, strided<[4096, 1], offset: 198189056>>
    %subview_10 = memref.subview %arg0[243277824] [45088768] [1] : memref<202383360xf32> to memref<45088768xf32, strided<[1], offset: 243277824>>
    %expand_shape_11 = memref.expand_shape %subview_10 [[0, 1]] : memref<45088768xf32, strided<[1], offset: 243277824>> into memref<11008x4096xf32, strided<[4096, 1], offset: 243277824>>
    %subview_12 = memref.subview %arg0[288366592] [45088768] [1] : memref<202383360xf32> to memref<45088768xf32, strided<[1], offset: 288366592>>
    %expand_shape_13 = memref.expand_shape %subview_12 [[0, 1]] : memref<45088768xf32, strided<[1], offset: 288366592>> into memref<4096x11008xf32, strided<[11008, 1], offset: 288366592>>
    %cast = memref.cast %arg1 : memref<1x40x4096xf32> to memref<1x40x4096xf32, strided<[163840, 4096, 1], offset: ?>>
    %cast_14 = memref.cast %subview : memref<4096xf32, strided<[1], offset: 131072000>> to memref<4096xf32, strided<[1], offset: ?>>
    %cast_15 = memref.cast %expand_shape : memref<4096x4096xf32, strided<[4096, 1], offset: 131076096>> to memref<4096x4096xf32, strided<[4096, 1], offset: ?>>
    %cast_16 = memref.cast %expand_shape_2 : memref<4096x4096xf32, strided<[4096, 1], offset: 147853312>> to memref<4096x4096xf32, strided<[4096, 1], offset: ?>>
    %cast_17 = memref.cast %expand_shape_4 : memref<4096x4096xf32, strided<[4096, 1], offset: 164630528>> to memref<4096x4096xf32, strided<[4096, 1], offset: ?>>
    %cast_18 = memref.cast %arg2 : memref<1x40x128xf32> to memref<1x40x128xf32, strided<[5120, 128, 1], offset: ?>>
    %cast_19 = memref.cast %arg3 : memref<1x40x128xf32> to memref<1x40x128xf32, strided<[5120, 128, 1], offset: ?>>
    %cast_20 = memref.cast %arg4 : memref<40x41xf32> to memref<40x41xf32, strided<[41, 1], offset: ?>>
    %cast_21 = memref.cast %expand_shape_6 : memref<4096x4096xf32, strided<[4096, 1], offset: 181407744>> to memref<4096x4096xf32, strided<[4096, 1], offset: ?>>
    %cast_22 = memref.cast %subview_7 : memref<4096xf32, strided<[1], offset: 198184960>> to memref<4096xf32, strided<[1], offset: ?>>
    %cast_23 = memref.cast %expand_shape_9 : memref<11008x4096xf32, strided<[4096, 1], offset: 198189056>> to memref<11008x4096xf32, strided<[4096, 1], offset: ?>>
    %cast_24 = memref.cast %expand_shape_11 : memref<11008x4096xf32, strided<[4096, 1], offset: 243277824>> to memref<11008x4096xf32, strided<[4096, 1], offset: ?>>
    %cast_25 = memref.cast %expand_shape_13 : memref<4096x11008xf32, strided<[11008, 1], offset: 288366592>> to memref<4096x11008xf32, strided<[11008, 1], offset: ?>>
    %0 = call @subgraph1(%cast, %cast_14, %cast_15, %cast_16, %cast_17, %cast_18, %cast_19, %cast_20, %cast_21, %cast_22, %cast_23, %cast_24, %cast_25) : (memref<1x40x4096xf32, strided<[163840, 4096, 1], offset: ?>>, memref<4096xf32, strided<[1], offset: ?>>, memref<4096x4096xf32, strided<[4096, 1], offset: ?>>, memref<4096x4096xf32, strided<[4096, 1], offset: ?>>, memref<4096x4096xf32, strided<[4096, 1], offset: ?>>, memref<1x40x128xf32, strided<[5120, 128, 1], offset: ?>>, memref<1x40x128xf32, strided<[5120, 128, 1], offset: ?>>, memref<40x41xf32, strided<[41, 1], offset: ?>>, memref<4096x4096xf32, strided<[4096, 1], offset: ?>>, memref<4096xf32, strided<[1], offset: ?>>, memref<11008x4096xf32, strided<[4096, 1], offset: ?>>, memref<11008x4096xf32, strided<[4096, 1], offset: ?>>, memref<4096x11008xf32, strided<[11008, 1], offset: ?>>) -> memref<1x40x4096xf32>
    return %0 : memref<1x40x4096xf32>
  }
}

