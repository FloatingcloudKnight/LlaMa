module {
  func.func private @subgraph5(memref<1x40x4096xf32, strided<[163840, 4096, 1], offset: ?>>, memref<1x40x4096xf32, strided<[163840, 4096, 1], offset: ?>>, memref<11008x4096xf32, strided<[4096, 1], offset: ?>>, memref<11008x4096xf32, strided<[4096, 1], offset: ?>>, memref<4096x11008xf32, strided<[11008, 1], offset: ?>>) -> memref<1x40x4096xf32>
  func.func @forward5(%arg0: memref<6738415680xf32>, %arg1: memref<1x40x4096xf32>, %arg2: memref<1x40x4096xf32>) -> memref<1x40x4096xf32> {
    %subview_10 = memref.subview %arg0[198189056] [45088768] [1] : memref<6738415680xf32> to memref<45088768xf32, strided<[1], offset: 198189056>>
    %expand_shape_11 = memref.expand_shape %subview_10 [[0, 1]] : memref<45088768xf32, strided<[1], offset: 198189056>> into memref<11008x4096xf32, strided<[4096, 1], offset: 198189056>>
    %subview_12 = memref.subview %arg0[243277824] [45088768] [1] : memref<6738415680xf32> to memref<45088768xf32, strided<[1], offset: 243277824>>
    %expand_shape_13 = memref.expand_shape %subview_12 [[0, 1]] : memref<45088768xf32, strided<[1], offset: 243277824>> into memref<11008x4096xf32, strided<[4096, 1], offset: 243277824>>
    %subview_14 = memref.subview %arg0[288366592] [45088768] [1] : memref<6738415680xf32> to memref<45088768xf32, strided<[1], offset: 288366592>>
    %expand_shape_15 = memref.expand_shape %subview_14 [[0, 1]] : memref<45088768xf32, strided<[1], offset: 288366592>> into memref<4096x11008xf32, strided<[11008, 1], offset: 288366592>>
    %cast = memref.cast %arg1 : memref<1x40x4096xf32> to memref<1x40x4096xf32, strided<[163840, 4096, 1], offset: ?>>
    %cast0 = memref.cast %arg2 : memref<1x40x4096xf32> to memref<1x40x4096xf32, strided<[163840, 4096, 1], offset: ?>>
    %cast_524 = memref.cast %expand_shape_11 : memref<11008x4096xf32, strided<[4096, 1], offset: 198189056>> to memref<11008x4096xf32, strided<[4096, 1], offset: ?>>
    %cast_525 = memref.cast %expand_shape_13 : memref<11008x4096xf32, strided<[4096, 1], offset: 243277824>> to memref<11008x4096xf32, strided<[4096, 1], offset: ?>>
    %cast_526 = memref.cast %expand_shape_15 : memref<4096x11008xf32, strided<[11008, 1], offset: 288366592>> to memref<4096x11008xf32, strided<[11008, 1], offset: ?>>
    %0 = call @subgraph5(%cast, %cast0, %cast_524, %cast_525, %cast_526) : (memref<1x40x4096xf32, strided<[163840, 4096, 1], offset: ?>>, memref<1x40x4096xf32, strided<[163840, 4096, 1], offset: ?>>, memref<11008x4096xf32, strided<[4096, 1], offset: ?>>, memref<11008x4096xf32, strided<[4096, 1], offset: ?>>, memref<4096x11008xf32, strided<[11008, 1], offset: ?>>) -> memref<1x40x4096xf32>
    return %0 : memref<1x40x4096xf32>
  }
}

