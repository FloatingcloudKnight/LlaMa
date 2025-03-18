module {
  func.func private @subgraph2(memref<1x40x4096xf32, strided<[163840, 4096, 1], offset: ?>>, memref<64xf32, strided<[1], offset: ?>>, memref<4096x4096xf32, strided<[4096, 1], offset: ?>>, memref<4096x4096xf32, strided<[4096, 1], offset: ?>>, memref<4096x4096xf32, strided<[4096, 1], offset: ?>>, memref<4096x4096xf32, strided<[4096, 1], offset: ?>>) -> memref<1x40x4096xf32>
  func.func @forward2(%arg0: memref<6738415680xf32>, %arg1: memref<1x40x4096xf32>) -> memref<1x40x4096xf32> {
    %subview_1 = memref.subview %arg0[131076096] [16777216] [1] : memref<6738415680xf32> to memref<16777216xf32, strided<[1], offset: 131076096>>
    %expand_shape_2 = memref.expand_shape %subview_1 [[0, 1]] : memref<16777216xf32, strided<[1], offset: 131076096>> into memref<4096x4096xf32, strided<[4096, 1], offset: 131076096>>
    %subview_3 = memref.subview %arg0[147853312] [16777216] [1] : memref<6738415680xf32> to memref<16777216xf32, strided<[1], offset: 147853312>>
    %expand_shape_4 = memref.expand_shape %subview_3 [[0, 1]] : memref<16777216xf32, strided<[1], offset: 147853312>> into memref<4096x4096xf32, strided<[4096, 1], offset: 147853312>>
    %subview_5 = memref.subview %arg0[164630528] [16777216] [1] : memref<6738415680xf32> to memref<16777216xf32, strided<[1], offset: 164630528>>
    %expand_shape_6 = memref.expand_shape %subview_5 [[0, 1]] : memref<16777216xf32, strided<[1], offset: 164630528>> into memref<4096x4096xf32, strided<[4096, 1], offset: 164630528>>
    %subview_7 = memref.subview %arg0[181407744] [16777216] [1] : memref<6738415680xf32> to memref<16777216xf32, strided<[1], offset: 181407744>>
    %expand_shape_8 = memref.expand_shape %subview_7 [[0, 1]] : memref<16777216xf32, strided<[1], offset: 181407744>> into memref<4096x4096xf32, strided<[4096, 1], offset: 181407744>>
    %subview_515 = memref.subview %arg0[6738415616] [64] [1] : memref<6738415680xf32> to memref<64xf32, strided<[1], offset: 6738415616>>
    %cast = memref.cast %arg1 : memref<1x40x4096xf32> to memref<1x40x4096xf32, strided<[163840, 4096, 1], offset: ?>>
    %cast_517 = memref.cast %subview_515 : memref<64xf32, strided<[1], offset: 6738415616>> to memref<64xf32, strided<[1], offset: ?>>
    %cast_519 = memref.cast %expand_shape_2 : memref<4096x4096xf32, strided<[4096, 1], offset: 131076096>> to memref<4096x4096xf32, strided<[4096, 1], offset: ?>>
    %cast_520 = memref.cast %expand_shape_4 : memref<4096x4096xf32, strided<[4096, 1], offset: 147853312>> to memref<4096x4096xf32, strided<[4096, 1], offset: ?>>
    %cast_521 = memref.cast %expand_shape_6 : memref<4096x4096xf32, strided<[4096, 1], offset: 164630528>> to memref<4096x4096xf32, strided<[4096, 1], offset: ?>>
    %cast_522 = memref.cast %expand_shape_8 : memref<4096x4096xf32, strided<[4096, 1], offset: 181407744>> to memref<4096x4096xf32, strided<[4096, 1], offset: ?>>
    %0 = call @subgraph2(%cast, %cast_517, %cast_519, %cast_520, %cast_521, %cast_522) : (memref<1x40x4096xf32, strided<[163840, 4096, 1], offset: ?>>, memref<64xf32, strided<[1], offset: ?>>, memref<4096x4096xf32, strided<[4096, 1], offset: ?>>, memref<4096x4096xf32, strided<[4096, 1], offset: ?>>, memref<4096x4096xf32, strided<[4096, 1], offset: ?>>, memref<4096x4096xf32, strided<[4096, 1], offset: ?>>) -> memref<1x40x4096xf32>
    return %0 : memref<1x40x4096xf32>
  }
}

