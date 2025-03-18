#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module {
  // subgraph2的输入: X1--->%arg0
  // MHA(X1)
  func.func @subgraph2(%arg0: tensor<1x40x4096xf32>, %arg2: tensor<64xf32>, %arg4: tensor<4096x4096xf32>, %arg5: tensor<4096x4096xf32>, %arg6: tensor<4096x4096xf32>, %arg7: tensor<4096x4096xf32>) -> tensor<1x40x4096xf32> {
    %4 = "tosa.const"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]> : tensor<40xi64>}> : () -> tensor<40xi64>
    %5 = tosa.reshape %4 {new_shape = array<i64: 1, 40>} : (tensor<40xi64>) -> tensor<1x40xi64>
    %cst = arith.constant dense<-3.40282347E+38> : tensor<40x41xf32>
    %6 = "tosa.const"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]> : tensor<41xi64>}> : () -> tensor<41xi64>
    %7 = tosa.reshape %6 {new_shape = array<i64: 1, 41>} : (tensor<41xi64>) -> tensor<1x41xi64>
    %8 = "tosa.const"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]> : tensor<40xi64>}> : () -> tensor<40xi64>
    %9 = tosa.reshape %8 {new_shape = array<i64: 40, 1>} : (tensor<40xi64>) -> tensor<40x1xi64>
    %10 = tosa.sub %7, %9 : (tensor<1x41xi64>, tensor<40x1xi64>) -> tensor<40x41xi64>
    %c1_i64 = arith.constant 1 : i64
    %splat = tensor.splat %c1_i64 : tensor<40x41xi64>
    %11 = arith.cmpi sge, %10, %splat : tensor<40x41xi64>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %12 = tensor.empty() : tensor<40x41xf32>
    %splat_1 = tensor.splat %cst_0 : tensor<40x41xf32>
    %13 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%11, %cst, %splat_1 : tensor<40x41xi1>, tensor<40x41xf32>, tensor<40x41xf32>) outs(%12 : tensor<40x41xf32>) {
    ^bb0(%in: i1, %in_873: f32, %in_874: f32, %out: f32):
      %3745 = arith.select %in, %in_873, %in_874 : f32
      linalg.yield %3745 : f32
    } -> tensor<40x41xf32>
    %14 = "tosa.const"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]> : tensor<41xi64>}> : () -> tensor<41xi64>
    %15 = tosa.reshape %4 {new_shape = array<i64: 40, 1>} : (tensor<40xi64>) -> tensor<40x1xi64>
    %16 = tensor.empty() : tensor<40x41xi1>
    %17 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%14, %15 : tensor<41xi64>, tensor<40x1xi64>) outs(%16 : tensor<40x41xi1>) {
    ^bb0(%in: i64, %in_873: i64, %out: i1):
      %3745 = arith.cmpi sgt, %in, %in_873 : i64
      linalg.yield %3745 : i1
    } -> tensor<40x41xi1>
    %18 = tosa.cast %17 : (tensor<40x41xi1>) -> tensor<40x41xf32>
    %19 = tosa.mul %13, %18 {shift = 0 : i8} : (tensor<40x41xf32>, tensor<40x41xf32>) -> tensor<40x41xf32>
    %20 = tosa.reshape %arg2 {new_shape = array<i64: 1, 64>} : (tensor<64xf32>) -> tensor<1x64xf32>
    %extracted_slice = tensor.extract_slice %20[0, 0] [1, 64] [1, 1] : tensor<1x64xf32> to tensor<1x64xf32>
    %21 = tosa.reshape %extracted_slice {new_shape = array<i64: 1, 64, 1>} : (tensor<1x64xf32>) -> tensor<1x64x1xf32>
    %22 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x64x1xf32>}> : () -> tensor<1x64x1xf32>
    %23 = tosa.add %21, %22 : (tensor<1x64x1xf32>, tensor<1x64x1xf32>) -> tensor<1x64x1xf32>
    %extracted_slice_2 = tensor.extract_slice %5[0, 0] [1, 40] [1, 1] : tensor<1x40xi64> to tensor<1x40xi64>
    %24 = tosa.reshape %extracted_slice_2 {new_shape = array<i64: 1, 1, 40>} : (tensor<1x40xi64>) -> tensor<1x1x40xi64>
    %extracted_slice_3 = tensor.extract_slice %24[0, 0, 0] [1, 1, 40] [1, 1, 1] : tensor<1x1x40xi64> to tensor<1x1x40xi64>
    %25 = tosa.cast %extracted_slice_3 : (tensor<1x1x40xi64>) -> tensor<1x1x40xf32>
    %26 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x64x1xf32>}> : () -> tensor<1x64x1xf32>
    %27 = tosa.add %23, %26 : (tensor<1x64x1xf32>, tensor<1x64x1xf32>) -> tensor<1x64x1xf32>
    %28 = tosa.reshape %27 {new_shape = array<i64: 1, 64, 1>} : (tensor<1x64x1xf32>) -> tensor<1x64x1xf32>
    %29 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x1x40xf32>}> : () -> tensor<1x1x40xf32>
    %30 = tosa.add %25, %29 : (tensor<1x1x40xf32>, tensor<1x1x40xf32>) -> tensor<1x1x40xf32>
    %31 = tosa.reshape %30 {new_shape = array<i64: 1, 1, 40>} : (tensor<1x1x40xf32>) -> tensor<1x1x40xf32>
    %32 = tosa.matmul %28, %31 : (tensor<1x64x1xf32>, tensor<1x1x40xf32>) -> tensor<1x64x40xf32>
    %33 = tosa.reshape %32 {new_shape = array<i64: 1, 64, 40>} : (tensor<1x64x40xf32>) -> tensor<1x64x40xf32>
    %34 = "tosa.const"() <{value = dense<[0, 2, 1]> : tensor<3xi32>}> : () -> tensor<3xi32>
    %35 = tosa.transpose %33, %34 : (tensor<1x64x40xf32>, tensor<3xi32>) -> tensor<1x40x64xf32>
    %36 = tosa.reshape %35 {new_shape = array<i64: 1, 40, 1, 64>} : (tensor<1x40x64xf32>) -> tensor<1x40x1x64xf32>
    %37 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x40x2x64xf32>}> : () -> tensor<1x40x2x64xf32>
    %38 = tosa.add %36, %37 : (tensor<1x40x1x64xf32>, tensor<1x40x2x64xf32>) -> tensor<1x40x2x64xf32>
    %39 = tosa.identity %38 : (tensor<1x40x2x64xf32>) -> tensor<1x40x2x64xf32>
    %40 = tosa.reshape %39 {new_shape = array<i64: 1, 40, 128>} : (tensor<1x40x2x64xf32>) -> tensor<1x40x128xf32>
    %41 = tosa.identity %40 : (tensor<1x40x128xf32>) -> tensor<1x40x128xf32>
    %42 = math.cos %41 : tensor<1x40x128xf32>
    %43 = math.sin %41 : tensor<1x40x128xf32>
    %cst_4 = arith.constant dense<1.000000e+00> : tensor<1xf32>
    %44 = tosa.reshape %cst_4 {new_shape = array<i64: 1, 1, 1>} : (tensor<1xf32>) -> tensor<1x1x1xf32>
    %45 = tosa.mul %42, %44 {shift = 0 : i8} : (tensor<1x40x128xf32>, tensor<1x1x1xf32>) -> tensor<1x40x128xf32>
    %cst_5 = arith.constant dense<1.000000e+00> : tensor<1xf32>
    %46 = tosa.reshape %cst_5 {new_shape = array<i64: 1, 1, 1>} : (tensor<1xf32>) -> tensor<1x1x1xf32>
    %47 = tosa.mul %43, %46 {shift = 0 : i8} : (tensor<1x40x128xf32>, tensor<1x1x1xf32>) -> tensor<1x40x128xf32>
    %60 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %61 = tosa.transpose %arg4, %60 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %62 = tosa.reshape %arg0 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_6 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %63 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%62, %61 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_6 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %64 = tosa.reshape %63 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %65 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %66 = tosa.transpose %arg5, %65 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %67 = tosa.reshape %arg0 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_7 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %68 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%67, %66 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_7 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %69 = tosa.reshape %68 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %70 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %71 = tosa.transpose %arg6, %70 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %72 = tosa.reshape %arg0 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_8 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %73 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%72, %71 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_8 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %74 = tosa.reshape %73 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %75 = tosa.reshape %64 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %76 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %77 = tosa.transpose %75, %76 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %78 = tosa.reshape %69 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %79 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %80 = tosa.transpose %78, %79 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %81 = tosa.reshape %74 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %82 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %83 = tosa.transpose %81, %82 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %84 = tosa.reshape %45 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %85 = tosa.reshape %47 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %86 = tosa.mul %77, %84 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_9 = tensor.extract_slice %77[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_10 = tensor.extract_slice %77[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %87 = tensor.empty() : tensor<1x32x40x64xf32>
    %88 = linalg.negf ins(%extracted_slice_10 : tensor<1x32x40x64xf32>) outs(%87 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %89 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice = tensor.insert_slice %88 into %89[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_11 = tensor.insert_slice %extracted_slice_9 into %inserted_slice[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %90 = tosa.mul %inserted_slice_11, %85 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %91 = tosa.add %86, %90 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %92 = tosa.mul %80, %84 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_12 = tensor.extract_slice %80[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_13 = tensor.extract_slice %80[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %93 = tensor.empty() : tensor<1x32x40x64xf32>
    %94 = linalg.negf ins(%extracted_slice_13 : tensor<1x32x40x64xf32>) outs(%93 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %95 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_14 = tensor.insert_slice %94 into %95[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_15 = tensor.insert_slice %extracted_slice_12 into %inserted_slice_14[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %96 = tosa.mul %inserted_slice_15, %85 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %97 = tosa.add %92, %96 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %98 = tosa.reshape %19 {new_shape = array<i64: 1, 40, 41>} : (tensor<40x41xf32>) -> tensor<1x40x41xf32>
    %99 = tosa.reshape %98 {new_shape = array<i64: 1, 1, 40, 41>} : (tensor<1x40x41xf32>) -> tensor<1x1x40x41xf32>
    %extracted_slice_16 = tensor.extract_slice %99[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_17 = tensor.extract_slice %extracted_slice_16[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %100 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x1x40x41xf32>}> : () -> tensor<1x1x40x41xf32>
    %101 = tosa.add %extracted_slice_17, %100 : (tensor<1x1x40x41xf32>, tensor<1x1x40x41xf32>) -> tensor<1x1x40x41xf32>
    %extracted_slice_18 = tensor.extract_slice %101[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_19 = tensor.extract_slice %extracted_slice_18[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_20 = tensor.extract_slice %extracted_slice_19[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_21 = tensor.extract_slice %extracted_slice_20[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x40xf32>
    %cst_22 = arith.constant 0.000000e+00 : f32
    %splat_23 = tensor.splat %cst_22 : tensor<40x40xf32>
    %102 = tosa.reshape %extracted_slice_21 {new_shape = array<i64: 40, 40>} : (tensor<1x1x40x40xf32>) -> tensor<40x40xf32>
    %103 = tosa.add %splat_23, %102 : (tensor<40x40xf32>, tensor<40x40xf32>) -> tensor<40x40xf32>
    %104 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %105 = tosa.transpose %97, %104 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %106 = tosa.reshape %91 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %107 = tosa.reshape %105 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %108 = tosa.matmul %106, %107 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %cst_24 = arith.constant 0.0883883461 : f32
    %splat_25 = tensor.splat %cst_24 : tensor<32x40x40xf32>
    %109 = tosa.mul %108, %splat_25 {shift = 0 : i8} : (tensor<32x40x40xf32>, tensor<32x40x40xf32>) -> tensor<32x40x40xf32>
    %110 = tosa.add %109, %103 : (tensor<32x40x40xf32>, tensor<40x40xf32>) -> tensor<32x40x40xf32>
    %111 = tosa.reduce_max %110 {axis = 2 : i32} : (tensor<32x40x40xf32>) -> tensor<32x40x1xf32>
    %112 = tosa.sub %110, %111 : (tensor<32x40x40xf32>, tensor<32x40x1xf32>) -> tensor<32x40x40xf32>
    %113 = math.exp %112 : tensor<32x40x40xf32>
    %114 = tosa.reduce_sum %113 {axis = 2 : i32} : (tensor<32x40x40xf32>) -> tensor<32x40x1xf32>
    %115 = tosa.log %114 : (tensor<32x40x1xf32>) -> tensor<32x40x1xf32>
    %116 = tosa.add %111, %115 : (tensor<32x40x1xf32>, tensor<32x40x1xf32>) -> tensor<32x40x1xf32>
    %117 = tosa.sub %110, %116 : (tensor<32x40x40xf32>, tensor<32x40x1xf32>) -> tensor<32x40x40xf32>
    %118 = math.exp %117 : tensor<32x40x40xf32>
    %119 = tosa.reshape %116 {new_shape = array<i64: 1, 32, 40>} : (tensor<32x40x1xf32>) -> tensor<1x32x40xf32>
    %120 = tosa.reshape %83 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %121 = tosa.matmul %118, %120 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %122 = tosa.reshape %121 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %123 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %124 = tosa.transpose %122, %123 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %125 = tosa.reshape %124 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %126 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %127 = tosa.transpose %arg7, %126 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %128 = tosa.reshape %125 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_26 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %129 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%128, %127 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_26 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %130 = tosa.reshape %129 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    return %130 : tensor<1x40x4096xf32>
  }
}