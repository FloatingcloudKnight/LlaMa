#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module {
  func.func @subgraph1(%arg0: tensor<64xf32>, %arg1: tensor<1x40x4096xf32>, %arg2: tensor<1x40x4096xf32>, %arg3: tensor<4096xf32>, %arg4: tensor<4096x4096xf32>, %arg5: tensor<4096x4096xf32>, %arg6: tensor<4096x4096xf32>, %arg7: tensor<4096x4096xf32>, %arg8: tensor<1x40x4096xf32>, %arg9: tensor<4096xf32>, %arg10: tensor<11008x4096xf32>, %arg11: tensor<11008x4096xf32>, %arg12: tensor<4096x11008xf32>) -> (tensor<40x41xf32>, tensor<40x41xf32>, tensor<40x41xf32>, tensor<40x41xf32>, tensor<40x41xf32>, tensor<40x41xf32>, tensor<40x41xf32>, tensor<40x41xf32>, tensor<40x41xf32>, tensor<40x41xf32>, tensor<40x41xf32>, tensor<40x41xf32>, tensor<40x41xf32>, tensor<40x41xf32>, tensor<40x41xf32>, tensor<40x41xf32>, tensor<40x41xf32>, tensor<40x41xf32>, tensor<40x41xf32>, tensor<40x41xf32>, tensor<40x41xf32>, tensor<40x41xf32>, tensor<40x41xf32>, tensor<40x41xf32>, tensor<40x41xf32>, tensor<40x41xf32>, tensor<40x41xf32>, tensor<40x41xf32>, tensor<40x41xf32>, tensor<40x41xf32>, tensor<40x41xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x4096xf32>) {
    %0 = "tosa.const"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]> : tensor<40xi64>}> : () -> tensor<40xi64>
    %1 = tosa.reshape %0 {new_shape = array<i64: 1, 40>} : (tensor<40xi64>) -> tensor<1x40xi64>
    %cst = arith.constant dense<-3.40282347E+38> : tensor<40x41xf32>
    %2 = "tosa.const"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]> : tensor<41xi64>}> : () -> tensor<41xi64>
    %3 = tosa.reshape %2 {new_shape = array<i64: 1, 41>} : (tensor<41xi64>) -> tensor<1x41xi64>
    %4 = "tosa.const"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]> : tensor<40xi64>}> : () -> tensor<40xi64>
    %5 = tosa.reshape %4 {new_shape = array<i64: 40, 1>} : (tensor<40xi64>) -> tensor<40x1xi64>
    %6 = tosa.sub %3, %5 : (tensor<1x41xi64>, tensor<40x1xi64>) -> tensor<40x41xi64>
    %c1_i64 = arith.constant 1 : i64
    %splat = tensor.splat %c1_i64 : tensor<40x41xi64>
    %7 = arith.cmpi sge, %6, %splat : tensor<40x41xi64>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %8 = tensor.empty() : tensor<40x41xf32>
    %splat_1 = tensor.splat %cst_0 : tensor<40x41xf32>
    %9 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%7, %cst, %splat_1 : tensor<40x41xi1>, tensor<40x41xf32>, tensor<40x41xf32>) outs(%8 : tensor<40x41xf32>) {
    ^bb0(%in: i1, %in_31: f32, %in_32: f32, %out: f32):
      %159 = arith.select %in, %in_31, %in_32 : f32
      linalg.yield %159 : f32
    } -> tensor<40x41xf32>
    %10 = "tosa.const"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]> : tensor<41xi64>}> : () -> tensor<41xi64>
    %11 = tosa.reshape %0 {new_shape = array<i64: 40, 1>} : (tensor<40xi64>) -> tensor<40x1xi64>
    %12 = tensor.empty() : tensor<40x41xi1>
    %13 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%10, %11 : tensor<41xi64>, tensor<40x1xi64>) outs(%12 : tensor<40x41xi1>) {
    ^bb0(%in: i64, %in_31: i64, %out: i1):
      %159 = arith.cmpi sgt, %in, %in_31 : i64
      linalg.yield %159 : i1
    } -> tensor<40x41xi1>
    %14 = tosa.cast %13 : (tensor<40x41xi1>) -> tensor<40x41xf32>
    %15 = tosa.mul %9, %14 {shift = 0 : i8} : (tensor<40x41xf32>, tensor<40x41xf32>) -> tensor<40x41xf32>
    %16 = tosa.reshape %arg0 {new_shape = array<i64: 1, 64>} : (tensor<64xf32>) -> tensor<1x64xf32>
    %extracted_slice = tensor.extract_slice %16[0, 0] [1, 64] [1, 1] : tensor<1x64xf32> to tensor<1x64xf32>
    %17 = tosa.reshape %extracted_slice {new_shape = array<i64: 1, 64, 1>} : (tensor<1x64xf32>) -> tensor<1x64x1xf32>
    %18 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x64x1xf32>}> : () -> tensor<1x64x1xf32>
    %19 = tosa.add %17, %18 : (tensor<1x64x1xf32>, tensor<1x64x1xf32>) -> tensor<1x64x1xf32>
    %extracted_slice_2 = tensor.extract_slice %1[0, 0] [1, 40] [1, 1] : tensor<1x40xi64> to tensor<1x40xi64>
    %20 = tosa.reshape %extracted_slice_2 {new_shape = array<i64: 1, 1, 40>} : (tensor<1x40xi64>) -> tensor<1x1x40xi64>
    %extracted_slice_3 = tensor.extract_slice %20[0, 0, 0] [1, 1, 40] [1, 1, 1] : tensor<1x1x40xi64> to tensor<1x1x40xi64>
    %21 = tosa.cast %extracted_slice_3 : (tensor<1x1x40xi64>) -> tensor<1x1x40xf32>
    %22 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x64x1xf32>}> : () -> tensor<1x64x1xf32>
    %23 = tosa.add %19, %22 : (tensor<1x64x1xf32>, tensor<1x64x1xf32>) -> tensor<1x64x1xf32>
    %24 = tosa.reshape %23 {new_shape = array<i64: 1, 64, 1>} : (tensor<1x64x1xf32>) -> tensor<1x64x1xf32>
    %25 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x1x40xf32>}> : () -> tensor<1x1x40xf32>
    %26 = tosa.add %21, %25 : (tensor<1x1x40xf32>, tensor<1x1x40xf32>) -> tensor<1x1x40xf32>
    %27 = tosa.reshape %26 {new_shape = array<i64: 1, 1, 40>} : (tensor<1x1x40xf32>) -> tensor<1x1x40xf32>
    %28 = tosa.matmul %24, %27 : (tensor<1x64x1xf32>, tensor<1x1x40xf32>) -> tensor<1x64x40xf32>
    %29 = tosa.reshape %28 {new_shape = array<i64: 1, 64, 40>} : (tensor<1x64x40xf32>) -> tensor<1x64x40xf32>
    %30 = "tosa.const"() <{value = dense<[0, 2, 1]> : tensor<3xi32>}> : () -> tensor<3xi32>
    %31 = tosa.transpose %29, %30 : (tensor<1x64x40xf32>, tensor<3xi32>) -> tensor<1x40x64xf32>
    %32 = tosa.reshape %31 {new_shape = array<i64: 1, 40, 1, 64>} : (tensor<1x40x64xf32>) -> tensor<1x40x1x64xf32>
    %33 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x40x2x64xf32>}> : () -> tensor<1x40x2x64xf32>
    %34 = tosa.add %32, %33 : (tensor<1x40x1x64xf32>, tensor<1x40x2x64xf32>) -> tensor<1x40x2x64xf32>
    %35 = tosa.identity %34 : (tensor<1x40x2x64xf32>) -> tensor<1x40x2x64xf32>
    %36 = tosa.reshape %35 {new_shape = array<i64: 1, 40, 128>} : (tensor<1x40x2x64xf32>) -> tensor<1x40x128xf32>
    %37 = tosa.identity %36 : (tensor<1x40x128xf32>) -> tensor<1x40x128xf32>
    %38 = math.cos %37 : tensor<1x40x128xf32>
    %39 = math.sin %37 : tensor<1x40x128xf32>
    %cst_4 = arith.constant dense<1.000000e+00> : tensor<1xf32>
    %40 = tosa.reshape %cst_4 {new_shape = array<i64: 1, 1, 1>} : (tensor<1xf32>) -> tensor<1x1x1xf32>
    %41 = tosa.mul %38, %40 {shift = 0 : i8} : (tensor<1x40x128xf32>, tensor<1x1x1xf32>) -> tensor<1x40x128xf32>  // **************
    %cst_5 = arith.constant dense<1.000000e+00> : tensor<1xf32>
    %42 = tosa.reshape %cst_5 {new_shape = array<i64: 1, 1, 1>} : (tensor<1xf32>) -> tensor<1x1x1xf32>
    %43 = tosa.mul %39, %42 {shift = 0 : i8} : (tensor<1x40x128xf32>, tensor<1x1x1xf32>) -> tensor<1x40x128xf32>  // ***************
    %44 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32 = arith.constant 2 : i32
    %45 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg8 : tensor<1x40x4096xf32>) outs(%44 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %159 = math.fpowi %in, %c2_i32 : f32, i32
      linalg.yield %159 : f32
    } -> tensor<1x40x4096xf32>
    %46 = tosa.reduce_sum %45 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %47 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %48 = tosa.reciprocal %47 : (tensor<1xf32>) -> tensor<1xf32>
    %49 = tosa.mul %48, %46 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %50 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %51 = tosa.add %49, %50 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %52 = tosa.rsqrt %51 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %53 = tosa.mul %arg8, %52 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %54 = tosa.reshape %arg3 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %55 = tosa.mul %54, %53 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %56 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %57 = tosa.transpose %arg4, %56 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %58 = tosa.reshape %55 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_6 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %59 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%58, %57 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_6 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %60 = tosa.reshape %59 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %61 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %62 = tosa.transpose %arg5, %61 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %63 = tosa.reshape %55 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_7 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %64 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%63, %62 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_7 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %65 = tosa.reshape %64 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %66 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %67 = tosa.transpose %arg6, %66 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %68 = tosa.reshape %55 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_8 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %69 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%68, %67 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_8 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %70 = tosa.reshape %69 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %71 = tosa.reshape %60 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %72 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %73 = tosa.transpose %71, %72 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %74 = tosa.reshape %65 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %75 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %76 = tosa.transpose %74, %75 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %77 = tosa.reshape %70 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %78 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %79 = tosa.transpose %77, %78 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %80 = tosa.reshape %41 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %81 = tosa.reshape %43 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %82 = tosa.mul %73, %80 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_9 = tensor.extract_slice %73[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_10 = tensor.extract_slice %73[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %83 = tensor.empty() : tensor<1x32x40x64xf32>
    %84 = linalg.negf ins(%extracted_slice_10 : tensor<1x32x40x64xf32>) outs(%83 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %85 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice = tensor.insert_slice %84 into %85[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_11 = tensor.insert_slice %extracted_slice_9 into %inserted_slice[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %86 = tosa.mul %inserted_slice_11, %81 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %87 = tosa.add %82, %86 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %88 = tosa.mul %76, %80 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_12 = tensor.extract_slice %76[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_13 = tensor.extract_slice %76[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %89 = tensor.empty() : tensor<1x32x40x64xf32>
    %90 = linalg.negf ins(%extracted_slice_13 : tensor<1x32x40x64xf32>) outs(%89 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %91 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_14 = tensor.insert_slice %90 into %91[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_15 = tensor.insert_slice %extracted_slice_12 into %inserted_slice_14[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %92 = tosa.mul %inserted_slice_15, %81 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %93 = tosa.add %88, %92 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %94 = tosa.reshape %15 {new_shape = array<i64: 1, 40, 41>} : (tensor<40x41xf32>) -> tensor<1x40x41xf32>
    %95 = tosa.reshape %94 {new_shape = array<i64: 1, 1, 40, 41>} : (tensor<1x40x41xf32>) -> tensor<1x1x40x41xf32>
    %extracted_slice_16 = tensor.extract_slice %95[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_17 = tensor.extract_slice %extracted_slice_16[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %96 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x1x40x41xf32>}> : () -> tensor<1x1x40x41xf32>
    %97 = tosa.add %extracted_slice_17, %96 : (tensor<1x1x40x41xf32>, tensor<1x1x40x41xf32>) -> tensor<1x1x40x41xf32>
    %extracted_slice_18 = tensor.extract_slice %97[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_19 = tensor.extract_slice %extracted_slice_18[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_20 = tensor.extract_slice %extracted_slice_19[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_21 = tensor.extract_slice %extracted_slice_20[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x40xf32>
    %cst_22 = arith.constant 0.000000e+00 : f32
    %splat_23 = tensor.splat %cst_22 : tensor<40x40xf32>
    %98 = tosa.reshape %extracted_slice_21 {new_shape = array<i64: 40, 40>} : (tensor<1x1x40x40xf32>) -> tensor<40x40xf32>
    %99 = tosa.add %splat_23, %98 : (tensor<40x40xf32>, tensor<40x40xf32>) -> tensor<40x40xf32>
    %100 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %101 = tosa.transpose %93, %100 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %102 = tosa.reshape %87 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %103 = tosa.reshape %101 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %104 = tosa.matmul %102, %103 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %cst_24 = arith.constant 0.0883883461 : f32
    %splat_25 = tensor.splat %cst_24 : tensor<32x40x40xf32>
    %105 = tosa.mul %104, %splat_25 {shift = 0 : i8} : (tensor<32x40x40xf32>, tensor<32x40x40xf32>) -> tensor<32x40x40xf32>
    %106 = tosa.add %105, %99 : (tensor<32x40x40xf32>, tensor<40x40xf32>) -> tensor<32x40x40xf32>
    %107 = tosa.reduce_max %106 {axis = 2 : i32} : (tensor<32x40x40xf32>) -> tensor<32x40x1xf32>
    %108 = tosa.sub %106, %107 : (tensor<32x40x40xf32>, tensor<32x40x1xf32>) -> tensor<32x40x40xf32>
    %109 = math.exp %108 : tensor<32x40x40xf32>
    %110 = tosa.reduce_sum %109 {axis = 2 : i32} : (tensor<32x40x40xf32>) -> tensor<32x40x1xf32>
    %111 = tosa.log %110 : (tensor<32x40x1xf32>) -> tensor<32x40x1xf32>
    %112 = tosa.add %107, %111 : (tensor<32x40x1xf32>, tensor<32x40x1xf32>) -> tensor<32x40x1xf32>
    %113 = tosa.sub %106, %112 : (tensor<32x40x40xf32>, tensor<32x40x1xf32>) -> tensor<32x40x40xf32>
    %114 = math.exp %113 : tensor<32x40x40xf32>
    %115 = tosa.reshape %112 {new_shape = array<i64: 1, 32, 40>} : (tensor<32x40x1xf32>) -> tensor<1x32x40xf32>
    %116 = tosa.reshape %79 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %117 = tosa.matmul %114, %116 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %118 = tosa.reshape %117 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %119 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %120 = tosa.transpose %118, %119 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %121 = tosa.reshape %120 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %122 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %123 = tosa.transpose %arg7, %122 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %124 = tosa.reshape %121 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_26 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %125 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%124, %123 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_26 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %126 = tosa.reshape %125 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %127 = tosa.add %arg8, %126 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %128 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_27 = arith.constant 2 : i32
    %129 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%127 : tensor<1x40x4096xf32>) outs(%128 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %159 = math.fpowi %in, %c2_i32_27 : f32, i32
      linalg.yield %159 : f32
    } -> tensor<1x40x4096xf32>
    %130 = tosa.reduce_sum %129 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %131 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %132 = tosa.reciprocal %131 : (tensor<1xf32>) -> tensor<1xf32>
    %133 = tosa.mul %132, %130 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %134 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %135 = tosa.add %133, %134 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %136 = tosa.rsqrt %135 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %137 = tosa.mul %127, %136 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %138 = tosa.reshape %arg9 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %139 = tosa.mul %138, %137 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %140 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %141 = tosa.transpose %arg10, %140 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %142 = tosa.reshape %139 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_28 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %143 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%142, %141 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_28 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %144 = tosa.reshape %143 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %145 = tosa.sigmoid %144 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %146 = tosa.mul %144, %145 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %147 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %148 = tosa.transpose %arg11, %147 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %149 = tosa.reshape %139 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_29 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %150 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%149, %148 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_29 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %151 = tosa.reshape %150 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %152 = tosa.mul %146, %151 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %153 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %154 = tosa.transpose %arg12, %153 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %155 = tosa.reshape %152 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_30 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %156 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%155, %154 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_30 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %157 = tosa.reshape %156 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %158 = tosa.add %127, %157 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    return %15, %15, %15, %15, %15, %15, %15, %15, %15, %15, %15, %15, %15, %15, %15, %15, %15, %15, %15, %15, %15, %15, %15, %15, %15, %15, %15, %15, %15, %15, %15, %41, %41, %41, %41, %41, %41, %41, %41, %41, %41, %41, %41, %41, %41, %41, %41, %41, %41, %41, %41, %41, %41, %41, %41, %41, %41, %41, %41, %41, %41, %41, %43, %43, %43, %43, %43, %43, %43, %43, %43, %43, %43, %43, %43, %43, %43, %43, %43, %43, %43, %43, %43, %43, %43, %43, %43, %43, %43, %43, %43, %43, %43, %158 : tensor<40x41xf32>, tensor<40x41xf32>, tensor<40x41xf32>, tensor<40x41xf32>, tensor<40x41xf32>, tensor<40x41xf32>, tensor<40x41xf32>, tensor<40x41xf32>, tensor<40x41xf32>, tensor<40x41xf32>, tensor<40x41xf32>, tensor<40x41xf32>, tensor<40x41xf32>, tensor<40x41xf32>, tensor<40x41xf32>, tensor<40x41xf32>, tensor<40x41xf32>, tensor<40x41xf32>, tensor<40x41xf32>, tensor<40x41xf32>, tensor<40x41xf32>, tensor<40x41xf32>, tensor<40x41xf32>, tensor<40x41xf32>, tensor<40x41xf32>, tensor<40x41xf32>, tensor<40x41xf32>, tensor<40x41xf32>, tensor<40x41xf32>, tensor<40x41xf32>, tensor<40x41xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x128xf32>, tensor<1x40x4096xf32>
  }
}

