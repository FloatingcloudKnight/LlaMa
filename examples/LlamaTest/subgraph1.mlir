#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module {
  func.func @subgraph1(%arg0: tensor<1x40x4096xf32>, %arg1: tensor<4096xf32>, %arg2: tensor<4096x4096xf32>, %arg3: tensor<4096x4096xf32>, %arg4: tensor<4096x4096xf32>, %arg5: tensor<1x40x128xf32>, %arg6: tensor<1x40x128xf32>, %arg7: tensor<40x41xf32>, %arg8: tensor<4096x4096xf32>, %arg9: tensor<4096xf32>, %arg10: tensor<11008x4096xf32>, %arg11: tensor<11008x4096xf32>, %arg12: tensor<4096x11008xf32>) -> tensor<1x40x4096xf32> {
    %0 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32 = arith.constant 2 : i32
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x40x4096xf32>) outs(%0 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %115 = math.fpowi %in, %c2_i32 : f32, i32
      linalg.yield %115 : f32
    } -> tensor<1x40x4096xf32>
    %2 = tosa.reduce_sum %1 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %3 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %4 = tosa.reciprocal %3 : (tensor<1xf32>) -> tensor<1xf32>
    %5 = tosa.mul %4, %2 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %6 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %7 = tosa.add %5, %6 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %8 = tosa.rsqrt %7 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %9 = tosa.mul %arg0, %8 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %10 = tosa.reshape %arg1 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %11 = tosa.mul %10, %9 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %12 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %13 = tosa.transpose %arg2, %12 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %14 = tosa.reshape %11 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %15 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%14, %13 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %16 = tosa.reshape %15 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %17 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %18 = tosa.transpose %arg3, %17 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %19 = tosa.reshape %11 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %20 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%19, %18 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_0 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %21 = tosa.reshape %20 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %22 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %23 = tosa.transpose %arg4, %22 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %24 = tosa.reshape %11 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %25 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%24, %23 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_1 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %26 = tosa.reshape %25 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %27 = tosa.reshape %16 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %28 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %29 = tosa.transpose %27, %28 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %30 = tosa.reshape %21 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %31 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %32 = tosa.transpose %30, %31 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %33 = tosa.reshape %26 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %34 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %35 = tosa.transpose %33, %34 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %36 = tosa.reshape %arg5 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %37 = tosa.reshape %arg6 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %38 = tosa.mul %29, %36 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice = tensor.extract_slice %29[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_2 = tensor.extract_slice %29[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %39 = tensor.empty() : tensor<1x32x40x64xf32>
    %40 = linalg.negf ins(%extracted_slice_2 : tensor<1x32x40x64xf32>) outs(%39 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %41 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice = tensor.insert_slice %40 into %41[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_3 = tensor.insert_slice %extracted_slice into %inserted_slice[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %42 = tosa.mul %inserted_slice_3, %37 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %43 = tosa.add %38, %42 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %44 = tosa.mul %32, %36 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_4 = tensor.extract_slice %32[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_5 = tensor.extract_slice %32[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %45 = tensor.empty() : tensor<1x32x40x64xf32>
    %46 = linalg.negf ins(%extracted_slice_5 : tensor<1x32x40x64xf32>) outs(%45 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %47 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_6 = tensor.insert_slice %46 into %47[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_7 = tensor.insert_slice %extracted_slice_4 into %inserted_slice_6[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %48 = tosa.mul %inserted_slice_7, %37 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %49 = tosa.add %44, %48 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %50 = tosa.reshape %arg7 {new_shape = array<i64: 1, 40, 41>} : (tensor<40x41xf32>) -> tensor<1x40x41xf32>
    %51 = tosa.reshape %50 {new_shape = array<i64: 1, 1, 40, 41>} : (tensor<1x40x41xf32>) -> tensor<1x1x40x41xf32>
    %extracted_slice_8 = tensor.extract_slice %51[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_9 = tensor.extract_slice %extracted_slice_8[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %52 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x1x40x41xf32>}> : () -> tensor<1x1x40x41xf32>
    %53 = tosa.add %extracted_slice_9, %52 : (tensor<1x1x40x41xf32>, tensor<1x1x40x41xf32>) -> tensor<1x1x40x41xf32>
    %extracted_slice_10 = tensor.extract_slice %53[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_11 = tensor.extract_slice %extracted_slice_10[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_12 = tensor.extract_slice %extracted_slice_11[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_13 = tensor.extract_slice %extracted_slice_12[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x40xf32>
    %cst_14 = arith.constant 0.000000e+00 : f32
    %splat = tensor.splat %cst_14 : tensor<40x40xf32>
    %54 = tosa.reshape %extracted_slice_13 {new_shape = array<i64: 40, 40>} : (tensor<1x1x40x40xf32>) -> tensor<40x40xf32>
    %55 = tosa.add %splat, %54 : (tensor<40x40xf32>, tensor<40x40xf32>) -> tensor<40x40xf32>
    %56 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %57 = tosa.transpose %49, %56 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %58 = tosa.reshape %43 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %59 = tosa.reshape %57 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %60 = tosa.matmul %58, %59 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %cst_15 = arith.constant 0.0883883461 : f32
    %splat_16 = tensor.splat %cst_15 : tensor<32x40x40xf32>
    %61 = tosa.mul %60, %splat_16 {shift = 0 : i8} : (tensor<32x40x40xf32>, tensor<32x40x40xf32>) -> tensor<32x40x40xf32>
    %62 = tosa.add %61, %55 : (tensor<32x40x40xf32>, tensor<40x40xf32>) -> tensor<32x40x40xf32>
    %63 = tosa.reduce_max %62 {axis = 2 : i32} : (tensor<32x40x40xf32>) -> tensor<32x40x1xf32>
    %64 = tosa.sub %62, %63 : (tensor<32x40x40xf32>, tensor<32x40x1xf32>) -> tensor<32x40x40xf32>
    %65 = math.exp %64 : tensor<32x40x40xf32>
    %66 = tosa.reduce_sum %65 {axis = 2 : i32} : (tensor<32x40x40xf32>) -> tensor<32x40x1xf32>
    %67 = tosa.log %66 : (tensor<32x40x1xf32>) -> tensor<32x40x1xf32>
    %68 = tosa.add %63, %67 : (tensor<32x40x1xf32>, tensor<32x40x1xf32>) -> tensor<32x40x1xf32>
    %69 = tosa.sub %62, %68 : (tensor<32x40x40xf32>, tensor<32x40x1xf32>) -> tensor<32x40x40xf32>
    %70 = math.exp %69 : tensor<32x40x40xf32>
    %71 = tosa.reshape %68 {new_shape = array<i64: 1, 32, 40>} : (tensor<32x40x1xf32>) -> tensor<1x32x40xf32>
    %72 = tosa.reshape %35 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %73 = tosa.matmul %70, %72 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %74 = tosa.reshape %73 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %75 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %76 = tosa.transpose %74, %75 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %77 = tosa.reshape %76 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %78 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %79 = tosa.transpose %arg8, %78 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %80 = tosa.reshape %77 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_17 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %81 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%80, %79 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_17 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %82 = tosa.reshape %81 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %83 = tosa.add %arg0, %82 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %84 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_18 = arith.constant 2 : i32
    %85 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%83 : tensor<1x40x4096xf32>) outs(%84 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %115 = math.fpowi %in, %c2_i32_18 : f32, i32
      linalg.yield %115 : f32
    } -> tensor<1x40x4096xf32>
    %86 = tosa.reduce_sum %85 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %87 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %88 = tosa.reciprocal %87 : (tensor<1xf32>) -> tensor<1xf32>
    %89 = tosa.mul %88, %86 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %90 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %91 = tosa.add %89, %90 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %92 = tosa.rsqrt %91 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %93 = tosa.mul %83, %92 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %94 = tosa.reshape %arg9 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %95 = tosa.mul %94, %93 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %96 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %97 = tosa.transpose %arg10, %96 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %98 = tosa.reshape %95 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_19 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %99 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%98, %97 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_19 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %100 = tosa.reshape %99 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %101 = tosa.sigmoid %100 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %102 = tosa.mul %100, %101 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %103 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %104 = tosa.transpose %arg11, %103 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %105 = tosa.reshape %95 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_20 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %106 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%105, %104 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_20 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %107 = tosa.reshape %106 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %108 = tosa.mul %102, %107 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %109 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %110 = tosa.transpose %arg12, %109 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %111 = tosa.reshape %108 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_21 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %112 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%111, %110 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_21 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %113 = tosa.reshape %112 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %114 = tosa.add %83, %113 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    return %114 : tensor<1x40x4096xf32>
  }
}

