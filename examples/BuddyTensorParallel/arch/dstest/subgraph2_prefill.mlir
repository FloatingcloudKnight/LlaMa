#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module {
  func.func @subgraph2_prefill(%arg0: tensor<1x1024x1536xf32>, %arg1: tensor<768x1536xf32>, %arg2: tensor<768xf32>, %arg3: tensor<128x1536xf32>, %arg4: tensor<128xf32>, %arg5: tensor<128x1536xf32>, %arg6: tensor<128xf32>, %arg7: tensor<1x1024x128xf32>, %arg8: tensor<1x1024x128xf32>, %arg9: tensor<1x1x1024x1024xi1>, %arg10: tensor<1536x768xf32>) -> (tensor<1x1x1024x128xf32>, tensor<1x1x1024x128xf32>, tensor<1024x1536xf32>) {
    %0 = tosa.reshape %arg0 {new_shape = array<i64: 1024, 1536>} : (tensor<1x1024x1536xf32>) -> tensor<1024x1536xf32>
    %1 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2 = tosa.transpose %arg1, %1 : (tensor<768x1536xf32>, tensor<2xi32>) -> tensor<1536x768xf32>
    %cst = arith.constant dense<0.000000e+00> : tensor<1024x768xf32>
    %3 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%0, %2 : tensor<1024x1536xf32>, tensor<1536x768xf32>) outs(%cst : tensor<1024x768xf32>) -> tensor<1024x768xf32>
    %4 = tosa.reshape %arg2 {new_shape = array<i64: 1, 768>} : (tensor<768xf32>) -> tensor<1x768xf32>
    %5 = tosa.add %4, %3 : (tensor<1x768xf32>, tensor<1024x768xf32>) -> tensor<1024x768xf32>
    %6 = tosa.reshape %5 {new_shape = array<i64: 1, 1024, 768>} : (tensor<1024x768xf32>) -> tensor<1x1024x768xf32>
    %7 = tosa.reshape %6 {new_shape = array<i64: 1, 1024, 6, 128>} : (tensor<1x1024x768xf32>) -> tensor<1x1024x6x128xf32>
    %8 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %9 = tosa.transpose %7, %8 : (tensor<1x1024x6x128xf32>, tensor<4xi32>) -> tensor<1x6x1024x128xf32>
    %10 = tosa.reshape %arg0 {new_shape = array<i64: 1024, 1536>} : (tensor<1x1024x1536xf32>) -> tensor<1024x1536xf32>
    %11 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %12 = tosa.transpose %arg3, %11 : (tensor<128x1536xf32>, tensor<2xi32>) -> tensor<1536x128xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<1024x128xf32>
    %13 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%10, %12 : tensor<1024x1536xf32>, tensor<1536x128xf32>) outs(%cst_0 : tensor<1024x128xf32>) -> tensor<1024x128xf32>
    %14 = tosa.reshape %arg4 {new_shape = array<i64: 1, 128>} : (tensor<128xf32>) -> tensor<1x128xf32>
    %15 = tosa.add %14, %13 : (tensor<1x128xf32>, tensor<1024x128xf32>) -> tensor<1024x128xf32>
    %16 = tosa.reshape %15 {new_shape = array<i64: 1, 1024, 128>} : (tensor<1024x128xf32>) -> tensor<1x1024x128xf32>
    %17 = tosa.reshape %16 {new_shape = array<i64: 1, 1024, 1, 128>} : (tensor<1x1024x128xf32>) -> tensor<1x1024x1x128xf32>
    %18 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %19 = tosa.transpose %17, %18 : (tensor<1x1024x1x128xf32>, tensor<4xi32>) -> tensor<1x1x1024x128xf32>
    %20 = tosa.reshape %arg0 {new_shape = array<i64: 1024, 1536>} : (tensor<1x1024x1536xf32>) -> tensor<1024x1536xf32>
    %21 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %22 = tosa.transpose %arg5, %21 : (tensor<128x1536xf32>, tensor<2xi32>) -> tensor<1536x128xf32>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<1024x128xf32>
    %23 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%20, %22 : tensor<1024x1536xf32>, tensor<1536x128xf32>) outs(%cst_1 : tensor<1024x128xf32>) -> tensor<1024x128xf32>
    %24 = tosa.reshape %arg6 {new_shape = array<i64: 1, 128>} : (tensor<128xf32>) -> tensor<1x128xf32>
    %25 = tosa.add %24, %23 : (tensor<1x128xf32>, tensor<1024x128xf32>) -> tensor<1024x128xf32>
    %26 = tosa.reshape %25 {new_shape = array<i64: 1, 1024, 128>} : (tensor<1024x128xf32>) -> tensor<1x1024x128xf32>
    %27 = tosa.reshape %26 {new_shape = array<i64: 1, 1024, 1, 128>} : (tensor<1x1024x128xf32>) -> tensor<1x1024x1x128xf32>
    %28 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %29 = tosa.transpose %27, %28 : (tensor<1x1024x1x128xf32>, tensor<4xi32>) -> tensor<1x1x1024x128xf32>
    %30 = tosa.reshape %arg7 {new_shape = array<i64: 1, 1, 1024, 128>} : (tensor<1x1024x128xf32>) -> tensor<1x1x1024x128xf32>
    %31 = tosa.reshape %arg8 {new_shape = array<i64: 1, 1, 1024, 128>} : (tensor<1x1024x128xf32>) -> tensor<1x1x1024x128xf32>
    %32 = tosa.mul %9, %30 : (tensor<1x6x1024x128xf32>, tensor<1x1x1024x128xf32>) -> tensor<1x6x1024x128xf32>
    %extracted_slice = tensor.extract_slice %9[0, 0, 0, 0] [1, 6, 1024, 64] [1, 1, 1, 1] : tensor<1x6x1024x128xf32> to tensor<1x6x1024x64xf32>
    %extracted_slice_2 = tensor.extract_slice %9[0, 0, 0, 64] [1, 6, 1024, 64] [1, 1, 1, 1] : tensor<1x6x1024x128xf32> to tensor<1x6x1024x64xf32>
    %33 = tensor.empty() : tensor<1x6x1024x64xf32>
    %34 = linalg.negf ins(%extracted_slice_2 : tensor<1x6x1024x64xf32>) outs(%33 : tensor<1x6x1024x64xf32>) -> tensor<1x6x1024x64xf32>
    %35 = tensor.empty() : tensor<1x6x1024x128xf32>
    %inserted_slice = tensor.insert_slice %34 into %35[0, 0, 0, 0] [1, 6, 1024, 64] [1, 1, 1, 1] : tensor<1x6x1024x64xf32> into tensor<1x6x1024x128xf32>
    %inserted_slice_3 = tensor.insert_slice %extracted_slice into %inserted_slice[0, 0, 0, 64] [1, 6, 1024, 64] [1, 1, 1, 1] : tensor<1x6x1024x64xf32> into tensor<1x6x1024x128xf32>
    %36 = tosa.mul %inserted_slice_3, %31 : (tensor<1x6x1024x128xf32>, tensor<1x1x1024x128xf32>) -> tensor<1x6x1024x128xf32>
    %37 = tosa.add %32, %36 : (tensor<1x6x1024x128xf32>, tensor<1x6x1024x128xf32>) -> tensor<1x6x1024x128xf32>
    %38 = tosa.mul %19, %30 : (tensor<1x1x1024x128xf32>, tensor<1x1x1024x128xf32>) -> tensor<1x1x1024x128xf32>
    %extracted_slice_4 = tensor.extract_slice %19[0, 0, 0, 0] [1, 1, 1024, 64] [1, 1, 1, 1] : tensor<1x1x1024x128xf32> to tensor<1x1x1024x64xf32>
    %extracted_slice_5 = tensor.extract_slice %19[0, 0, 0, 64] [1, 1, 1024, 64] [1, 1, 1, 1] : tensor<1x1x1024x128xf32> to tensor<1x1x1024x64xf32>
    %39 = tensor.empty() : tensor<1x1x1024x64xf32>
    %40 = linalg.negf ins(%extracted_slice_5 : tensor<1x1x1024x64xf32>) outs(%39 : tensor<1x1x1024x64xf32>) -> tensor<1x1x1024x64xf32>
    %41 = tensor.empty() : tensor<1x1x1024x128xf32>
    %inserted_slice_6 = tensor.insert_slice %40 into %41[0, 0, 0, 0] [1, 1, 1024, 64] [1, 1, 1, 1] : tensor<1x1x1024x64xf32> into tensor<1x1x1024x128xf32>
    %inserted_slice_7 = tensor.insert_slice %extracted_slice_4 into %inserted_slice_6[0, 0, 0, 64] [1, 1, 1024, 64] [1, 1, 1, 1] : tensor<1x1x1024x64xf32> into tensor<1x1x1024x128xf32>
    %42 = tosa.mul %inserted_slice_7, %31 : (tensor<1x1x1024x128xf32>, tensor<1x1x1024x128xf32>) -> tensor<1x1x1024x128xf32>
    %43 = tosa.add %38, %42 : (tensor<1x1x1024x128xf32>, tensor<1x1x1024x128xf32>) -> tensor<1x1x1024x128xf32>
    %44 = tosa.reshape %43 {new_shape = array<i64: 1, 1, 1, 1024, 128>} : (tensor<1x1x1024x128xf32>) -> tensor<1x1x1x1024x128xf32>
    %45 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x1x6x1024x128xf32>}> : () -> tensor<1x1x6x1024x128xf32>
    %46 = tosa.add %44, %45 : (tensor<1x1x1x1024x128xf32>, tensor<1x1x6x1024x128xf32>) -> tensor<1x1x6x1024x128xf32>
    %47 = tosa.reshape %46 {new_shape = array<i64: 1, 6, 1024, 128>} : (tensor<1x1x6x1024x128xf32>) -> tensor<1x6x1024x128xf32>
    %48 = tosa.reshape %29 {new_shape = array<i64: 1, 1, 1, 1024, 128>} : (tensor<1x1x1024x128xf32>) -> tensor<1x1x1x1024x128xf32>
    %49 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x1x6x1024x128xf32>}> : () -> tensor<1x1x6x1024x128xf32>
    %50 = tosa.add %48, %49 : (tensor<1x1x1x1024x128xf32>, tensor<1x1x6x1024x128xf32>) -> tensor<1x1x6x1024x128xf32>
    %51 = tosa.reshape %50 {new_shape = array<i64: 1, 6, 1024, 128>} : (tensor<1x1x6x1024x128xf32>) -> tensor<1x6x1024x128xf32>
    %cst_8 = arith.constant 0xFF800000 : f32
    %cst_9 = arith.constant 0.000000e+00 : f32
    %52 = tensor.empty() : tensor<1x1x1024x1024xf32>
    %splat = tensor.splat %cst_9 : tensor<1x1x1024x1024xf32>
    %splat_10 = tensor.splat %cst_8 : tensor<1x1x1024x1024xf32>
    %53 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg9, %splat, %splat_10 : tensor<1x1x1024x1024xi1>, tensor<1x1x1024x1024xf32>, tensor<1x1x1024x1024xf32>) outs(%52 : tensor<1x1x1024x1024xf32>) {
    ^bb0(%in: i1, %in_16: f32, %in_17: f32, %out: f32):
      %83 = arith.select %in, %in_16, %in_17 : f32
      linalg.yield %83 : f32
    } -> tensor<1x1x1024x1024xf32>
    %cst_11 = arith.constant 0.000000e+00 : f32
    %splat_12 = tensor.splat %cst_11 : tensor<1024x1024xf32>
    %54 = tosa.reshape %53 {new_shape = array<i64: 1024, 1024>} : (tensor<1x1x1024x1024xf32>) -> tensor<1024x1024xf32>
    %55 = tosa.add %splat_12, %54 : (tensor<1024x1024xf32>, tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
    %56 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %57 = tosa.transpose %47, %56 : (tensor<1x6x1024x128xf32>, tensor<4xi32>) -> tensor<1x6x128x1024xf32>
    %58 = tosa.reshape %37 {new_shape = array<i64: 6, 1024, 128>} : (tensor<1x6x1024x128xf32>) -> tensor<6x1024x128xf32>
    %59 = tosa.reshape %57 {new_shape = array<i64: 6, 128, 1024>} : (tensor<1x6x128x1024xf32>) -> tensor<6x128x1024xf32>
    %60 = tosa.matmul %58, %59 : (tensor<6x1024x128xf32>, tensor<6x128x1024xf32>) -> tensor<6x1024x1024xf32>
    %cst_13 = arith.constant 0.0883883461 : f32
    %splat_14 = tensor.splat %cst_13 : tensor<6x1024x1024xf32>
    %61 = tosa.mul %60, %splat_14 : (tensor<6x1024x1024xf32>, tensor<6x1024x1024xf32>) -> tensor<6x1024x1024xf32>
    %62 = tosa.reshape %55 {new_shape = array<i64: 1, 1024, 1024>} : (tensor<1024x1024xf32>) -> tensor<1x1024x1024xf32>
    %63 = tosa.add %61, %62 : (tensor<6x1024x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<6x1024x1024xf32>
    %64 = tosa.reduce_max %63 {axis = 2 : i32} : (tensor<6x1024x1024xf32>) -> tensor<6x1024x1xf32>
    %65 = tosa.sub %63, %64 : (tensor<6x1024x1024xf32>, tensor<6x1024x1xf32>) -> tensor<6x1024x1024xf32>
    %66 = math.exp %65 : tensor<6x1024x1024xf32>
    %67 = tosa.reduce_sum %66 {axis = 2 : i32} : (tensor<6x1024x1024xf32>) -> tensor<6x1024x1xf32>
    %68 = tosa.log %67 : (tensor<6x1024x1xf32>) -> tensor<6x1024x1xf32>
    %69 = tosa.add %64, %68 : (tensor<6x1024x1xf32>, tensor<6x1024x1xf32>) -> tensor<6x1024x1xf32>
    %70 = tosa.sub %63, %69 : (tensor<6x1024x1024xf32>, tensor<6x1024x1xf32>) -> tensor<6x1024x1024xf32>
    %71 = math.exp %70 : tensor<6x1024x1024xf32>
    %72 = tosa.reshape %69 {new_shape = array<i64: 1, 6, 1024>} : (tensor<6x1024x1xf32>) -> tensor<1x6x1024xf32>
    %73 = tosa.reshape %51 {new_shape = array<i64: 6, 1024, 128>} : (tensor<1x6x1024x128xf32>) -> tensor<6x1024x128xf32>
    %74 = tosa.matmul %71, %73 : (tensor<6x1024x1024xf32>, tensor<6x1024x128xf32>) -> tensor<6x1024x128xf32>
    %75 = tosa.reshape %74 {new_shape = array<i64: 1, 6, 1024, 128>} : (tensor<6x1024x128xf32>) -> tensor<1x6x1024x128xf32>
    %76 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %77 = tosa.transpose %75, %76 : (tensor<1x6x1024x128xf32>, tensor<4xi32>) -> tensor<1x1024x6x128xf32>
    %78 = tosa.reshape %77 {new_shape = array<i64: 1, 1024, 768>} : (tensor<1x1024x6x128xf32>) -> tensor<1x1024x768xf32>
    %79 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %80 = tosa.transpose %arg10, %79 : (tensor<1536x768xf32>, tensor<2xi32>) -> tensor<768x1536xf32>
    %81 = tosa.reshape %78 {new_shape = array<i64: 1024, 768>} : (tensor<1x1024x768xf32>) -> tensor<1024x768xf32>
    %cst_15 = arith.constant dense<0.000000e+00> : tensor<1024x1536xf32>
    %82 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%81, %80 : tensor<1024x768xf32>, tensor<768x1536xf32>) outs(%cst_15 : tensor<1024x1536xf32>) -> tensor<1024x1536xf32>
    return %43, %29, %82 : tensor<1x1x1024x128xf32>, tensor<1x1x1024x128xf32>, tensor<1024x1536xf32>
  }
}

