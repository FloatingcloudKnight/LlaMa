#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module {
  func.func @subgraph2_decode(%arg0: tensor<1x1x1536xf32>, %arg1: tensor<768x1536xf32>, %arg2: tensor<768xf32>, %arg3: tensor<128x1536xf32>, %arg4: tensor<128xf32>, %arg5: tensor<128x1536xf32>, %arg6: tensor<128xf32>, %arg7: tensor<1x1x128xf32>, %arg8: tensor<1x1x128xf32>, %arg9: tensor<1x1x1024x128xf32>, %arg10: tensor<1xi64>, %arg11: tensor<1x1x1024x128xf32>, %arg12: tensor<1x1x1x1024xi1>, %arg13: tensor<1536x768xf32>) -> (tensor<1x1x1024x128xf32>, tensor<1x1x1024x128xf32>, tensor<1x1536xf32>) {
    %0 = tosa.reshape %arg0 {new_shape = array<i64: 1, 1536>} : (tensor<1x1x1536xf32>) -> tensor<1x1536xf32>
    %1 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2 = tosa.transpose %arg1, %1 : (tensor<768x1536xf32>, tensor<2xi32>) -> tensor<1536x768xf32>
    %cst = arith.constant dense<0.000000e+00> : tensor<1x768xf32>
    %3 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%0, %2 : tensor<1x1536xf32>, tensor<1536x768xf32>) outs(%cst : tensor<1x768xf32>) -> tensor<1x768xf32>
    %4 = tosa.reshape %arg2 {new_shape = array<i64: 1, 768>} : (tensor<768xf32>) -> tensor<1x768xf32>
    %5 = tosa.add %4, %3 : (tensor<1x768xf32>, tensor<1x768xf32>) -> tensor<1x768xf32>
    %6 = tosa.reshape %5 {new_shape = array<i64: 1, 1, 768>} : (tensor<1x768xf32>) -> tensor<1x1x768xf32>
    %7 = tosa.reshape %6 {new_shape = array<i64: 1, 1, 6, 128>} : (tensor<1x1x768xf32>) -> tensor<1x1x6x128xf32>
    %8 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %9 = tosa.transpose %7, %8 : (tensor<1x1x6x128xf32>, tensor<4xi32>) -> tensor<1x6x1x128xf32>
    %10 = tosa.reshape %arg0 {new_shape = array<i64: 1, 1536>} : (tensor<1x1x1536xf32>) -> tensor<1x1536xf32>
    %11 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %12 = tosa.transpose %arg3, %11 : (tensor<128x1536xf32>, tensor<2xi32>) -> tensor<1536x128xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<1x128xf32>
    %13 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%10, %12 : tensor<1x1536xf32>, tensor<1536x128xf32>) outs(%cst_0 : tensor<1x128xf32>) -> tensor<1x128xf32>
    %14 = tosa.reshape %arg4 {new_shape = array<i64: 1, 128>} : (tensor<128xf32>) -> tensor<1x128xf32>
    %15 = tosa.add %14, %13 : (tensor<1x128xf32>, tensor<1x128xf32>) -> tensor<1x128xf32>
    %16 = tosa.reshape %15 {new_shape = array<i64: 1, 1, 128>} : (tensor<1x128xf32>) -> tensor<1x1x128xf32>
    %17 = tosa.reshape %16 {new_shape = array<i64: 1, 1, 1, 128>} : (tensor<1x1x128xf32>) -> tensor<1x1x1x128xf32>
    %18 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %19 = tosa.transpose %17, %18 : (tensor<1x1x1x128xf32>, tensor<4xi32>) -> tensor<1x1x1x128xf32>
    %20 = tosa.reshape %arg0 {new_shape = array<i64: 1, 1536>} : (tensor<1x1x1536xf32>) -> tensor<1x1536xf32>
    %21 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %22 = tosa.transpose %arg5, %21 : (tensor<128x1536xf32>, tensor<2xi32>) -> tensor<1536x128xf32>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<1x128xf32>
    %23 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%20, %22 : tensor<1x1536xf32>, tensor<1536x128xf32>) outs(%cst_1 : tensor<1x128xf32>) -> tensor<1x128xf32>
    %24 = tosa.reshape %arg6 {new_shape = array<i64: 1, 128>} : (tensor<128xf32>) -> tensor<1x128xf32>
    %25 = tosa.add %24, %23 : (tensor<1x128xf32>, tensor<1x128xf32>) -> tensor<1x128xf32>
    %26 = tosa.reshape %25 {new_shape = array<i64: 1, 1, 128>} : (tensor<1x128xf32>) -> tensor<1x1x128xf32>
    %27 = tosa.reshape %26 {new_shape = array<i64: 1, 1, 1, 128>} : (tensor<1x1x128xf32>) -> tensor<1x1x1x128xf32>
    %28 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %29 = tosa.transpose %27, %28 : (tensor<1x1x1x128xf32>, tensor<4xi32>) -> tensor<1x1x1x128xf32>
    %30 = tosa.reshape %arg7 {new_shape = array<i64: 1, 1, 1, 128>} : (tensor<1x1x128xf32>) -> tensor<1x1x1x128xf32>
    %31 = tosa.reshape %arg8 {new_shape = array<i64: 1, 1, 1, 128>} : (tensor<1x1x128xf32>) -> tensor<1x1x1x128xf32>
    %32 = tosa.mul %9, %30 : (tensor<1x6x1x128xf32>, tensor<1x1x1x128xf32>) -> tensor<1x6x1x128xf32>
    %extracted_slice = tensor.extract_slice %9[0, 0, 0, 0] [1, 6, 1, 64] [1, 1, 1, 1] : tensor<1x6x1x128xf32> to tensor<1x6x1x64xf32>
    %extracted_slice_2 = tensor.extract_slice %9[0, 0, 0, 64] [1, 6, 1, 64] [1, 1, 1, 1] : tensor<1x6x1x128xf32> to tensor<1x6x1x64xf32>
    %33 = tensor.empty() : tensor<1x6x1x64xf32>
    %34 = linalg.negf ins(%extracted_slice_2 : tensor<1x6x1x64xf32>) outs(%33 : tensor<1x6x1x64xf32>) -> tensor<1x6x1x64xf32>
    %35 = tensor.empty() : tensor<1x6x1x128xf32>
    %inserted_slice = tensor.insert_slice %34 into %35[0, 0, 0, 0] [1, 6, 1, 64] [1, 1, 1, 1] : tensor<1x6x1x64xf32> into tensor<1x6x1x128xf32>
    %inserted_slice_3 = tensor.insert_slice %extracted_slice into %inserted_slice[0, 0, 0, 64] [1, 6, 1, 64] [1, 1, 1, 1] : tensor<1x6x1x64xf32> into tensor<1x6x1x128xf32>
    %36 = tosa.mul %inserted_slice_3, %31 : (tensor<1x6x1x128xf32>, tensor<1x1x1x128xf32>) -> tensor<1x6x1x128xf32>
    %37 = tosa.add %32, %36 : (tensor<1x6x1x128xf32>, tensor<1x6x1x128xf32>) -> tensor<1x6x1x128xf32>
    %38 = tosa.mul %19, %30 : (tensor<1x1x1x128xf32>, tensor<1x1x1x128xf32>) -> tensor<1x1x1x128xf32>
    %extracted_slice_4 = tensor.extract_slice %19[0, 0, 0, 0] [1, 1, 1, 64] [1, 1, 1, 1] : tensor<1x1x1x128xf32> to tensor<1x1x1x64xf32>
    %extracted_slice_5 = tensor.extract_slice %19[0, 0, 0, 64] [1, 1, 1, 64] [1, 1, 1, 1] : tensor<1x1x1x128xf32> to tensor<1x1x1x64xf32>
    %39 = tensor.empty() : tensor<1x1x1x64xf32>
    %40 = linalg.negf ins(%extracted_slice_5 : tensor<1x1x1x64xf32>) outs(%39 : tensor<1x1x1x64xf32>) -> tensor<1x1x1x64xf32>
    %41 = tensor.empty() : tensor<1x1x1x128xf32>
    %inserted_slice_6 = tensor.insert_slice %40 into %41[0, 0, 0, 0] [1, 1, 1, 64] [1, 1, 1, 1] : tensor<1x1x1x64xf32> into tensor<1x1x1x128xf32>
    %inserted_slice_7 = tensor.insert_slice %extracted_slice_4 into %inserted_slice_6[0, 0, 0, 64] [1, 1, 1, 64] [1, 1, 1, 1] : tensor<1x1x1x64xf32> into tensor<1x1x1x128xf32>
    %42 = tosa.mul %inserted_slice_7, %31 : (tensor<1x1x1x128xf32>, tensor<1x1x1x128xf32>) -> tensor<1x1x1x128xf32>
    %43 = tosa.add %38, %42 : (tensor<1x1x1x128xf32>, tensor<1x1x1x128xf32>) -> tensor<1x1x1x128xf32>
    %44 = bufferization.to_memref %arg9 : tensor<1x1x1024x128xf32> to memref<1x1x1024x128xf32>
    %45 = bufferization.to_memref %arg10 : tensor<1xi64> to memref<1xi64>
    %46 = bufferization.to_memref %43 : tensor<1x1x1x128xf32> to memref<1x1x1x128xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1_8 = arith.constant 1 : index
    %c1_9 = arith.constant 1 : index
    %c1_10 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    scf.for %arg14 = %c0 to %c1_8 step %c1 {
      scf.for %arg15 = %c0 to %c1_9 step %c1 {
        scf.for %arg16 = %c0 to %c1_10 step %c1 {
          %91 = memref.load %45[%arg16] : memref<1xi64>
          %92 = arith.index_cast %91 : i64 to index
          scf.for %arg17 = %c0 to %c128 step %c1 {
            %93 = memref.load %46[%arg14, %arg15, %arg16, %arg17] : memref<1x1x1x128xf32>
            memref.store %93, %44[%arg14, %arg15, %92, %arg17] : memref<1x1x1024x128xf32>
          }
        }
      }
    }
    %47 = bufferization.to_tensor %44 restrict : memref<1x1x1024x128xf32> to tensor<1x1x1024x128xf32>
    %48 = bufferization.to_memref %arg11 : tensor<1x1x1024x128xf32> to memref<1x1x1024x128xf32>
    %49 = bufferization.to_memref %arg10 : tensor<1xi64> to memref<1xi64>
    %50 = bufferization.to_memref %29 : tensor<1x1x1x128xf32> to memref<1x1x1x128xf32>
    %c0_11 = arith.constant 0 : index
    %c1_12 = arith.constant 1 : index
    %c1_13 = arith.constant 1 : index
    %c1_14 = arith.constant 1 : index
    %c1_15 = arith.constant 1 : index
    %c128_16 = arith.constant 128 : index
    scf.for %arg14 = %c0_11 to %c1_13 step %c1_12 {
      scf.for %arg15 = %c0_11 to %c1_14 step %c1_12 {
        scf.for %arg16 = %c0_11 to %c1_15 step %c1_12 {
          %91 = memref.load %49[%arg16] : memref<1xi64>
          %92 = arith.index_cast %91 : i64 to index
          scf.for %arg17 = %c0_11 to %c128_16 step %c1_12 {
            %93 = memref.load %50[%arg14, %arg15, %arg16, %arg17] : memref<1x1x1x128xf32>
            memref.store %93, %48[%arg14, %arg15, %92, %arg17] : memref<1x1x1024x128xf32>
          }
        }
      }
    }
    %51 = bufferization.to_tensor %48 restrict : memref<1x1x1024x128xf32> to tensor<1x1x1024x128xf32>
    %52 = tosa.reshape %47 {new_shape = array<i64: 1, 1, 1, 1024, 128>} : (tensor<1x1x1024x128xf32>) -> tensor<1x1x1x1024x128xf32>
    %53 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x1x6x1024x128xf32>}> : () -> tensor<1x1x6x1024x128xf32>
    %54 = tosa.add %52, %53 : (tensor<1x1x1x1024x128xf32>, tensor<1x1x6x1024x128xf32>) -> tensor<1x1x6x1024x128xf32>
    %55 = tosa.reshape %54 {new_shape = array<i64: 1, 6, 1024, 128>} : (tensor<1x1x6x1024x128xf32>) -> tensor<1x6x1024x128xf32>
    %56 = tosa.reshape %51 {new_shape = array<i64: 1, 1, 1, 1024, 128>} : (tensor<1x1x1024x128xf32>) -> tensor<1x1x1x1024x128xf32>
    %57 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x1x6x1024x128xf32>}> : () -> tensor<1x1x6x1024x128xf32>
    %58 = tosa.add %56, %57 : (tensor<1x1x1x1024x128xf32>, tensor<1x1x6x1024x128xf32>) -> tensor<1x1x6x1024x128xf32>
    %59 = tosa.reshape %58 {new_shape = array<i64: 1, 6, 1024, 128>} : (tensor<1x1x6x1024x128xf32>) -> tensor<1x6x1024x128xf32>
    %cst_17 = arith.constant 0xFF800000 : f32
    %cst_18 = arith.constant 0.000000e+00 : f32
    %60 = tensor.empty() : tensor<1x1x1x1024xf32>
    %splat = tensor.splat %cst_18 : tensor<1x1x1x1024xf32>
    %splat_19 = tensor.splat %cst_17 : tensor<1x1x1x1024xf32>
    %61 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg12, %splat, %splat_19 : tensor<1x1x1x1024xi1>, tensor<1x1x1x1024xf32>, tensor<1x1x1x1024xf32>) outs(%60 : tensor<1x1x1x1024xf32>) {
    ^bb0(%in: i1, %in_25: f32, %in_26: f32, %out: f32):
      %91 = arith.select %in, %in_25, %in_26 : f32
      linalg.yield %91 : f32
    } -> tensor<1x1x1x1024xf32>
    %cst_20 = arith.constant 0.000000e+00 : f32
    %splat_21 = tensor.splat %cst_20 : tensor<1x1024xf32>
    %62 = tosa.reshape %61 {new_shape = array<i64: 1, 1024>} : (tensor<1x1x1x1024xf32>) -> tensor<1x1024xf32>
    %63 = tosa.add %splat_21, %62 : (tensor<1x1024xf32>, tensor<1x1024xf32>) -> tensor<1x1024xf32>
    %64 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %65 = tosa.transpose %55, %64 : (tensor<1x6x1024x128xf32>, tensor<4xi32>) -> tensor<1x6x128x1024xf32>
    %66 = tosa.reshape %37 {new_shape = array<i64: 6, 1, 128>} : (tensor<1x6x1x128xf32>) -> tensor<6x1x128xf32>
    %67 = tosa.reshape %65 {new_shape = array<i64: 6, 128, 1024>} : (tensor<1x6x128x1024xf32>) -> tensor<6x128x1024xf32>
    %68 = tosa.matmul %66, %67 : (tensor<6x1x128xf32>, tensor<6x128x1024xf32>) -> tensor<6x1x1024xf32>
    %cst_22 = arith.constant 0.0883883461 : f32
    %splat_23 = tensor.splat %cst_22 : tensor<6x1x1024xf32>
    %69 = tosa.mul %68, %splat_23 : (tensor<6x1x1024xf32>, tensor<6x1x1024xf32>) -> tensor<6x1x1024xf32>
    %70 = tosa.reshape %63 {new_shape = array<i64: 1, 1, 1024>} : (tensor<1x1024xf32>) -> tensor<1x1x1024xf32>
    %71 = tosa.add %69, %70 : (tensor<6x1x1024xf32>, tensor<1x1x1024xf32>) -> tensor<6x1x1024xf32>
    %72 = tosa.reduce_max %71 {axis = 2 : i32} : (tensor<6x1x1024xf32>) -> tensor<6x1x1xf32>
    %73 = tosa.sub %71, %72 : (tensor<6x1x1024xf32>, tensor<6x1x1xf32>) -> tensor<6x1x1024xf32>
    %74 = math.exp %73 : tensor<6x1x1024xf32>
    %75 = tosa.reduce_sum %74 {axis = 2 : i32} : (tensor<6x1x1024xf32>) -> tensor<6x1x1xf32>
    %76 = tosa.log %75 : (tensor<6x1x1xf32>) -> tensor<6x1x1xf32>
    %77 = tosa.add %72, %76 : (tensor<6x1x1xf32>, tensor<6x1x1xf32>) -> tensor<6x1x1xf32>
    %78 = tosa.sub %71, %77 : (tensor<6x1x1024xf32>, tensor<6x1x1xf32>) -> tensor<6x1x1024xf32>
    %79 = math.exp %78 : tensor<6x1x1024xf32>
    %80 = tosa.reshape %77 {new_shape = array<i64: 1, 6, 1>} : (tensor<6x1x1xf32>) -> tensor<1x6x1xf32>
    %81 = tosa.reshape %59 {new_shape = array<i64: 6, 1024, 128>} : (tensor<1x6x1024x128xf32>) -> tensor<6x1024x128xf32>
    %82 = tosa.matmul %79, %81 : (tensor<6x1x1024xf32>, tensor<6x1024x128xf32>) -> tensor<6x1x128xf32>
    %83 = tosa.reshape %82 {new_shape = array<i64: 1, 6, 1, 128>} : (tensor<6x1x128xf32>) -> tensor<1x6x1x128xf32>
    %84 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %85 = tosa.transpose %83, %84 : (tensor<1x6x1x128xf32>, tensor<4xi32>) -> tensor<1x1x6x128xf32>
    %86 = tosa.reshape %85 {new_shape = array<i64: 1, 1, 768>} : (tensor<1x1x6x128xf32>) -> tensor<1x1x768xf32>
    %87 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %88 = tosa.transpose %arg13, %87 : (tensor<1536x768xf32>, tensor<2xi32>) -> tensor<768x1536xf32>
    %89 = tosa.reshape %86 {new_shape = array<i64: 1, 768>} : (tensor<1x1x768xf32>) -> tensor<1x768xf32>
    %cst_24 = arith.constant dense<0.000000e+00> : tensor<1x1536xf32>
    %90 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%89, %88 : tensor<1x768xf32>, tensor<768x1536xf32>) outs(%cst_24 : tensor<1x1536xf32>) -> tensor<1x1536xf32>
    return %47, %51, %90 : tensor<1x1x1024x128xf32>, tensor<1x1x1024x128xf32>, tensor<1x1536xf32>
  }
}

