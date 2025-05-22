#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module {
  func.func @subgraph0(%arg0: tensor<32000x4096xf32>, %arg1: tensor<1x40xi64>, %arg2: tensor<64xf32>, %arg3: tensor<4096xf32>, %arg4: tensor<4096x4096xf32>, %arg5: tensor<4096x4096xf32>, %arg6: tensor<4096x4096xf32>, %arg7: tensor<4096x4096xf32>, %arg8: tensor<4096xf32>, %arg9: tensor<11008x4096xf32>, %arg10: tensor<11008x4096xf32>, %arg11: tensor<4096x11008xf32>, %arg12: tensor<4096xf32>, %arg13: tensor<4096x4096xf32>, %arg14: tensor<4096x4096xf32>, %arg15: tensor<4096x4096xf32>, %arg16: tensor<4096x4096xf32>, %arg17: tensor<4096xf32>, %arg18: tensor<11008x4096xf32>, %arg19: tensor<11008x4096xf32>, %arg20: tensor<4096x11008xf32>, %arg21: tensor<4096xf32>, %arg22: tensor<4096x4096xf32>, %arg23: tensor<4096x4096xf32>, %arg24: tensor<4096x4096xf32>, %arg25: tensor<4096x4096xf32>, %arg26: tensor<4096xf32>, %arg27: tensor<11008x4096xf32>, %arg28: tensor<11008x4096xf32>, %arg29: tensor<4096x11008xf32>, %arg30: tensor<4096xf32>, %arg31: tensor<4096x4096xf32>, %arg32: tensor<4096x4096xf32>, %arg33: tensor<4096x4096xf32>, %arg34: tensor<4096x4096xf32>, %arg35: tensor<4096xf32>, %arg36: tensor<11008x4096xf32>, %arg37: tensor<11008x4096xf32>, %arg38: tensor<4096x11008xf32>, %arg39: tensor<4096xf32>, %arg40: tensor<4096x4096xf32>, %arg41: tensor<4096x4096xf32>, %arg42: tensor<4096x4096xf32>, %arg43: tensor<4096x4096xf32>, %arg44: tensor<4096xf32>, %arg45: tensor<11008x4096xf32>, %arg46: tensor<11008x4096xf32>, %arg47: tensor<4096x11008xf32>, %arg48: tensor<4096xf32>, %arg49: tensor<4096x4096xf32>, %arg50: tensor<4096x4096xf32>, %arg51: tensor<4096x4096xf32>, %arg52: tensor<4096x4096xf32>, %arg53: tensor<4096xf32>, %arg54: tensor<11008x4096xf32>, %arg55: tensor<11008x4096xf32>, %arg56: tensor<4096x11008xf32>, %arg57: tensor<4096xf32>, %arg58: tensor<4096x4096xf32>, %arg59: tensor<4096x4096xf32>, %arg60: tensor<4096x4096xf32>, %arg61: tensor<4096x4096xf32>, %arg62: tensor<4096xf32>, %arg63: tensor<11008x4096xf32>, %arg64: tensor<11008x4096xf32>, %arg65: tensor<4096x11008xf32>, %arg66: tensor<4096xf32>, %arg67: tensor<4096x4096xf32>, %arg68: tensor<4096x4096xf32>, %arg69: tensor<4096x4096xf32>, %arg70: tensor<4096x4096xf32>, %arg71: tensor<4096xf32>, %arg72: tensor<11008x4096xf32>, %arg73: tensor<11008x4096xf32>, %arg74: tensor<4096x11008xf32>, %arg75: tensor<4096xf32>, %arg76: tensor<4096x4096xf32>, %arg77: tensor<4096x4096xf32>, %arg78: tensor<4096x4096xf32>, %arg79: tensor<4096x4096xf32>, %arg80: tensor<4096xf32>, %arg81: tensor<11008x4096xf32>, %arg82: tensor<11008x4096xf32>, %arg83: tensor<4096x11008xf32>, %arg84: tensor<4096xf32>, %arg85: tensor<4096x4096xf32>, %arg86: tensor<4096x4096xf32>, %arg87: tensor<4096x4096xf32>, %arg88: tensor<4096x4096xf32>, %arg89: tensor<4096xf32>, %arg90: tensor<11008x4096xf32>, %arg91: tensor<11008x4096xf32>, %arg92: tensor<4096x11008xf32>, %arg93: tensor<4096xf32>, %arg94: tensor<4096x4096xf32>, %arg95: tensor<4096x4096xf32>, %arg96: tensor<4096x4096xf32>, %arg97: tensor<4096x4096xf32>, %arg98: tensor<4096xf32>, %arg99: tensor<11008x4096xf32>, %arg100: tensor<11008x4096xf32>, %arg101: tensor<4096x11008xf32>, %arg102: tensor<4096xf32>, %arg103: tensor<4096x4096xf32>, %arg104: tensor<4096x4096xf32>, %arg105: tensor<4096x4096xf32>, %arg106: tensor<4096x4096xf32>, %arg107: tensor<4096xf32>, %arg108: tensor<11008x4096xf32>, %arg109: tensor<11008x4096xf32>, %arg110: tensor<4096x11008xf32>, %arg111: tensor<4096xf32>, %arg112: tensor<4096x4096xf32>, %arg113: tensor<4096x4096xf32>, %arg114: tensor<4096x4096xf32>, %arg115: tensor<4096x4096xf32>, %arg116: tensor<4096xf32>, %arg117: tensor<11008x4096xf32>, %arg118: tensor<11008x4096xf32>, %arg119: tensor<4096x11008xf32>, %arg120: tensor<4096xf32>, %arg121: tensor<4096x4096xf32>, %arg122: tensor<4096x4096xf32>, %arg123: tensor<4096x4096xf32>, %arg124: tensor<4096x4096xf32>, %arg125: tensor<4096xf32>, %arg126: tensor<11008x4096xf32>, %arg127: tensor<11008x4096xf32>, %arg128: tensor<4096x11008xf32>, %arg129: tensor<4096xf32>, %arg130: tensor<4096x4096xf32>, %arg131: tensor<4096x4096xf32>, %arg132: tensor<4096x4096xf32>, %arg133: tensor<4096x4096xf32>, %arg134: tensor<4096xf32>, %arg135: tensor<11008x4096xf32>, %arg136: tensor<11008x4096xf32>, %arg137: tensor<4096x11008xf32>, %arg138: tensor<4096xf32>, %arg139: tensor<4096x4096xf32>, %arg140: tensor<4096x4096xf32>, %arg141: tensor<4096x4096xf32>, %arg142: tensor<4096x4096xf32>, %arg143: tensor<4096xf32>, %arg144: tensor<11008x4096xf32>, %arg145: tensor<11008x4096xf32>, %arg146: tensor<4096x11008xf32>, %arg147: tensor<4096xf32>, %arg148: tensor<4096x4096xf32>, %arg149: tensor<4096x4096xf32>, %arg150: tensor<4096x4096xf32>, %arg151: tensor<4096x4096xf32>, %arg152: tensor<4096xf32>, %arg153: tensor<11008x4096xf32>, %arg154: tensor<11008x4096xf32>, %arg155: tensor<4096x11008xf32>, %arg156: tensor<4096xf32>, %arg157: tensor<4096x4096xf32>, %arg158: tensor<4096x4096xf32>, %arg159: tensor<4096x4096xf32>, %arg160: tensor<4096x4096xf32>, %arg161: tensor<4096xf32>, %arg162: tensor<11008x4096xf32>, %arg163: tensor<11008x4096xf32>, %arg164: tensor<4096x11008xf32>, %arg165: tensor<4096xf32>, %arg166: tensor<4096x4096xf32>, %arg167: tensor<4096x4096xf32>, %arg168: tensor<4096x4096xf32>, %arg169: tensor<4096x4096xf32>, %arg170: tensor<4096xf32>, %arg171: tensor<11008x4096xf32>, %arg172: tensor<11008x4096xf32>, %arg173: tensor<4096x11008xf32>, %arg174: tensor<4096xf32>, %arg175: tensor<4096x4096xf32>, %arg176: tensor<4096x4096xf32>, %arg177: tensor<4096x4096xf32>, %arg178: tensor<4096x4096xf32>, %arg179: tensor<4096xf32>, %arg180: tensor<11008x4096xf32>, %arg181: tensor<11008x4096xf32>, %arg182: tensor<4096x11008xf32>, %arg183: tensor<4096xf32>, %arg184: tensor<4096x4096xf32>, %arg185: tensor<4096x4096xf32>, %arg186: tensor<4096x4096xf32>, %arg187: tensor<4096x4096xf32>, %arg188: tensor<4096xf32>, %arg189: tensor<11008x4096xf32>, %arg190: tensor<11008x4096xf32>, %arg191: tensor<4096x11008xf32>, %arg192: tensor<4096xf32>, %arg193: tensor<4096x4096xf32>, %arg194: tensor<4096x4096xf32>, %arg195: tensor<4096x4096xf32>, %arg196: tensor<4096x4096xf32>, %arg197: tensor<4096xf32>, %arg198: tensor<11008x4096xf32>, %arg199: tensor<11008x4096xf32>, %arg200: tensor<4096x11008xf32>, %arg201: tensor<4096xf32>, %arg202: tensor<4096x4096xf32>, %arg203: tensor<4096x4096xf32>, %arg204: tensor<4096x4096xf32>, %arg205: tensor<4096x4096xf32>, %arg206: tensor<4096xf32>, %arg207: tensor<11008x4096xf32>, %arg208: tensor<11008x4096xf32>, %arg209: tensor<4096x11008xf32>, %arg210: tensor<4096xf32>, %arg211: tensor<4096x4096xf32>, %arg212: tensor<4096x4096xf32>, %arg213: tensor<4096x4096xf32>, %arg214: tensor<4096x4096xf32>, %arg215: tensor<4096xf32>, %arg216: tensor<11008x4096xf32>, %arg217: tensor<11008x4096xf32>, %arg218: tensor<4096x11008xf32>, %arg219: tensor<4096xf32>, %arg220: tensor<4096x4096xf32>, %arg221: tensor<4096x4096xf32>, %arg222: tensor<4096x4096xf32>, %arg223: tensor<4096x4096xf32>, %arg224: tensor<4096xf32>, %arg225: tensor<11008x4096xf32>, %arg226: tensor<11008x4096xf32>, %arg227: tensor<4096x11008xf32>, %arg228: tensor<4096xf32>, %arg229: tensor<4096x4096xf32>, %arg230: tensor<4096x4096xf32>, %arg231: tensor<4096x4096xf32>, %arg232: tensor<4096x4096xf32>, %arg233: tensor<4096xf32>, %arg234: tensor<11008x4096xf32>, %arg235: tensor<11008x4096xf32>, %arg236: tensor<4096x11008xf32>, %arg237: tensor<4096xf32>, %arg238: tensor<4096x4096xf32>, %arg239: tensor<4096x4096xf32>, %arg240: tensor<4096x4096xf32>, %arg241: tensor<4096x4096xf32>, %arg242: tensor<4096xf32>, %arg243: tensor<11008x4096xf32>, %arg244: tensor<11008x4096xf32>, %arg245: tensor<4096x11008xf32>, %arg246: tensor<4096xf32>, %arg247: tensor<4096x4096xf32>, %arg248: tensor<4096x4096xf32>, %arg249: tensor<4096x4096xf32>, %arg250: tensor<4096x4096xf32>, %arg251: tensor<4096xf32>, %arg252: tensor<11008x4096xf32>, %arg253: tensor<11008x4096xf32>, %arg254: tensor<4096x11008xf32>, %arg255: tensor<4096xf32>, %arg256: tensor<4096x4096xf32>, %arg257: tensor<4096x4096xf32>, %arg258: tensor<4096x4096xf32>, %arg259: tensor<4096x4096xf32>, %arg260: tensor<4096xf32>, %arg261: tensor<11008x4096xf32>, %arg262: tensor<11008x4096xf32>, %arg263: tensor<4096x11008xf32>, %arg264: tensor<4096xf32>, %arg265: tensor<4096x4096xf32>, %arg266: tensor<4096x4096xf32>, %arg267: tensor<4096x4096xf32>, %arg268: tensor<4096x4096xf32>, %arg269: tensor<4096xf32>, %arg270: tensor<11008x4096xf32>, %arg271: tensor<11008x4096xf32>, %arg272: tensor<4096x11008xf32>, %arg273: tensor<4096xf32>, %arg274: tensor<4096x4096xf32>, %arg275: tensor<4096x4096xf32>, %arg276: tensor<4096x4096xf32>, %arg277: tensor<4096x4096xf32>, %arg278: tensor<4096xf32>, %arg279: tensor<11008x4096xf32>, %arg280: tensor<11008x4096xf32>, %arg281: tensor<4096x11008xf32>, %arg282: tensor<4096xf32>, %arg283: tensor<4096x4096xf32>, %arg284: tensor<4096x4096xf32>, %arg285: tensor<4096x4096xf32>, %arg286: tensor<4096x4096xf32>, %arg287: tensor<4096xf32>, %arg288: tensor<11008x4096xf32>, %arg289: tensor<11008x4096xf32>, %arg290: tensor<4096x11008xf32>, %arg291: tensor<4096xf32>, %arg292: tensor<32000x4096xf32>) -> tensor<1x40x32000xf32> {
    %0 = tosa.cast %arg1 : (tensor<1x40xi64>) -> tensor<1x40xi32>
    %1 = tosa.reshape %arg0 {new_shape = array<i64: 1, 32000, 4096>} : (tensor<32000x4096xf32>) -> tensor<1x32000x4096xf32>
    %2 = tosa.gather %1, %0 : (tensor<1x32000x4096xf32>, tensor<1x40xi32>) -> tensor<1x40x4096xf32>
    %3 = tosa.reshape %2 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
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
    %19 = tosa.mul %13, %18 {shift = 0 : i8} : (tensor<40x41xf32>, tensor<40x41xf32>) -> tensor<40x41xf32> // *******
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
    %45 = tosa.mul %42, %44 {shift = 0 : i8} : (tensor<1x40x128xf32>, tensor<1x1x1xf32>) -> tensor<1x40x128xf32>  // ***************
    %cst_5 = arith.constant dense<1.000000e+00> : tensor<1xf32>
    %46 = tosa.reshape %cst_5 {new_shape = array<i64: 1, 1, 1>} : (tensor<1xf32>) -> tensor<1x1x1xf32>
    %47 = tosa.mul %43, %46 {shift = 0 : i8} : (tensor<1x40x128xf32>, tensor<1x1x1xf32>) -> tensor<1x40x128xf32>  // ***************
    %48 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32 = arith.constant 2 : i32
    %49 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3 : tensor<1x40x4096xf32>) outs(%48 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3745 = math.fpowi %in, %c2_i32 : f32, i32
      linalg.yield %3745 : f32
    } -> tensor<1x40x4096xf32>
    %50 = tosa.reduce_sum %49 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %51 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %52 = tosa.reciprocal %51 : (tensor<1xf32>) -> tensor<1xf32>
    %53 = tosa.mul %52, %50 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %54 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %55 = tosa.add %53, %54 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %56 = tosa.rsqrt %55 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %57 = tosa.mul %3, %56 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %58 = tosa.reshape %arg3 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %59 = tosa.mul %58, %57 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %60 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %61 = tosa.transpose %arg4, %60 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %62 = tosa.reshape %59 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_6 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %63 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%62, %61 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_6 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %64 = tosa.reshape %63 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %65 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %66 = tosa.transpose %arg5, %65 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %67 = tosa.reshape %59 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_7 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %68 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%67, %66 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_7 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %69 = tosa.reshape %68 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %70 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %71 = tosa.transpose %arg6, %70 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %72 = tosa.reshape %59 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
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
    %130 = tosa.reshape %129 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %131 = tosa.add %3, %130 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %132 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_27 = arith.constant 2 : i32
    %133 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%131 : tensor<1x40x4096xf32>) outs(%132 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3745 = math.fpowi %in, %c2_i32_27 : f32, i32
      linalg.yield %3745 : f32
    } -> tensor<1x40x4096xf32>
    %134 = tosa.reduce_sum %133 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %135 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %136 = tosa.reciprocal %135 : (tensor<1xf32>) -> tensor<1xf32>
    %137 = tosa.mul %136, %134 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %138 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %139 = tosa.add %137, %138 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %140 = tosa.rsqrt %139 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %141 = tosa.mul %131, %140 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %142 = tosa.reshape %arg8 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %143 = tosa.mul %142, %141 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %144 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %145 = tosa.transpose %arg9, %144 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %146 = tosa.reshape %143 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_28 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %147 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%146, %145 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_28 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %148 = tosa.reshape %147 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %149 = tosa.sigmoid %148 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %150 = tosa.mul %148, %149 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %151 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %152 = tosa.transpose %arg10, %151 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %153 = tosa.reshape %143 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_29 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %154 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%153, %152 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_29 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %155 = tosa.reshape %154 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %156 = tosa.mul %150, %155 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %157 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %158 = tosa.transpose %arg11, %157 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %159 = tosa.reshape %156 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_30 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %160 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%159, %158 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_30 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %161 = tosa.reshape %160 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %162 = tosa.add %131, %161 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %163 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_31 = arith.constant 2 : i32
    %164 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%162 : tensor<1x40x4096xf32>) outs(%163 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3745 = math.fpowi %in, %c2_i32_31 : f32, i32
      linalg.yield %3745 : f32
    } -> tensor<1x40x4096xf32>
    %165 = tosa.reduce_sum %164 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %166 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %167 = tosa.reciprocal %166 : (tensor<1xf32>) -> tensor<1xf32>
    %168 = tosa.mul %167, %165 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %169 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %170 = tosa.add %168, %169 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %171 = tosa.rsqrt %170 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %172 = tosa.mul %162, %171 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %173 = tosa.reshape %arg12 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %174 = tosa.mul %173, %172 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %175 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %176 = tosa.transpose %arg13, %175 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %177 = tosa.reshape %174 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_32 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %178 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%177, %176 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_32 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %179 = tosa.reshape %178 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %180 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %181 = tosa.transpose %arg14, %180 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %182 = tosa.reshape %174 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_33 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %183 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%182, %181 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_33 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %184 = tosa.reshape %183 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %185 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %186 = tosa.transpose %arg15, %185 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %187 = tosa.reshape %174 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_34 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %188 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%187, %186 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_34 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %189 = tosa.reshape %188 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %190 = tosa.reshape %179 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %191 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %192 = tosa.transpose %190, %191 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %193 = tosa.reshape %184 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %194 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %195 = tosa.transpose %193, %194 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %196 = tosa.reshape %189 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %197 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %198 = tosa.transpose %196, %197 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %199 = tosa.reshape %45 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %200 = tosa.reshape %47 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %201 = tosa.mul %192, %199 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_35 = tensor.extract_slice %192[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_36 = tensor.extract_slice %192[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %202 = tensor.empty() : tensor<1x32x40x64xf32>
    %203 = linalg.negf ins(%extracted_slice_36 : tensor<1x32x40x64xf32>) outs(%202 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %204 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_37 = tensor.insert_slice %203 into %204[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_38 = tensor.insert_slice %extracted_slice_35 into %inserted_slice_37[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %205 = tosa.mul %inserted_slice_38, %200 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %206 = tosa.add %201, %205 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %207 = tosa.mul %195, %199 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_39 = tensor.extract_slice %195[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_40 = tensor.extract_slice %195[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %208 = tensor.empty() : tensor<1x32x40x64xf32>
    %209 = linalg.negf ins(%extracted_slice_40 : tensor<1x32x40x64xf32>) outs(%208 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %210 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_41 = tensor.insert_slice %209 into %210[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_42 = tensor.insert_slice %extracted_slice_39 into %inserted_slice_41[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %211 = tosa.mul %inserted_slice_42, %200 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %212 = tosa.add %207, %211 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %213 = tosa.reshape %19 {new_shape = array<i64: 1, 40, 41>} : (tensor<40x41xf32>) -> tensor<1x40x41xf32>
    %214 = tosa.reshape %213 {new_shape = array<i64: 1, 1, 40, 41>} : (tensor<1x40x41xf32>) -> tensor<1x1x40x41xf32>
    %extracted_slice_43 = tensor.extract_slice %214[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_44 = tensor.extract_slice %extracted_slice_43[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %215 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x1x40x41xf32>}> : () -> tensor<1x1x40x41xf32>
    %216 = tosa.add %extracted_slice_44, %215 : (tensor<1x1x40x41xf32>, tensor<1x1x40x41xf32>) -> tensor<1x1x40x41xf32>
    %extracted_slice_45 = tensor.extract_slice %216[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_46 = tensor.extract_slice %extracted_slice_45[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_47 = tensor.extract_slice %extracted_slice_46[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_48 = tensor.extract_slice %extracted_slice_47[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x40xf32>
    %cst_49 = arith.constant 0.000000e+00 : f32
    %splat_50 = tensor.splat %cst_49 : tensor<40x40xf32>
    %217 = tosa.reshape %extracted_slice_48 {new_shape = array<i64: 40, 40>} : (tensor<1x1x40x40xf32>) -> tensor<40x40xf32>
    %218 = tosa.add %splat_50, %217 : (tensor<40x40xf32>, tensor<40x40xf32>) -> tensor<40x40xf32>
    %219 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %220 = tosa.transpose %212, %219 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %221 = tosa.reshape %206 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %222 = tosa.reshape %220 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %223 = tosa.matmul %221, %222 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %cst_51 = arith.constant 0.0883883461 : f32
    %splat_52 = tensor.splat %cst_51 : tensor<32x40x40xf32>
    %224 = tosa.mul %223, %splat_52 {shift = 0 : i8} : (tensor<32x40x40xf32>, tensor<32x40x40xf32>) -> tensor<32x40x40xf32>
    %225 = tosa.add %224, %218 : (tensor<32x40x40xf32>, tensor<40x40xf32>) -> tensor<32x40x40xf32>
    %226 = tosa.reduce_max %225 {axis = 2 : i32} : (tensor<32x40x40xf32>) -> tensor<32x40x1xf32>
    %227 = tosa.sub %225, %226 : (tensor<32x40x40xf32>, tensor<32x40x1xf32>) -> tensor<32x40x40xf32>
    %228 = math.exp %227 : tensor<32x40x40xf32>
    %229 = tosa.reduce_sum %228 {axis = 2 : i32} : (tensor<32x40x40xf32>) -> tensor<32x40x1xf32>
    %230 = tosa.log %229 : (tensor<32x40x1xf32>) -> tensor<32x40x1xf32>
    %231 = tosa.add %226, %230 : (tensor<32x40x1xf32>, tensor<32x40x1xf32>) -> tensor<32x40x1xf32>
    %232 = tosa.sub %225, %231 : (tensor<32x40x40xf32>, tensor<32x40x1xf32>) -> tensor<32x40x40xf32>
    %233 = math.exp %232 : tensor<32x40x40xf32>
    %234 = tosa.reshape %231 {new_shape = array<i64: 1, 32, 40>} : (tensor<32x40x1xf32>) -> tensor<1x32x40xf32>
    %235 = tosa.reshape %198 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %236 = tosa.matmul %233, %235 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %237 = tosa.reshape %236 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %238 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %239 = tosa.transpose %237, %238 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %240 = tosa.reshape %239 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %241 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %242 = tosa.transpose %arg16, %241 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %243 = tosa.reshape %240 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_53 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %244 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%243, %242 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_53 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %245 = tosa.reshape %244 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %246 = tosa.add %162, %245 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %247 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_54 = arith.constant 2 : i32
    %248 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%246 : tensor<1x40x4096xf32>) outs(%247 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3745 = math.fpowi %in, %c2_i32_54 : f32, i32
      linalg.yield %3745 : f32
    } -> tensor<1x40x4096xf32>
    %249 = tosa.reduce_sum %248 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %250 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %251 = tosa.reciprocal %250 : (tensor<1xf32>) -> tensor<1xf32>
    %252 = tosa.mul %251, %249 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %253 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %254 = tosa.add %252, %253 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %255 = tosa.rsqrt %254 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %256 = tosa.mul %246, %255 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %257 = tosa.reshape %arg17 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %258 = tosa.mul %257, %256 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %259 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %260 = tosa.transpose %arg18, %259 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %261 = tosa.reshape %258 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_55 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %262 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%261, %260 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_55 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %263 = tosa.reshape %262 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %264 = tosa.sigmoid %263 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %265 = tosa.mul %263, %264 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %266 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %267 = tosa.transpose %arg19, %266 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %268 = tosa.reshape %258 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_56 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %269 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%268, %267 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_56 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %270 = tosa.reshape %269 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %271 = tosa.mul %265, %270 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %272 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %273 = tosa.transpose %arg20, %272 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %274 = tosa.reshape %271 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_57 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %275 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%274, %273 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_57 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %276 = tosa.reshape %275 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %277 = tosa.add %246, %276 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %278 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_58 = arith.constant 2 : i32
    %279 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%277 : tensor<1x40x4096xf32>) outs(%278 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3745 = math.fpowi %in, %c2_i32_58 : f32, i32
      linalg.yield %3745 : f32
    } -> tensor<1x40x4096xf32>
    %280 = tosa.reduce_sum %279 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %281 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %282 = tosa.reciprocal %281 : (tensor<1xf32>) -> tensor<1xf32>
    %283 = tosa.mul %282, %280 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %284 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %285 = tosa.add %283, %284 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %286 = tosa.rsqrt %285 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %287 = tosa.mul %277, %286 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %288 = tosa.reshape %arg21 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %289 = tosa.mul %288, %287 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %290 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %291 = tosa.transpose %arg22, %290 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %292 = tosa.reshape %289 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_59 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %293 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%292, %291 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_59 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %294 = tosa.reshape %293 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %295 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %296 = tosa.transpose %arg23, %295 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %297 = tosa.reshape %289 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_60 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %298 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%297, %296 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_60 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %299 = tosa.reshape %298 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %300 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %301 = tosa.transpose %arg24, %300 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %302 = tosa.reshape %289 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_61 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %303 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%302, %301 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_61 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %304 = tosa.reshape %303 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %305 = tosa.reshape %294 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %306 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %307 = tosa.transpose %305, %306 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %308 = tosa.reshape %299 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %309 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %310 = tosa.transpose %308, %309 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %311 = tosa.reshape %304 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %312 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %313 = tosa.transpose %311, %312 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %314 = tosa.reshape %45 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %315 = tosa.reshape %47 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %316 = tosa.mul %307, %314 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_62 = tensor.extract_slice %307[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_63 = tensor.extract_slice %307[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %317 = tensor.empty() : tensor<1x32x40x64xf32>
    %318 = linalg.negf ins(%extracted_slice_63 : tensor<1x32x40x64xf32>) outs(%317 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %319 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_64 = tensor.insert_slice %318 into %319[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_65 = tensor.insert_slice %extracted_slice_62 into %inserted_slice_64[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %320 = tosa.mul %inserted_slice_65, %315 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %321 = tosa.add %316, %320 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %322 = tosa.mul %310, %314 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_66 = tensor.extract_slice %310[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_67 = tensor.extract_slice %310[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %323 = tensor.empty() : tensor<1x32x40x64xf32>
    %324 = linalg.negf ins(%extracted_slice_67 : tensor<1x32x40x64xf32>) outs(%323 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %325 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_68 = tensor.insert_slice %324 into %325[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_69 = tensor.insert_slice %extracted_slice_66 into %inserted_slice_68[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %326 = tosa.mul %inserted_slice_69, %315 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %327 = tosa.add %322, %326 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %328 = tosa.reshape %19 {new_shape = array<i64: 1, 40, 41>} : (tensor<40x41xf32>) -> tensor<1x40x41xf32>
    %329 = tosa.reshape %328 {new_shape = array<i64: 1, 1, 40, 41>} : (tensor<1x40x41xf32>) -> tensor<1x1x40x41xf32>
    %extracted_slice_70 = tensor.extract_slice %329[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_71 = tensor.extract_slice %extracted_slice_70[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %330 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x1x40x41xf32>}> : () -> tensor<1x1x40x41xf32>
    %331 = tosa.add %extracted_slice_71, %330 : (tensor<1x1x40x41xf32>, tensor<1x1x40x41xf32>) -> tensor<1x1x40x41xf32>
    %extracted_slice_72 = tensor.extract_slice %331[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_73 = tensor.extract_slice %extracted_slice_72[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_74 = tensor.extract_slice %extracted_slice_73[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_75 = tensor.extract_slice %extracted_slice_74[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x40xf32>
    %cst_76 = arith.constant 0.000000e+00 : f32
    %splat_77 = tensor.splat %cst_76 : tensor<40x40xf32>
    %332 = tosa.reshape %extracted_slice_75 {new_shape = array<i64: 40, 40>} : (tensor<1x1x40x40xf32>) -> tensor<40x40xf32>
    %333 = tosa.add %splat_77, %332 : (tensor<40x40xf32>, tensor<40x40xf32>) -> tensor<40x40xf32>
    %334 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %335 = tosa.transpose %327, %334 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %336 = tosa.reshape %321 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %337 = tosa.reshape %335 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %338 = tosa.matmul %336, %337 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %cst_78 = arith.constant 0.0883883461 : f32
    %splat_79 = tensor.splat %cst_78 : tensor<32x40x40xf32>
    %339 = tosa.mul %338, %splat_79 {shift = 0 : i8} : (tensor<32x40x40xf32>, tensor<32x40x40xf32>) -> tensor<32x40x40xf32>
    %340 = tosa.add %339, %333 : (tensor<32x40x40xf32>, tensor<40x40xf32>) -> tensor<32x40x40xf32>
    %341 = tosa.reduce_max %340 {axis = 2 : i32} : (tensor<32x40x40xf32>) -> tensor<32x40x1xf32>
    %342 = tosa.sub %340, %341 : (tensor<32x40x40xf32>, tensor<32x40x1xf32>) -> tensor<32x40x40xf32>
    %343 = math.exp %342 : tensor<32x40x40xf32>
    %344 = tosa.reduce_sum %343 {axis = 2 : i32} : (tensor<32x40x40xf32>) -> tensor<32x40x1xf32>
    %345 = tosa.log %344 : (tensor<32x40x1xf32>) -> tensor<32x40x1xf32>
    %346 = tosa.add %341, %345 : (tensor<32x40x1xf32>, tensor<32x40x1xf32>) -> tensor<32x40x1xf32>
    %347 = tosa.sub %340, %346 : (tensor<32x40x40xf32>, tensor<32x40x1xf32>) -> tensor<32x40x40xf32>
    %348 = math.exp %347 : tensor<32x40x40xf32>
    %349 = tosa.reshape %346 {new_shape = array<i64: 1, 32, 40>} : (tensor<32x40x1xf32>) -> tensor<1x32x40xf32>
    %350 = tosa.reshape %313 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %351 = tosa.matmul %348, %350 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %352 = tosa.reshape %351 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %353 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %354 = tosa.transpose %352, %353 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %355 = tosa.reshape %354 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %356 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %357 = tosa.transpose %arg25, %356 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %358 = tosa.reshape %355 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_80 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %359 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%358, %357 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_80 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %360 = tosa.reshape %359 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %361 = tosa.add %277, %360 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %362 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_81 = arith.constant 2 : i32
    %363 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%361 : tensor<1x40x4096xf32>) outs(%362 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3745 = math.fpowi %in, %c2_i32_81 : f32, i32
      linalg.yield %3745 : f32
    } -> tensor<1x40x4096xf32>
    %364 = tosa.reduce_sum %363 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %365 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %366 = tosa.reciprocal %365 : (tensor<1xf32>) -> tensor<1xf32>
    %367 = tosa.mul %366, %364 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %368 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %369 = tosa.add %367, %368 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %370 = tosa.rsqrt %369 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %371 = tosa.mul %361, %370 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %372 = tosa.reshape %arg26 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %373 = tosa.mul %372, %371 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %374 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %375 = tosa.transpose %arg27, %374 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %376 = tosa.reshape %373 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_82 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %377 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%376, %375 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_82 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %378 = tosa.reshape %377 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %379 = tosa.sigmoid %378 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %380 = tosa.mul %378, %379 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %381 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %382 = tosa.transpose %arg28, %381 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %383 = tosa.reshape %373 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_83 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %384 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%383, %382 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_83 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %385 = tosa.reshape %384 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %386 = tosa.mul %380, %385 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %387 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %388 = tosa.transpose %arg29, %387 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %389 = tosa.reshape %386 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_84 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %390 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%389, %388 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_84 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %391 = tosa.reshape %390 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %392 = tosa.add %361, %391 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %393 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_85 = arith.constant 2 : i32
    %394 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%392 : tensor<1x40x4096xf32>) outs(%393 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3745 = math.fpowi %in, %c2_i32_85 : f32, i32
      linalg.yield %3745 : f32
    } -> tensor<1x40x4096xf32>
    %395 = tosa.reduce_sum %394 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %396 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %397 = tosa.reciprocal %396 : (tensor<1xf32>) -> tensor<1xf32>
    %398 = tosa.mul %397, %395 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %399 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %400 = tosa.add %398, %399 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %401 = tosa.rsqrt %400 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %402 = tosa.mul %392, %401 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %403 = tosa.reshape %arg30 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %404 = tosa.mul %403, %402 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %405 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %406 = tosa.transpose %arg31, %405 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %407 = tosa.reshape %404 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_86 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %408 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%407, %406 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_86 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %409 = tosa.reshape %408 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %410 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %411 = tosa.transpose %arg32, %410 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %412 = tosa.reshape %404 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_87 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %413 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%412, %411 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_87 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %414 = tosa.reshape %413 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %415 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %416 = tosa.transpose %arg33, %415 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %417 = tosa.reshape %404 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_88 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %418 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%417, %416 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_88 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %419 = tosa.reshape %418 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %420 = tosa.reshape %409 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %421 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %422 = tosa.transpose %420, %421 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %423 = tosa.reshape %414 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %424 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %425 = tosa.transpose %423, %424 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %426 = tosa.reshape %419 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %427 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %428 = tosa.transpose %426, %427 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %429 = tosa.reshape %45 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %430 = tosa.reshape %47 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %431 = tosa.mul %422, %429 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_89 = tensor.extract_slice %422[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_90 = tensor.extract_slice %422[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %432 = tensor.empty() : tensor<1x32x40x64xf32>
    %433 = linalg.negf ins(%extracted_slice_90 : tensor<1x32x40x64xf32>) outs(%432 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %434 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_91 = tensor.insert_slice %433 into %434[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_92 = tensor.insert_slice %extracted_slice_89 into %inserted_slice_91[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %435 = tosa.mul %inserted_slice_92, %430 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %436 = tosa.add %431, %435 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %437 = tosa.mul %425, %429 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_93 = tensor.extract_slice %425[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_94 = tensor.extract_slice %425[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %438 = tensor.empty() : tensor<1x32x40x64xf32>
    %439 = linalg.negf ins(%extracted_slice_94 : tensor<1x32x40x64xf32>) outs(%438 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %440 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_95 = tensor.insert_slice %439 into %440[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_96 = tensor.insert_slice %extracted_slice_93 into %inserted_slice_95[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %441 = tosa.mul %inserted_slice_96, %430 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %442 = tosa.add %437, %441 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %443 = tosa.reshape %19 {new_shape = array<i64: 1, 40, 41>} : (tensor<40x41xf32>) -> tensor<1x40x41xf32>
    %444 = tosa.reshape %443 {new_shape = array<i64: 1, 1, 40, 41>} : (tensor<1x40x41xf32>) -> tensor<1x1x40x41xf32>
    %extracted_slice_97 = tensor.extract_slice %444[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_98 = tensor.extract_slice %extracted_slice_97[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %445 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x1x40x41xf32>}> : () -> tensor<1x1x40x41xf32>
    %446 = tosa.add %extracted_slice_98, %445 : (tensor<1x1x40x41xf32>, tensor<1x1x40x41xf32>) -> tensor<1x1x40x41xf32>
    %extracted_slice_99 = tensor.extract_slice %446[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_100 = tensor.extract_slice %extracted_slice_99[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_101 = tensor.extract_slice %extracted_slice_100[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_102 = tensor.extract_slice %extracted_slice_101[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x40xf32>
    %cst_103 = arith.constant 0.000000e+00 : f32
    %splat_104 = tensor.splat %cst_103 : tensor<40x40xf32>
    %447 = tosa.reshape %extracted_slice_102 {new_shape = array<i64: 40, 40>} : (tensor<1x1x40x40xf32>) -> tensor<40x40xf32>
    %448 = tosa.add %splat_104, %447 : (tensor<40x40xf32>, tensor<40x40xf32>) -> tensor<40x40xf32>
    %449 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %450 = tosa.transpose %442, %449 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %451 = tosa.reshape %436 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %452 = tosa.reshape %450 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %453 = tosa.matmul %451, %452 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %cst_105 = arith.constant 0.0883883461 : f32
    %splat_106 = tensor.splat %cst_105 : tensor<32x40x40xf32>
    %454 = tosa.mul %453, %splat_106 {shift = 0 : i8} : (tensor<32x40x40xf32>, tensor<32x40x40xf32>) -> tensor<32x40x40xf32>
    %455 = tosa.add %454, %448 : (tensor<32x40x40xf32>, tensor<40x40xf32>) -> tensor<32x40x40xf32>
    %456 = tosa.reduce_max %455 {axis = 2 : i32} : (tensor<32x40x40xf32>) -> tensor<32x40x1xf32>
    %457 = tosa.sub %455, %456 : (tensor<32x40x40xf32>, tensor<32x40x1xf32>) -> tensor<32x40x40xf32>
    %458 = math.exp %457 : tensor<32x40x40xf32>
    %459 = tosa.reduce_sum %458 {axis = 2 : i32} : (tensor<32x40x40xf32>) -> tensor<32x40x1xf32>
    %460 = tosa.log %459 : (tensor<32x40x1xf32>) -> tensor<32x40x1xf32>
    %461 = tosa.add %456, %460 : (tensor<32x40x1xf32>, tensor<32x40x1xf32>) -> tensor<32x40x1xf32>
    %462 = tosa.sub %455, %461 : (tensor<32x40x40xf32>, tensor<32x40x1xf32>) -> tensor<32x40x40xf32>
    %463 = math.exp %462 : tensor<32x40x40xf32>
    %464 = tosa.reshape %461 {new_shape = array<i64: 1, 32, 40>} : (tensor<32x40x1xf32>) -> tensor<1x32x40xf32>
    %465 = tosa.reshape %428 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %466 = tosa.matmul %463, %465 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %467 = tosa.reshape %466 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %468 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %469 = tosa.transpose %467, %468 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %470 = tosa.reshape %469 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %471 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %472 = tosa.transpose %arg34, %471 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %473 = tosa.reshape %470 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_107 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %474 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%473, %472 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_107 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %475 = tosa.reshape %474 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %476 = tosa.add %392, %475 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %477 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_108 = arith.constant 2 : i32
    %478 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%476 : tensor<1x40x4096xf32>) outs(%477 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3745 = math.fpowi %in, %c2_i32_108 : f32, i32
      linalg.yield %3745 : f32
    } -> tensor<1x40x4096xf32>
    %479 = tosa.reduce_sum %478 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %480 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %481 = tosa.reciprocal %480 : (tensor<1xf32>) -> tensor<1xf32>
    %482 = tosa.mul %481, %479 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %483 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %484 = tosa.add %482, %483 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %485 = tosa.rsqrt %484 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %486 = tosa.mul %476, %485 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %487 = tosa.reshape %arg35 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %488 = tosa.mul %487, %486 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %489 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %490 = tosa.transpose %arg36, %489 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %491 = tosa.reshape %488 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_109 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %492 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%491, %490 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_109 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %493 = tosa.reshape %492 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %494 = tosa.sigmoid %493 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %495 = tosa.mul %493, %494 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %496 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %497 = tosa.transpose %arg37, %496 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %498 = tosa.reshape %488 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_110 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %499 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%498, %497 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_110 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %500 = tosa.reshape %499 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %501 = tosa.mul %495, %500 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %502 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %503 = tosa.transpose %arg38, %502 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %504 = tosa.reshape %501 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_111 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %505 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%504, %503 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_111 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %506 = tosa.reshape %505 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %507 = tosa.add %476, %506 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %508 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_112 = arith.constant 2 : i32
    %509 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%507 : tensor<1x40x4096xf32>) outs(%508 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3745 = math.fpowi %in, %c2_i32_112 : f32, i32
      linalg.yield %3745 : f32
    } -> tensor<1x40x4096xf32>
    %510 = tosa.reduce_sum %509 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %511 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %512 = tosa.reciprocal %511 : (tensor<1xf32>) -> tensor<1xf32>
    %513 = tosa.mul %512, %510 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %514 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %515 = tosa.add %513, %514 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %516 = tosa.rsqrt %515 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %517 = tosa.mul %507, %516 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %518 = tosa.reshape %arg39 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %519 = tosa.mul %518, %517 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %520 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %521 = tosa.transpose %arg40, %520 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %522 = tosa.reshape %519 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_113 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %523 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%522, %521 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_113 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %524 = tosa.reshape %523 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %525 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %526 = tosa.transpose %arg41, %525 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %527 = tosa.reshape %519 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_114 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %528 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%527, %526 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_114 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %529 = tosa.reshape %528 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %530 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %531 = tosa.transpose %arg42, %530 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %532 = tosa.reshape %519 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_115 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %533 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%532, %531 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_115 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %534 = tosa.reshape %533 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %535 = tosa.reshape %524 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %536 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %537 = tosa.transpose %535, %536 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %538 = tosa.reshape %529 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %539 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %540 = tosa.transpose %538, %539 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %541 = tosa.reshape %534 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %542 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %543 = tosa.transpose %541, %542 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %544 = tosa.reshape %45 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %545 = tosa.reshape %47 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %546 = tosa.mul %537, %544 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_116 = tensor.extract_slice %537[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_117 = tensor.extract_slice %537[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %547 = tensor.empty() : tensor<1x32x40x64xf32>
    %548 = linalg.negf ins(%extracted_slice_117 : tensor<1x32x40x64xf32>) outs(%547 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %549 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_118 = tensor.insert_slice %548 into %549[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_119 = tensor.insert_slice %extracted_slice_116 into %inserted_slice_118[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %550 = tosa.mul %inserted_slice_119, %545 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %551 = tosa.add %546, %550 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %552 = tosa.mul %540, %544 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_120 = tensor.extract_slice %540[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_121 = tensor.extract_slice %540[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %553 = tensor.empty() : tensor<1x32x40x64xf32>
    %554 = linalg.negf ins(%extracted_slice_121 : tensor<1x32x40x64xf32>) outs(%553 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %555 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_122 = tensor.insert_slice %554 into %555[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_123 = tensor.insert_slice %extracted_slice_120 into %inserted_slice_122[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %556 = tosa.mul %inserted_slice_123, %545 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %557 = tosa.add %552, %556 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %558 = tosa.reshape %19 {new_shape = array<i64: 1, 40, 41>} : (tensor<40x41xf32>) -> tensor<1x40x41xf32>
    %559 = tosa.reshape %558 {new_shape = array<i64: 1, 1, 40, 41>} : (tensor<1x40x41xf32>) -> tensor<1x1x40x41xf32>
    %extracted_slice_124 = tensor.extract_slice %559[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_125 = tensor.extract_slice %extracted_slice_124[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %560 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x1x40x41xf32>}> : () -> tensor<1x1x40x41xf32>
    %561 = tosa.add %extracted_slice_125, %560 : (tensor<1x1x40x41xf32>, tensor<1x1x40x41xf32>) -> tensor<1x1x40x41xf32>
    %extracted_slice_126 = tensor.extract_slice %561[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_127 = tensor.extract_slice %extracted_slice_126[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_128 = tensor.extract_slice %extracted_slice_127[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_129 = tensor.extract_slice %extracted_slice_128[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x40xf32>
    %cst_130 = arith.constant 0.000000e+00 : f32
    %splat_131 = tensor.splat %cst_130 : tensor<40x40xf32>
    %562 = tosa.reshape %extracted_slice_129 {new_shape = array<i64: 40, 40>} : (tensor<1x1x40x40xf32>) -> tensor<40x40xf32>
    %563 = tosa.add %splat_131, %562 : (tensor<40x40xf32>, tensor<40x40xf32>) -> tensor<40x40xf32>
    %564 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %565 = tosa.transpose %557, %564 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %566 = tosa.reshape %551 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %567 = tosa.reshape %565 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %568 = tosa.matmul %566, %567 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %cst_132 = arith.constant 0.0883883461 : f32
    %splat_133 = tensor.splat %cst_132 : tensor<32x40x40xf32>
    %569 = tosa.mul %568, %splat_133 {shift = 0 : i8} : (tensor<32x40x40xf32>, tensor<32x40x40xf32>) -> tensor<32x40x40xf32>
    %570 = tosa.add %569, %563 : (tensor<32x40x40xf32>, tensor<40x40xf32>) -> tensor<32x40x40xf32>
    %571 = tosa.reduce_max %570 {axis = 2 : i32} : (tensor<32x40x40xf32>) -> tensor<32x40x1xf32>
    %572 = tosa.sub %570, %571 : (tensor<32x40x40xf32>, tensor<32x40x1xf32>) -> tensor<32x40x40xf32>
    %573 = math.exp %572 : tensor<32x40x40xf32>
    %574 = tosa.reduce_sum %573 {axis = 2 : i32} : (tensor<32x40x40xf32>) -> tensor<32x40x1xf32>
    %575 = tosa.log %574 : (tensor<32x40x1xf32>) -> tensor<32x40x1xf32>
    %576 = tosa.add %571, %575 : (tensor<32x40x1xf32>, tensor<32x40x1xf32>) -> tensor<32x40x1xf32>
    %577 = tosa.sub %570, %576 : (tensor<32x40x40xf32>, tensor<32x40x1xf32>) -> tensor<32x40x40xf32>
    %578 = math.exp %577 : tensor<32x40x40xf32>
    %579 = tosa.reshape %576 {new_shape = array<i64: 1, 32, 40>} : (tensor<32x40x1xf32>) -> tensor<1x32x40xf32>
    %580 = tosa.reshape %543 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %581 = tosa.matmul %578, %580 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %582 = tosa.reshape %581 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %583 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %584 = tosa.transpose %582, %583 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %585 = tosa.reshape %584 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %586 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %587 = tosa.transpose %arg43, %586 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %588 = tosa.reshape %585 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_134 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %589 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%588, %587 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_134 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %590 = tosa.reshape %589 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %591 = tosa.add %507, %590 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %592 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_135 = arith.constant 2 : i32
    %593 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%591 : tensor<1x40x4096xf32>) outs(%592 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3745 = math.fpowi %in, %c2_i32_135 : f32, i32
      linalg.yield %3745 : f32
    } -> tensor<1x40x4096xf32>
    %594 = tosa.reduce_sum %593 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %595 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %596 = tosa.reciprocal %595 : (tensor<1xf32>) -> tensor<1xf32>
    %597 = tosa.mul %596, %594 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %598 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %599 = tosa.add %597, %598 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %600 = tosa.rsqrt %599 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %601 = tosa.mul %591, %600 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %602 = tosa.reshape %arg44 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %603 = tosa.mul %602, %601 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %604 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %605 = tosa.transpose %arg45, %604 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %606 = tosa.reshape %603 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_136 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %607 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%606, %605 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_136 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %608 = tosa.reshape %607 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %609 = tosa.sigmoid %608 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %610 = tosa.mul %608, %609 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %611 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %612 = tosa.transpose %arg46, %611 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %613 = tosa.reshape %603 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_137 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %614 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%613, %612 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_137 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %615 = tosa.reshape %614 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %616 = tosa.mul %610, %615 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %617 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %618 = tosa.transpose %arg47, %617 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %619 = tosa.reshape %616 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_138 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %620 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%619, %618 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_138 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %621 = tosa.reshape %620 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %622 = tosa.add %591, %621 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %623 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_139 = arith.constant 2 : i32
    %624 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%622 : tensor<1x40x4096xf32>) outs(%623 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3745 = math.fpowi %in, %c2_i32_139 : f32, i32
      linalg.yield %3745 : f32
    } -> tensor<1x40x4096xf32>
    %625 = tosa.reduce_sum %624 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %626 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %627 = tosa.reciprocal %626 : (tensor<1xf32>) -> tensor<1xf32>
    %628 = tosa.mul %627, %625 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %629 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %630 = tosa.add %628, %629 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %631 = tosa.rsqrt %630 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %632 = tosa.mul %622, %631 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %633 = tosa.reshape %arg48 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %634 = tosa.mul %633, %632 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %635 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %636 = tosa.transpose %arg49, %635 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %637 = tosa.reshape %634 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_140 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %638 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%637, %636 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_140 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %639 = tosa.reshape %638 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %640 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %641 = tosa.transpose %arg50, %640 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %642 = tosa.reshape %634 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_141 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %643 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%642, %641 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_141 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %644 = tosa.reshape %643 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %645 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %646 = tosa.transpose %arg51, %645 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %647 = tosa.reshape %634 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_142 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %648 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%647, %646 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_142 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %649 = tosa.reshape %648 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %650 = tosa.reshape %639 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %651 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %652 = tosa.transpose %650, %651 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %653 = tosa.reshape %644 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %654 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %655 = tosa.transpose %653, %654 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %656 = tosa.reshape %649 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %657 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %658 = tosa.transpose %656, %657 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %659 = tosa.reshape %45 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %660 = tosa.reshape %47 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %661 = tosa.mul %652, %659 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_143 = tensor.extract_slice %652[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_144 = tensor.extract_slice %652[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %662 = tensor.empty() : tensor<1x32x40x64xf32>
    %663 = linalg.negf ins(%extracted_slice_144 : tensor<1x32x40x64xf32>) outs(%662 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %664 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_145 = tensor.insert_slice %663 into %664[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_146 = tensor.insert_slice %extracted_slice_143 into %inserted_slice_145[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %665 = tosa.mul %inserted_slice_146, %660 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %666 = tosa.add %661, %665 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %667 = tosa.mul %655, %659 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_147 = tensor.extract_slice %655[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_148 = tensor.extract_slice %655[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %668 = tensor.empty() : tensor<1x32x40x64xf32>
    %669 = linalg.negf ins(%extracted_slice_148 : tensor<1x32x40x64xf32>) outs(%668 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %670 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_149 = tensor.insert_slice %669 into %670[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_150 = tensor.insert_slice %extracted_slice_147 into %inserted_slice_149[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %671 = tosa.mul %inserted_slice_150, %660 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %672 = tosa.add %667, %671 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %673 = tosa.reshape %19 {new_shape = array<i64: 1, 40, 41>} : (tensor<40x41xf32>) -> tensor<1x40x41xf32>
    %674 = tosa.reshape %673 {new_shape = array<i64: 1, 1, 40, 41>} : (tensor<1x40x41xf32>) -> tensor<1x1x40x41xf32>
    %extracted_slice_151 = tensor.extract_slice %674[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_152 = tensor.extract_slice %extracted_slice_151[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %675 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x1x40x41xf32>}> : () -> tensor<1x1x40x41xf32>
    %676 = tosa.add %extracted_slice_152, %675 : (tensor<1x1x40x41xf32>, tensor<1x1x40x41xf32>) -> tensor<1x1x40x41xf32>
    %extracted_slice_153 = tensor.extract_slice %676[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_154 = tensor.extract_slice %extracted_slice_153[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_155 = tensor.extract_slice %extracted_slice_154[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_156 = tensor.extract_slice %extracted_slice_155[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x40xf32>
    %cst_157 = arith.constant 0.000000e+00 : f32
    %splat_158 = tensor.splat %cst_157 : tensor<40x40xf32>
    %677 = tosa.reshape %extracted_slice_156 {new_shape = array<i64: 40, 40>} : (tensor<1x1x40x40xf32>) -> tensor<40x40xf32>
    %678 = tosa.add %splat_158, %677 : (tensor<40x40xf32>, tensor<40x40xf32>) -> tensor<40x40xf32>
    %679 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %680 = tosa.transpose %672, %679 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %681 = tosa.reshape %666 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %682 = tosa.reshape %680 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %683 = tosa.matmul %681, %682 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %cst_159 = arith.constant 0.0883883461 : f32
    %splat_160 = tensor.splat %cst_159 : tensor<32x40x40xf32>
    %684 = tosa.mul %683, %splat_160 {shift = 0 : i8} : (tensor<32x40x40xf32>, tensor<32x40x40xf32>) -> tensor<32x40x40xf32>
    %685 = tosa.add %684, %678 : (tensor<32x40x40xf32>, tensor<40x40xf32>) -> tensor<32x40x40xf32>
    %686 = tosa.reduce_max %685 {axis = 2 : i32} : (tensor<32x40x40xf32>) -> tensor<32x40x1xf32>
    %687 = tosa.sub %685, %686 : (tensor<32x40x40xf32>, tensor<32x40x1xf32>) -> tensor<32x40x40xf32>
    %688 = math.exp %687 : tensor<32x40x40xf32>
    %689 = tosa.reduce_sum %688 {axis = 2 : i32} : (tensor<32x40x40xf32>) -> tensor<32x40x1xf32>
    %690 = tosa.log %689 : (tensor<32x40x1xf32>) -> tensor<32x40x1xf32>
    %691 = tosa.add %686, %690 : (tensor<32x40x1xf32>, tensor<32x40x1xf32>) -> tensor<32x40x1xf32>
    %692 = tosa.sub %685, %691 : (tensor<32x40x40xf32>, tensor<32x40x1xf32>) -> tensor<32x40x40xf32>
    %693 = math.exp %692 : tensor<32x40x40xf32>
    %694 = tosa.reshape %691 {new_shape = array<i64: 1, 32, 40>} : (tensor<32x40x1xf32>) -> tensor<1x32x40xf32>
    %695 = tosa.reshape %658 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %696 = tosa.matmul %693, %695 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %697 = tosa.reshape %696 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %698 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %699 = tosa.transpose %697, %698 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %700 = tosa.reshape %699 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %701 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %702 = tosa.transpose %arg52, %701 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %703 = tosa.reshape %700 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_161 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %704 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%703, %702 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_161 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %705 = tosa.reshape %704 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %706 = tosa.add %622, %705 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %707 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_162 = arith.constant 2 : i32
    %708 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%706 : tensor<1x40x4096xf32>) outs(%707 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3745 = math.fpowi %in, %c2_i32_162 : f32, i32
      linalg.yield %3745 : f32
    } -> tensor<1x40x4096xf32>
    %709 = tosa.reduce_sum %708 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %710 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %711 = tosa.reciprocal %710 : (tensor<1xf32>) -> tensor<1xf32>
    %712 = tosa.mul %711, %709 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %713 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %714 = tosa.add %712, %713 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %715 = tosa.rsqrt %714 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %716 = tosa.mul %706, %715 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %717 = tosa.reshape %arg53 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %718 = tosa.mul %717, %716 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %719 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %720 = tosa.transpose %arg54, %719 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %721 = tosa.reshape %718 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_163 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %722 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%721, %720 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_163 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %723 = tosa.reshape %722 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %724 = tosa.sigmoid %723 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %725 = tosa.mul %723, %724 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %726 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %727 = tosa.transpose %arg55, %726 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %728 = tosa.reshape %718 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_164 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %729 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%728, %727 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_164 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %730 = tosa.reshape %729 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %731 = tosa.mul %725, %730 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %732 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %733 = tosa.transpose %arg56, %732 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %734 = tosa.reshape %731 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_165 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %735 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%734, %733 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_165 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %736 = tosa.reshape %735 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %737 = tosa.add %706, %736 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %738 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_166 = arith.constant 2 : i32
    %739 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%737 : tensor<1x40x4096xf32>) outs(%738 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3745 = math.fpowi %in, %c2_i32_166 : f32, i32
      linalg.yield %3745 : f32
    } -> tensor<1x40x4096xf32>
    %740 = tosa.reduce_sum %739 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %741 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %742 = tosa.reciprocal %741 : (tensor<1xf32>) -> tensor<1xf32>
    %743 = tosa.mul %742, %740 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %744 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %745 = tosa.add %743, %744 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %746 = tosa.rsqrt %745 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %747 = tosa.mul %737, %746 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %748 = tosa.reshape %arg57 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %749 = tosa.mul %748, %747 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %750 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %751 = tosa.transpose %arg58, %750 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %752 = tosa.reshape %749 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_167 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %753 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%752, %751 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_167 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %754 = tosa.reshape %753 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %755 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %756 = tosa.transpose %arg59, %755 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %757 = tosa.reshape %749 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_168 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %758 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%757, %756 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_168 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %759 = tosa.reshape %758 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %760 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %761 = tosa.transpose %arg60, %760 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %762 = tosa.reshape %749 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_169 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %763 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%762, %761 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_169 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %764 = tosa.reshape %763 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %765 = tosa.reshape %754 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %766 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %767 = tosa.transpose %765, %766 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %768 = tosa.reshape %759 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %769 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %770 = tosa.transpose %768, %769 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %771 = tosa.reshape %764 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %772 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %773 = tosa.transpose %771, %772 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %774 = tosa.reshape %45 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %775 = tosa.reshape %47 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %776 = tosa.mul %767, %774 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_170 = tensor.extract_slice %767[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_171 = tensor.extract_slice %767[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %777 = tensor.empty() : tensor<1x32x40x64xf32>
    %778 = linalg.negf ins(%extracted_slice_171 : tensor<1x32x40x64xf32>) outs(%777 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %779 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_172 = tensor.insert_slice %778 into %779[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_173 = tensor.insert_slice %extracted_slice_170 into %inserted_slice_172[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %780 = tosa.mul %inserted_slice_173, %775 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %781 = tosa.add %776, %780 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %782 = tosa.mul %770, %774 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_174 = tensor.extract_slice %770[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_175 = tensor.extract_slice %770[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %783 = tensor.empty() : tensor<1x32x40x64xf32>
    %784 = linalg.negf ins(%extracted_slice_175 : tensor<1x32x40x64xf32>) outs(%783 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %785 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_176 = tensor.insert_slice %784 into %785[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_177 = tensor.insert_slice %extracted_slice_174 into %inserted_slice_176[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %786 = tosa.mul %inserted_slice_177, %775 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %787 = tosa.add %782, %786 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %788 = tosa.reshape %19 {new_shape = array<i64: 1, 40, 41>} : (tensor<40x41xf32>) -> tensor<1x40x41xf32>
    %789 = tosa.reshape %788 {new_shape = array<i64: 1, 1, 40, 41>} : (tensor<1x40x41xf32>) -> tensor<1x1x40x41xf32>
    %extracted_slice_178 = tensor.extract_slice %789[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_179 = tensor.extract_slice %extracted_slice_178[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %790 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x1x40x41xf32>}> : () -> tensor<1x1x40x41xf32>
    %791 = tosa.add %extracted_slice_179, %790 : (tensor<1x1x40x41xf32>, tensor<1x1x40x41xf32>) -> tensor<1x1x40x41xf32>
    %extracted_slice_180 = tensor.extract_slice %791[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_181 = tensor.extract_slice %extracted_slice_180[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_182 = tensor.extract_slice %extracted_slice_181[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_183 = tensor.extract_slice %extracted_slice_182[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x40xf32>
    %cst_184 = arith.constant 0.000000e+00 : f32
    %splat_185 = tensor.splat %cst_184 : tensor<40x40xf32>
    %792 = tosa.reshape %extracted_slice_183 {new_shape = array<i64: 40, 40>} : (tensor<1x1x40x40xf32>) -> tensor<40x40xf32>
    %793 = tosa.add %splat_185, %792 : (tensor<40x40xf32>, tensor<40x40xf32>) -> tensor<40x40xf32>
    %794 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %795 = tosa.transpose %787, %794 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %796 = tosa.reshape %781 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %797 = tosa.reshape %795 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %798 = tosa.matmul %796, %797 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %cst_186 = arith.constant 0.0883883461 : f32
    %splat_187 = tensor.splat %cst_186 : tensor<32x40x40xf32>
    %799 = tosa.mul %798, %splat_187 {shift = 0 : i8} : (tensor<32x40x40xf32>, tensor<32x40x40xf32>) -> tensor<32x40x40xf32>
    %800 = tosa.add %799, %793 : (tensor<32x40x40xf32>, tensor<40x40xf32>) -> tensor<32x40x40xf32>
    %801 = tosa.reduce_max %800 {axis = 2 : i32} : (tensor<32x40x40xf32>) -> tensor<32x40x1xf32>
    %802 = tosa.sub %800, %801 : (tensor<32x40x40xf32>, tensor<32x40x1xf32>) -> tensor<32x40x40xf32>
    %803 = math.exp %802 : tensor<32x40x40xf32>
    %804 = tosa.reduce_sum %803 {axis = 2 : i32} : (tensor<32x40x40xf32>) -> tensor<32x40x1xf32>
    %805 = tosa.log %804 : (tensor<32x40x1xf32>) -> tensor<32x40x1xf32>
    %806 = tosa.add %801, %805 : (tensor<32x40x1xf32>, tensor<32x40x1xf32>) -> tensor<32x40x1xf32>
    %807 = tosa.sub %800, %806 : (tensor<32x40x40xf32>, tensor<32x40x1xf32>) -> tensor<32x40x40xf32>
    %808 = math.exp %807 : tensor<32x40x40xf32>
    %809 = tosa.reshape %806 {new_shape = array<i64: 1, 32, 40>} : (tensor<32x40x1xf32>) -> tensor<1x32x40xf32>
    %810 = tosa.reshape %773 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %811 = tosa.matmul %808, %810 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %812 = tosa.reshape %811 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %813 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %814 = tosa.transpose %812, %813 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %815 = tosa.reshape %814 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %816 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %817 = tosa.transpose %arg61, %816 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %818 = tosa.reshape %815 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_188 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %819 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%818, %817 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_188 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %820 = tosa.reshape %819 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %821 = tosa.add %737, %820 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %822 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_189 = arith.constant 2 : i32
    %823 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%821 : tensor<1x40x4096xf32>) outs(%822 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3745 = math.fpowi %in, %c2_i32_189 : f32, i32
      linalg.yield %3745 : f32
    } -> tensor<1x40x4096xf32>
    %824 = tosa.reduce_sum %823 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %825 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %826 = tosa.reciprocal %825 : (tensor<1xf32>) -> tensor<1xf32>
    %827 = tosa.mul %826, %824 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %828 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %829 = tosa.add %827, %828 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %830 = tosa.rsqrt %829 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %831 = tosa.mul %821, %830 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %832 = tosa.reshape %arg62 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %833 = tosa.mul %832, %831 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %834 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %835 = tosa.transpose %arg63, %834 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %836 = tosa.reshape %833 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_190 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %837 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%836, %835 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_190 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %838 = tosa.reshape %837 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %839 = tosa.sigmoid %838 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %840 = tosa.mul %838, %839 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %841 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %842 = tosa.transpose %arg64, %841 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %843 = tosa.reshape %833 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_191 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %844 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%843, %842 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_191 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %845 = tosa.reshape %844 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %846 = tosa.mul %840, %845 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %847 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %848 = tosa.transpose %arg65, %847 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %849 = tosa.reshape %846 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_192 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %850 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%849, %848 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_192 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %851 = tosa.reshape %850 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %852 = tosa.add %821, %851 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %853 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_193 = arith.constant 2 : i32
    %854 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%852 : tensor<1x40x4096xf32>) outs(%853 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3745 = math.fpowi %in, %c2_i32_193 : f32, i32
      linalg.yield %3745 : f32
    } -> tensor<1x40x4096xf32>
    %855 = tosa.reduce_sum %854 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %856 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %857 = tosa.reciprocal %856 : (tensor<1xf32>) -> tensor<1xf32>
    %858 = tosa.mul %857, %855 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %859 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %860 = tosa.add %858, %859 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %861 = tosa.rsqrt %860 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %862 = tosa.mul %852, %861 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %863 = tosa.reshape %arg66 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %864 = tosa.mul %863, %862 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %865 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %866 = tosa.transpose %arg67, %865 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %867 = tosa.reshape %864 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_194 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %868 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%867, %866 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_194 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %869 = tosa.reshape %868 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %870 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %871 = tosa.transpose %arg68, %870 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %872 = tosa.reshape %864 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_195 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %873 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%872, %871 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_195 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %874 = tosa.reshape %873 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %875 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %876 = tosa.transpose %arg69, %875 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %877 = tosa.reshape %864 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_196 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %878 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%877, %876 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_196 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %879 = tosa.reshape %878 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %880 = tosa.reshape %869 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %881 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %882 = tosa.transpose %880, %881 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %883 = tosa.reshape %874 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %884 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %885 = tosa.transpose %883, %884 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %886 = tosa.reshape %879 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %887 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %888 = tosa.transpose %886, %887 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %889 = tosa.reshape %45 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %890 = tosa.reshape %47 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %891 = tosa.mul %882, %889 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_197 = tensor.extract_slice %882[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_198 = tensor.extract_slice %882[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %892 = tensor.empty() : tensor<1x32x40x64xf32>
    %893 = linalg.negf ins(%extracted_slice_198 : tensor<1x32x40x64xf32>) outs(%892 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %894 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_199 = tensor.insert_slice %893 into %894[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_200 = tensor.insert_slice %extracted_slice_197 into %inserted_slice_199[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %895 = tosa.mul %inserted_slice_200, %890 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %896 = tosa.add %891, %895 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %897 = tosa.mul %885, %889 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_201 = tensor.extract_slice %885[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_202 = tensor.extract_slice %885[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %898 = tensor.empty() : tensor<1x32x40x64xf32>
    %899 = linalg.negf ins(%extracted_slice_202 : tensor<1x32x40x64xf32>) outs(%898 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %900 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_203 = tensor.insert_slice %899 into %900[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_204 = tensor.insert_slice %extracted_slice_201 into %inserted_slice_203[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %901 = tosa.mul %inserted_slice_204, %890 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %902 = tosa.add %897, %901 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %903 = tosa.reshape %19 {new_shape = array<i64: 1, 40, 41>} : (tensor<40x41xf32>) -> tensor<1x40x41xf32>
    %904 = tosa.reshape %903 {new_shape = array<i64: 1, 1, 40, 41>} : (tensor<1x40x41xf32>) -> tensor<1x1x40x41xf32>
    %extracted_slice_205 = tensor.extract_slice %904[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_206 = tensor.extract_slice %extracted_slice_205[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %905 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x1x40x41xf32>}> : () -> tensor<1x1x40x41xf32>
    %906 = tosa.add %extracted_slice_206, %905 : (tensor<1x1x40x41xf32>, tensor<1x1x40x41xf32>) -> tensor<1x1x40x41xf32>
    %extracted_slice_207 = tensor.extract_slice %906[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_208 = tensor.extract_slice %extracted_slice_207[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_209 = tensor.extract_slice %extracted_slice_208[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_210 = tensor.extract_slice %extracted_slice_209[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x40xf32>
    %cst_211 = arith.constant 0.000000e+00 : f32
    %splat_212 = tensor.splat %cst_211 : tensor<40x40xf32>
    %907 = tosa.reshape %extracted_slice_210 {new_shape = array<i64: 40, 40>} : (tensor<1x1x40x40xf32>) -> tensor<40x40xf32>
    %908 = tosa.add %splat_212, %907 : (tensor<40x40xf32>, tensor<40x40xf32>) -> tensor<40x40xf32>
    %909 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %910 = tosa.transpose %902, %909 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %911 = tosa.reshape %896 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %912 = tosa.reshape %910 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %913 = tosa.matmul %911, %912 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %cst_213 = arith.constant 0.0883883461 : f32
    %splat_214 = tensor.splat %cst_213 : tensor<32x40x40xf32>
    %914 = tosa.mul %913, %splat_214 {shift = 0 : i8} : (tensor<32x40x40xf32>, tensor<32x40x40xf32>) -> tensor<32x40x40xf32>
    %915 = tosa.add %914, %908 : (tensor<32x40x40xf32>, tensor<40x40xf32>) -> tensor<32x40x40xf32>
    %916 = tosa.reduce_max %915 {axis = 2 : i32} : (tensor<32x40x40xf32>) -> tensor<32x40x1xf32>
    %917 = tosa.sub %915, %916 : (tensor<32x40x40xf32>, tensor<32x40x1xf32>) -> tensor<32x40x40xf32>
    %918 = math.exp %917 : tensor<32x40x40xf32>
    %919 = tosa.reduce_sum %918 {axis = 2 : i32} : (tensor<32x40x40xf32>) -> tensor<32x40x1xf32>
    %920 = tosa.log %919 : (tensor<32x40x1xf32>) -> tensor<32x40x1xf32>
    %921 = tosa.add %916, %920 : (tensor<32x40x1xf32>, tensor<32x40x1xf32>) -> tensor<32x40x1xf32>
    %922 = tosa.sub %915, %921 : (tensor<32x40x40xf32>, tensor<32x40x1xf32>) -> tensor<32x40x40xf32>
    %923 = math.exp %922 : tensor<32x40x40xf32>
    %924 = tosa.reshape %921 {new_shape = array<i64: 1, 32, 40>} : (tensor<32x40x1xf32>) -> tensor<1x32x40xf32>
    %925 = tosa.reshape %888 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %926 = tosa.matmul %923, %925 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %927 = tosa.reshape %926 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %928 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %929 = tosa.transpose %927, %928 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %930 = tosa.reshape %929 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %931 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %932 = tosa.transpose %arg70, %931 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %933 = tosa.reshape %930 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_215 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %934 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%933, %932 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_215 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %935 = tosa.reshape %934 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %936 = tosa.add %852, %935 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %937 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_216 = arith.constant 2 : i32
    %938 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%936 : tensor<1x40x4096xf32>) outs(%937 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3745 = math.fpowi %in, %c2_i32_216 : f32, i32
      linalg.yield %3745 : f32
    } -> tensor<1x40x4096xf32>
    %939 = tosa.reduce_sum %938 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %940 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %941 = tosa.reciprocal %940 : (tensor<1xf32>) -> tensor<1xf32>
    %942 = tosa.mul %941, %939 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %943 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %944 = tosa.add %942, %943 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %945 = tosa.rsqrt %944 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %946 = tosa.mul %936, %945 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %947 = tosa.reshape %arg71 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %948 = tosa.mul %947, %946 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %949 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %950 = tosa.transpose %arg72, %949 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %951 = tosa.reshape %948 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_217 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %952 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%951, %950 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_217 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %953 = tosa.reshape %952 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %954 = tosa.sigmoid %953 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %955 = tosa.mul %953, %954 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %956 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %957 = tosa.transpose %arg73, %956 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %958 = tosa.reshape %948 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_218 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %959 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%958, %957 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_218 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %960 = tosa.reshape %959 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %961 = tosa.mul %955, %960 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %962 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %963 = tosa.transpose %arg74, %962 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %964 = tosa.reshape %961 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_219 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %965 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%964, %963 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_219 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %966 = tosa.reshape %965 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %967 = tosa.add %936, %966 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %968 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_220 = arith.constant 2 : i32
    %969 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%967 : tensor<1x40x4096xf32>) outs(%968 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3745 = math.fpowi %in, %c2_i32_220 : f32, i32
      linalg.yield %3745 : f32
    } -> tensor<1x40x4096xf32>
    %970 = tosa.reduce_sum %969 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %971 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %972 = tosa.reciprocal %971 : (tensor<1xf32>) -> tensor<1xf32>
    %973 = tosa.mul %972, %970 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %974 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %975 = tosa.add %973, %974 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %976 = tosa.rsqrt %975 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %977 = tosa.mul %967, %976 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %978 = tosa.reshape %arg75 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %979 = tosa.mul %978, %977 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %980 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %981 = tosa.transpose %arg76, %980 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %982 = tosa.reshape %979 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_221 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %983 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%982, %981 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_221 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %984 = tosa.reshape %983 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %985 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %986 = tosa.transpose %arg77, %985 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %987 = tosa.reshape %979 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_222 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %988 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%987, %986 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_222 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %989 = tosa.reshape %988 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %990 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %991 = tosa.transpose %arg78, %990 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %992 = tosa.reshape %979 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_223 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %993 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%992, %991 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_223 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %994 = tosa.reshape %993 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %995 = tosa.reshape %984 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %996 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %997 = tosa.transpose %995, %996 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %998 = tosa.reshape %989 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %999 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1000 = tosa.transpose %998, %999 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %1001 = tosa.reshape %994 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %1002 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1003 = tosa.transpose %1001, %1002 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %1004 = tosa.reshape %45 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %1005 = tosa.reshape %47 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %1006 = tosa.mul %997, %1004 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_224 = tensor.extract_slice %997[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_225 = tensor.extract_slice %997[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %1007 = tensor.empty() : tensor<1x32x40x64xf32>
    %1008 = linalg.negf ins(%extracted_slice_225 : tensor<1x32x40x64xf32>) outs(%1007 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %1009 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_226 = tensor.insert_slice %1008 into %1009[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_227 = tensor.insert_slice %extracted_slice_224 into %inserted_slice_226[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %1010 = tosa.mul %inserted_slice_227, %1005 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1011 = tosa.add %1006, %1010 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1012 = tosa.mul %1000, %1004 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_228 = tensor.extract_slice %1000[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_229 = tensor.extract_slice %1000[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %1013 = tensor.empty() : tensor<1x32x40x64xf32>
    %1014 = linalg.negf ins(%extracted_slice_229 : tensor<1x32x40x64xf32>) outs(%1013 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %1015 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_230 = tensor.insert_slice %1014 into %1015[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_231 = tensor.insert_slice %extracted_slice_228 into %inserted_slice_230[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %1016 = tosa.mul %inserted_slice_231, %1005 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1017 = tosa.add %1012, %1016 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1018 = tosa.reshape %19 {new_shape = array<i64: 1, 40, 41>} : (tensor<40x41xf32>) -> tensor<1x40x41xf32>
    %1019 = tosa.reshape %1018 {new_shape = array<i64: 1, 1, 40, 41>} : (tensor<1x40x41xf32>) -> tensor<1x1x40x41xf32>
    %extracted_slice_232 = tensor.extract_slice %1019[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_233 = tensor.extract_slice %extracted_slice_232[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %1020 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x1x40x41xf32>}> : () -> tensor<1x1x40x41xf32>
    %1021 = tosa.add %extracted_slice_233, %1020 : (tensor<1x1x40x41xf32>, tensor<1x1x40x41xf32>) -> tensor<1x1x40x41xf32>
    %extracted_slice_234 = tensor.extract_slice %1021[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_235 = tensor.extract_slice %extracted_slice_234[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_236 = tensor.extract_slice %extracted_slice_235[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_237 = tensor.extract_slice %extracted_slice_236[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x40xf32>
    %cst_238 = arith.constant 0.000000e+00 : f32
    %splat_239 = tensor.splat %cst_238 : tensor<40x40xf32>
    %1022 = tosa.reshape %extracted_slice_237 {new_shape = array<i64: 40, 40>} : (tensor<1x1x40x40xf32>) -> tensor<40x40xf32>
    %1023 = tosa.add %splat_239, %1022 : (tensor<40x40xf32>, tensor<40x40xf32>) -> tensor<40x40xf32>
    %1024 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1025 = tosa.transpose %1017, %1024 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %1026 = tosa.reshape %1011 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %1027 = tosa.reshape %1025 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %1028 = tosa.matmul %1026, %1027 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %cst_240 = arith.constant 0.0883883461 : f32
    %splat_241 = tensor.splat %cst_240 : tensor<32x40x40xf32>
    %1029 = tosa.mul %1028, %splat_241 {shift = 0 : i8} : (tensor<32x40x40xf32>, tensor<32x40x40xf32>) -> tensor<32x40x40xf32>
    %1030 = tosa.add %1029, %1023 : (tensor<32x40x40xf32>, tensor<40x40xf32>) -> tensor<32x40x40xf32>
    %1031 = tosa.reduce_max %1030 {axis = 2 : i32} : (tensor<32x40x40xf32>) -> tensor<32x40x1xf32>
    %1032 = tosa.sub %1030, %1031 : (tensor<32x40x40xf32>, tensor<32x40x1xf32>) -> tensor<32x40x40xf32>
    %1033 = math.exp %1032 : tensor<32x40x40xf32>
    %1034 = tosa.reduce_sum %1033 {axis = 2 : i32} : (tensor<32x40x40xf32>) -> tensor<32x40x1xf32>
    %1035 = tosa.log %1034 : (tensor<32x40x1xf32>) -> tensor<32x40x1xf32>
    %1036 = tosa.add %1031, %1035 : (tensor<32x40x1xf32>, tensor<32x40x1xf32>) -> tensor<32x40x1xf32>
    %1037 = tosa.sub %1030, %1036 : (tensor<32x40x40xf32>, tensor<32x40x1xf32>) -> tensor<32x40x40xf32>
    %1038 = math.exp %1037 : tensor<32x40x40xf32>
    %1039 = tosa.reshape %1036 {new_shape = array<i64: 1, 32, 40>} : (tensor<32x40x1xf32>) -> tensor<1x32x40xf32>
    %1040 = tosa.reshape %1003 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %1041 = tosa.matmul %1038, %1040 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %1042 = tosa.reshape %1041 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1043 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1044 = tosa.transpose %1042, %1043 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %1045 = tosa.reshape %1044 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %1046 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1047 = tosa.transpose %arg79, %1046 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1048 = tosa.reshape %1045 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_242 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1049 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1048, %1047 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_242 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1050 = tosa.reshape %1049 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1051 = tosa.add %967, %1050 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1052 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_243 = arith.constant 2 : i32
    %1053 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1051 : tensor<1x40x4096xf32>) outs(%1052 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3745 = math.fpowi %in, %c2_i32_243 : f32, i32
      linalg.yield %3745 : f32
    } -> tensor<1x40x4096xf32>
    %1054 = tosa.reduce_sum %1053 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %1055 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1056 = tosa.reciprocal %1055 : (tensor<1xf32>) -> tensor<1xf32>
    %1057 = tosa.mul %1056, %1054 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1058 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %1059 = tosa.add %1057, %1058 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1060 = tosa.rsqrt %1059 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1061 = tosa.mul %1051, %1060 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %1062 = tosa.reshape %arg80 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %1063 = tosa.mul %1062, %1061 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1064 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1065 = tosa.transpose %arg81, %1064 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %1066 = tosa.reshape %1063 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_244 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %1067 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1066, %1065 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_244 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %1068 = tosa.reshape %1067 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %1069 = tosa.sigmoid %1068 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %1070 = tosa.mul %1068, %1069 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %1071 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1072 = tosa.transpose %arg82, %1071 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %1073 = tosa.reshape %1063 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_245 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %1074 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1073, %1072 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_245 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %1075 = tosa.reshape %1074 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %1076 = tosa.mul %1070, %1075 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %1077 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1078 = tosa.transpose %arg83, %1077 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %1079 = tosa.reshape %1076 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_246 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1080 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1079, %1078 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_246 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1081 = tosa.reshape %1080 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1082 = tosa.add %1051, %1081 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1083 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_247 = arith.constant 2 : i32
    %1084 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1082 : tensor<1x40x4096xf32>) outs(%1083 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3745 = math.fpowi %in, %c2_i32_247 : f32, i32
      linalg.yield %3745 : f32
    } -> tensor<1x40x4096xf32>
    %1085 = tosa.reduce_sum %1084 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %1086 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1087 = tosa.reciprocal %1086 : (tensor<1xf32>) -> tensor<1xf32>
    %1088 = tosa.mul %1087, %1085 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1089 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %1090 = tosa.add %1088, %1089 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1091 = tosa.rsqrt %1090 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1092 = tosa.mul %1082, %1091 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %1093 = tosa.reshape %arg84 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %1094 = tosa.mul %1093, %1092 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1095 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1096 = tosa.transpose %arg85, %1095 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1097 = tosa.reshape %1094 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_248 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1098 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1097, %1096 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_248 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1099 = tosa.reshape %1098 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1100 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1101 = tosa.transpose %arg86, %1100 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1102 = tosa.reshape %1094 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_249 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1103 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1102, %1101 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_249 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1104 = tosa.reshape %1103 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1105 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1106 = tosa.transpose %arg87, %1105 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1107 = tosa.reshape %1094 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_250 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1108 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1107, %1106 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_250 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1109 = tosa.reshape %1108 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1110 = tosa.reshape %1099 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %1111 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1112 = tosa.transpose %1110, %1111 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %1113 = tosa.reshape %1104 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %1114 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1115 = tosa.transpose %1113, %1114 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %1116 = tosa.reshape %1109 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %1117 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1118 = tosa.transpose %1116, %1117 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %1119 = tosa.reshape %45 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %1120 = tosa.reshape %47 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %1121 = tosa.mul %1112, %1119 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_251 = tensor.extract_slice %1112[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_252 = tensor.extract_slice %1112[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %1122 = tensor.empty() : tensor<1x32x40x64xf32>
    %1123 = linalg.negf ins(%extracted_slice_252 : tensor<1x32x40x64xf32>) outs(%1122 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %1124 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_253 = tensor.insert_slice %1123 into %1124[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_254 = tensor.insert_slice %extracted_slice_251 into %inserted_slice_253[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %1125 = tosa.mul %inserted_slice_254, %1120 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1126 = tosa.add %1121, %1125 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1127 = tosa.mul %1115, %1119 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_255 = tensor.extract_slice %1115[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_256 = tensor.extract_slice %1115[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %1128 = tensor.empty() : tensor<1x32x40x64xf32>
    %1129 = linalg.negf ins(%extracted_slice_256 : tensor<1x32x40x64xf32>) outs(%1128 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %1130 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_257 = tensor.insert_slice %1129 into %1130[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_258 = tensor.insert_slice %extracted_slice_255 into %inserted_slice_257[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %1131 = tosa.mul %inserted_slice_258, %1120 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1132 = tosa.add %1127, %1131 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1133 = tosa.reshape %19 {new_shape = array<i64: 1, 40, 41>} : (tensor<40x41xf32>) -> tensor<1x40x41xf32>
    %1134 = tosa.reshape %1133 {new_shape = array<i64: 1, 1, 40, 41>} : (tensor<1x40x41xf32>) -> tensor<1x1x40x41xf32>
    %extracted_slice_259 = tensor.extract_slice %1134[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_260 = tensor.extract_slice %extracted_slice_259[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %1135 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x1x40x41xf32>}> : () -> tensor<1x1x40x41xf32>
    %1136 = tosa.add %extracted_slice_260, %1135 : (tensor<1x1x40x41xf32>, tensor<1x1x40x41xf32>) -> tensor<1x1x40x41xf32>
    %extracted_slice_261 = tensor.extract_slice %1136[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_262 = tensor.extract_slice %extracted_slice_261[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_263 = tensor.extract_slice %extracted_slice_262[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_264 = tensor.extract_slice %extracted_slice_263[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x40xf32>
    %cst_265 = arith.constant 0.000000e+00 : f32
    %splat_266 = tensor.splat %cst_265 : tensor<40x40xf32>
    %1137 = tosa.reshape %extracted_slice_264 {new_shape = array<i64: 40, 40>} : (tensor<1x1x40x40xf32>) -> tensor<40x40xf32>
    %1138 = tosa.add %splat_266, %1137 : (tensor<40x40xf32>, tensor<40x40xf32>) -> tensor<40x40xf32>
    %1139 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1140 = tosa.transpose %1132, %1139 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %1141 = tosa.reshape %1126 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %1142 = tosa.reshape %1140 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %1143 = tosa.matmul %1141, %1142 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %cst_267 = arith.constant 0.0883883461 : f32
    %splat_268 = tensor.splat %cst_267 : tensor<32x40x40xf32>
    %1144 = tosa.mul %1143, %splat_268 {shift = 0 : i8} : (tensor<32x40x40xf32>, tensor<32x40x40xf32>) -> tensor<32x40x40xf32>
    %1145 = tosa.add %1144, %1138 : (tensor<32x40x40xf32>, tensor<40x40xf32>) -> tensor<32x40x40xf32>
    %1146 = tosa.reduce_max %1145 {axis = 2 : i32} : (tensor<32x40x40xf32>) -> tensor<32x40x1xf32>
    %1147 = tosa.sub %1145, %1146 : (tensor<32x40x40xf32>, tensor<32x40x1xf32>) -> tensor<32x40x40xf32>
    %1148 = math.exp %1147 : tensor<32x40x40xf32>
    %1149 = tosa.reduce_sum %1148 {axis = 2 : i32} : (tensor<32x40x40xf32>) -> tensor<32x40x1xf32>
    %1150 = tosa.log %1149 : (tensor<32x40x1xf32>) -> tensor<32x40x1xf32>
    %1151 = tosa.add %1146, %1150 : (tensor<32x40x1xf32>, tensor<32x40x1xf32>) -> tensor<32x40x1xf32>
    %1152 = tosa.sub %1145, %1151 : (tensor<32x40x40xf32>, tensor<32x40x1xf32>) -> tensor<32x40x40xf32>
    %1153 = math.exp %1152 : tensor<32x40x40xf32>
    %1154 = tosa.reshape %1151 {new_shape = array<i64: 1, 32, 40>} : (tensor<32x40x1xf32>) -> tensor<1x32x40xf32>
    %1155 = tosa.reshape %1118 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %1156 = tosa.matmul %1153, %1155 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %1157 = tosa.reshape %1156 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1158 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1159 = tosa.transpose %1157, %1158 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %1160 = tosa.reshape %1159 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %1161 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1162 = tosa.transpose %arg88, %1161 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1163 = tosa.reshape %1160 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_269 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1164 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1163, %1162 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_269 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1165 = tosa.reshape %1164 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1166 = tosa.add %1082, %1165 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1167 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_270 = arith.constant 2 : i32
    %1168 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1166 : tensor<1x40x4096xf32>) outs(%1167 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3745 = math.fpowi %in, %c2_i32_270 : f32, i32
      linalg.yield %3745 : f32
    } -> tensor<1x40x4096xf32>
    %1169 = tosa.reduce_sum %1168 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %1170 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1171 = tosa.reciprocal %1170 : (tensor<1xf32>) -> tensor<1xf32>
    %1172 = tosa.mul %1171, %1169 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1173 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %1174 = tosa.add %1172, %1173 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1175 = tosa.rsqrt %1174 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1176 = tosa.mul %1166, %1175 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %1177 = tosa.reshape %arg89 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %1178 = tosa.mul %1177, %1176 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1179 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1180 = tosa.transpose %arg90, %1179 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %1181 = tosa.reshape %1178 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_271 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %1182 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1181, %1180 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_271 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %1183 = tosa.reshape %1182 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %1184 = tosa.sigmoid %1183 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %1185 = tosa.mul %1183, %1184 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %1186 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1187 = tosa.transpose %arg91, %1186 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %1188 = tosa.reshape %1178 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_272 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %1189 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1188, %1187 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_272 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %1190 = tosa.reshape %1189 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %1191 = tosa.mul %1185, %1190 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %1192 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1193 = tosa.transpose %arg92, %1192 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %1194 = tosa.reshape %1191 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_273 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1195 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1194, %1193 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_273 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1196 = tosa.reshape %1195 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1197 = tosa.add %1166, %1196 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1198 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_274 = arith.constant 2 : i32
    %1199 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1197 : tensor<1x40x4096xf32>) outs(%1198 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3745 = math.fpowi %in, %c2_i32_274 : f32, i32
      linalg.yield %3745 : f32
    } -> tensor<1x40x4096xf32>
    %1200 = tosa.reduce_sum %1199 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %1201 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1202 = tosa.reciprocal %1201 : (tensor<1xf32>) -> tensor<1xf32>
    %1203 = tosa.mul %1202, %1200 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1204 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %1205 = tosa.add %1203, %1204 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1206 = tosa.rsqrt %1205 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1207 = tosa.mul %1197, %1206 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %1208 = tosa.reshape %arg93 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %1209 = tosa.mul %1208, %1207 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1210 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1211 = tosa.transpose %arg94, %1210 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1212 = tosa.reshape %1209 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_275 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1213 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1212, %1211 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_275 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1214 = tosa.reshape %1213 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1215 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1216 = tosa.transpose %arg95, %1215 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1217 = tosa.reshape %1209 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_276 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1218 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1217, %1216 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_276 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1219 = tosa.reshape %1218 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1220 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1221 = tosa.transpose %arg96, %1220 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1222 = tosa.reshape %1209 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_277 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1223 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1222, %1221 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_277 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1224 = tosa.reshape %1223 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1225 = tosa.reshape %1214 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %1226 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1227 = tosa.transpose %1225, %1226 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %1228 = tosa.reshape %1219 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %1229 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1230 = tosa.transpose %1228, %1229 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %1231 = tosa.reshape %1224 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %1232 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1233 = tosa.transpose %1231, %1232 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %1234 = tosa.reshape %45 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %1235 = tosa.reshape %47 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %1236 = tosa.mul %1227, %1234 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_278 = tensor.extract_slice %1227[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_279 = tensor.extract_slice %1227[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %1237 = tensor.empty() : tensor<1x32x40x64xf32>
    %1238 = linalg.negf ins(%extracted_slice_279 : tensor<1x32x40x64xf32>) outs(%1237 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %1239 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_280 = tensor.insert_slice %1238 into %1239[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_281 = tensor.insert_slice %extracted_slice_278 into %inserted_slice_280[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %1240 = tosa.mul %inserted_slice_281, %1235 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1241 = tosa.add %1236, %1240 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1242 = tosa.mul %1230, %1234 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_282 = tensor.extract_slice %1230[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_283 = tensor.extract_slice %1230[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %1243 = tensor.empty() : tensor<1x32x40x64xf32>
    %1244 = linalg.negf ins(%extracted_slice_283 : tensor<1x32x40x64xf32>) outs(%1243 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %1245 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_284 = tensor.insert_slice %1244 into %1245[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_285 = tensor.insert_slice %extracted_slice_282 into %inserted_slice_284[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %1246 = tosa.mul %inserted_slice_285, %1235 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1247 = tosa.add %1242, %1246 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1248 = tosa.reshape %19 {new_shape = array<i64: 1, 40, 41>} : (tensor<40x41xf32>) -> tensor<1x40x41xf32>
    %1249 = tosa.reshape %1248 {new_shape = array<i64: 1, 1, 40, 41>} : (tensor<1x40x41xf32>) -> tensor<1x1x40x41xf32>
    %extracted_slice_286 = tensor.extract_slice %1249[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_287 = tensor.extract_slice %extracted_slice_286[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %1250 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x1x40x41xf32>}> : () -> tensor<1x1x40x41xf32>
    %1251 = tosa.add %extracted_slice_287, %1250 : (tensor<1x1x40x41xf32>, tensor<1x1x40x41xf32>) -> tensor<1x1x40x41xf32>
    %extracted_slice_288 = tensor.extract_slice %1251[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_289 = tensor.extract_slice %extracted_slice_288[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_290 = tensor.extract_slice %extracted_slice_289[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_291 = tensor.extract_slice %extracted_slice_290[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x40xf32>
    %cst_292 = arith.constant 0.000000e+00 : f32
    %splat_293 = tensor.splat %cst_292 : tensor<40x40xf32>
    %1252 = tosa.reshape %extracted_slice_291 {new_shape = array<i64: 40, 40>} : (tensor<1x1x40x40xf32>) -> tensor<40x40xf32>
    %1253 = tosa.add %splat_293, %1252 : (tensor<40x40xf32>, tensor<40x40xf32>) -> tensor<40x40xf32>
    %1254 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1255 = tosa.transpose %1247, %1254 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %1256 = tosa.reshape %1241 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %1257 = tosa.reshape %1255 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %1258 = tosa.matmul %1256, %1257 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %cst_294 = arith.constant 0.0883883461 : f32
    %splat_295 = tensor.splat %cst_294 : tensor<32x40x40xf32>
    %1259 = tosa.mul %1258, %splat_295 {shift = 0 : i8} : (tensor<32x40x40xf32>, tensor<32x40x40xf32>) -> tensor<32x40x40xf32>
    %1260 = tosa.add %1259, %1253 : (tensor<32x40x40xf32>, tensor<40x40xf32>) -> tensor<32x40x40xf32>
    %1261 = tosa.reduce_max %1260 {axis = 2 : i32} : (tensor<32x40x40xf32>) -> tensor<32x40x1xf32>
    %1262 = tosa.sub %1260, %1261 : (tensor<32x40x40xf32>, tensor<32x40x1xf32>) -> tensor<32x40x40xf32>
    %1263 = math.exp %1262 : tensor<32x40x40xf32>
    %1264 = tosa.reduce_sum %1263 {axis = 2 : i32} : (tensor<32x40x40xf32>) -> tensor<32x40x1xf32>
    %1265 = tosa.log %1264 : (tensor<32x40x1xf32>) -> tensor<32x40x1xf32>
    %1266 = tosa.add %1261, %1265 : (tensor<32x40x1xf32>, tensor<32x40x1xf32>) -> tensor<32x40x1xf32>
    %1267 = tosa.sub %1260, %1266 : (tensor<32x40x40xf32>, tensor<32x40x1xf32>) -> tensor<32x40x40xf32>
    %1268 = math.exp %1267 : tensor<32x40x40xf32>
    %1269 = tosa.reshape %1266 {new_shape = array<i64: 1, 32, 40>} : (tensor<32x40x1xf32>) -> tensor<1x32x40xf32>
    %1270 = tosa.reshape %1233 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %1271 = tosa.matmul %1268, %1270 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %1272 = tosa.reshape %1271 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1273 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1274 = tosa.transpose %1272, %1273 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %1275 = tosa.reshape %1274 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %1276 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1277 = tosa.transpose %arg97, %1276 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1278 = tosa.reshape %1275 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_296 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1279 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1278, %1277 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_296 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1280 = tosa.reshape %1279 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1281 = tosa.add %1197, %1280 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1282 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_297 = arith.constant 2 : i32
    %1283 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1281 : tensor<1x40x4096xf32>) outs(%1282 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3745 = math.fpowi %in, %c2_i32_297 : f32, i32
      linalg.yield %3745 : f32
    } -> tensor<1x40x4096xf32>
    %1284 = tosa.reduce_sum %1283 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %1285 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1286 = tosa.reciprocal %1285 : (tensor<1xf32>) -> tensor<1xf32>
    %1287 = tosa.mul %1286, %1284 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1288 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %1289 = tosa.add %1287, %1288 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1290 = tosa.rsqrt %1289 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1291 = tosa.mul %1281, %1290 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %1292 = tosa.reshape %arg98 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %1293 = tosa.mul %1292, %1291 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1294 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1295 = tosa.transpose %arg99, %1294 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %1296 = tosa.reshape %1293 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_298 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %1297 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1296, %1295 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_298 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %1298 = tosa.reshape %1297 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %1299 = tosa.sigmoid %1298 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %1300 = tosa.mul %1298, %1299 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %1301 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1302 = tosa.transpose %arg100, %1301 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %1303 = tosa.reshape %1293 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_299 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %1304 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1303, %1302 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_299 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %1305 = tosa.reshape %1304 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %1306 = tosa.mul %1300, %1305 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %1307 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1308 = tosa.transpose %arg101, %1307 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %1309 = tosa.reshape %1306 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_300 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1310 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1309, %1308 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_300 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1311 = tosa.reshape %1310 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1312 = tosa.add %1281, %1311 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1313 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_301 = arith.constant 2 : i32
    %1314 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1312 : tensor<1x40x4096xf32>) outs(%1313 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3745 = math.fpowi %in, %c2_i32_301 : f32, i32
      linalg.yield %3745 : f32
    } -> tensor<1x40x4096xf32>
    %1315 = tosa.reduce_sum %1314 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %1316 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1317 = tosa.reciprocal %1316 : (tensor<1xf32>) -> tensor<1xf32>
    %1318 = tosa.mul %1317, %1315 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1319 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %1320 = tosa.add %1318, %1319 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1321 = tosa.rsqrt %1320 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1322 = tosa.mul %1312, %1321 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %1323 = tosa.reshape %arg102 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %1324 = tosa.mul %1323, %1322 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1325 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1326 = tosa.transpose %arg103, %1325 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1327 = tosa.reshape %1324 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_302 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1328 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1327, %1326 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_302 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1329 = tosa.reshape %1328 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1330 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1331 = tosa.transpose %arg104, %1330 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1332 = tosa.reshape %1324 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_303 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1333 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1332, %1331 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_303 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1334 = tosa.reshape %1333 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1335 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1336 = tosa.transpose %arg105, %1335 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1337 = tosa.reshape %1324 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_304 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1338 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1337, %1336 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_304 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1339 = tosa.reshape %1338 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1340 = tosa.reshape %1329 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %1341 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1342 = tosa.transpose %1340, %1341 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %1343 = tosa.reshape %1334 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %1344 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1345 = tosa.transpose %1343, %1344 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %1346 = tosa.reshape %1339 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %1347 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1348 = tosa.transpose %1346, %1347 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %1349 = tosa.reshape %45 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %1350 = tosa.reshape %47 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %1351 = tosa.mul %1342, %1349 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_305 = tensor.extract_slice %1342[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_306 = tensor.extract_slice %1342[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %1352 = tensor.empty() : tensor<1x32x40x64xf32>
    %1353 = linalg.negf ins(%extracted_slice_306 : tensor<1x32x40x64xf32>) outs(%1352 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %1354 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_307 = tensor.insert_slice %1353 into %1354[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_308 = tensor.insert_slice %extracted_slice_305 into %inserted_slice_307[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %1355 = tosa.mul %inserted_slice_308, %1350 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1356 = tosa.add %1351, %1355 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1357 = tosa.mul %1345, %1349 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_309 = tensor.extract_slice %1345[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_310 = tensor.extract_slice %1345[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %1358 = tensor.empty() : tensor<1x32x40x64xf32>
    %1359 = linalg.negf ins(%extracted_slice_310 : tensor<1x32x40x64xf32>) outs(%1358 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %1360 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_311 = tensor.insert_slice %1359 into %1360[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_312 = tensor.insert_slice %extracted_slice_309 into %inserted_slice_311[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %1361 = tosa.mul %inserted_slice_312, %1350 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1362 = tosa.add %1357, %1361 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1363 = tosa.reshape %19 {new_shape = array<i64: 1, 40, 41>} : (tensor<40x41xf32>) -> tensor<1x40x41xf32>
    %1364 = tosa.reshape %1363 {new_shape = array<i64: 1, 1, 40, 41>} : (tensor<1x40x41xf32>) -> tensor<1x1x40x41xf32>
    %extracted_slice_313 = tensor.extract_slice %1364[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_314 = tensor.extract_slice %extracted_slice_313[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %1365 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x1x40x41xf32>}> : () -> tensor<1x1x40x41xf32>
    %1366 = tosa.add %extracted_slice_314, %1365 : (tensor<1x1x40x41xf32>, tensor<1x1x40x41xf32>) -> tensor<1x1x40x41xf32>
    %extracted_slice_315 = tensor.extract_slice %1366[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_316 = tensor.extract_slice %extracted_slice_315[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_317 = tensor.extract_slice %extracted_slice_316[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_318 = tensor.extract_slice %extracted_slice_317[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x40xf32>
    %cst_319 = arith.constant 0.000000e+00 : f32
    %splat_320 = tensor.splat %cst_319 : tensor<40x40xf32>
    %1367 = tosa.reshape %extracted_slice_318 {new_shape = array<i64: 40, 40>} : (tensor<1x1x40x40xf32>) -> tensor<40x40xf32>
    %1368 = tosa.add %splat_320, %1367 : (tensor<40x40xf32>, tensor<40x40xf32>) -> tensor<40x40xf32>
    %1369 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1370 = tosa.transpose %1362, %1369 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %1371 = tosa.reshape %1356 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %1372 = tosa.reshape %1370 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %1373 = tosa.matmul %1371, %1372 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %cst_321 = arith.constant 0.0883883461 : f32
    %splat_322 = tensor.splat %cst_321 : tensor<32x40x40xf32>
    %1374 = tosa.mul %1373, %splat_322 {shift = 0 : i8} : (tensor<32x40x40xf32>, tensor<32x40x40xf32>) -> tensor<32x40x40xf32>
    %1375 = tosa.add %1374, %1368 : (tensor<32x40x40xf32>, tensor<40x40xf32>) -> tensor<32x40x40xf32>
    %1376 = tosa.reduce_max %1375 {axis = 2 : i32} : (tensor<32x40x40xf32>) -> tensor<32x40x1xf32>
    %1377 = tosa.sub %1375, %1376 : (tensor<32x40x40xf32>, tensor<32x40x1xf32>) -> tensor<32x40x40xf32>
    %1378 = math.exp %1377 : tensor<32x40x40xf32>
    %1379 = tosa.reduce_sum %1378 {axis = 2 : i32} : (tensor<32x40x40xf32>) -> tensor<32x40x1xf32>
    %1380 = tosa.log %1379 : (tensor<32x40x1xf32>) -> tensor<32x40x1xf32>
    %1381 = tosa.add %1376, %1380 : (tensor<32x40x1xf32>, tensor<32x40x1xf32>) -> tensor<32x40x1xf32>
    %1382 = tosa.sub %1375, %1381 : (tensor<32x40x40xf32>, tensor<32x40x1xf32>) -> tensor<32x40x40xf32>
    %1383 = math.exp %1382 : tensor<32x40x40xf32>
    %1384 = tosa.reshape %1381 {new_shape = array<i64: 1, 32, 40>} : (tensor<32x40x1xf32>) -> tensor<1x32x40xf32>
    %1385 = tosa.reshape %1348 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %1386 = tosa.matmul %1383, %1385 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %1387 = tosa.reshape %1386 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1388 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1389 = tosa.transpose %1387, %1388 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %1390 = tosa.reshape %1389 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %1391 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1392 = tosa.transpose %arg106, %1391 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1393 = tosa.reshape %1390 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_323 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1394 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1393, %1392 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_323 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1395 = tosa.reshape %1394 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1396 = tosa.add %1312, %1395 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1397 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_324 = arith.constant 2 : i32
    %1398 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1396 : tensor<1x40x4096xf32>) outs(%1397 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3745 = math.fpowi %in, %c2_i32_324 : f32, i32
      linalg.yield %3745 : f32
    } -> tensor<1x40x4096xf32>
    %1399 = tosa.reduce_sum %1398 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %1400 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1401 = tosa.reciprocal %1400 : (tensor<1xf32>) -> tensor<1xf32>
    %1402 = tosa.mul %1401, %1399 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1403 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %1404 = tosa.add %1402, %1403 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1405 = tosa.rsqrt %1404 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1406 = tosa.mul %1396, %1405 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %1407 = tosa.reshape %arg107 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %1408 = tosa.mul %1407, %1406 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1409 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1410 = tosa.transpose %arg108, %1409 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %1411 = tosa.reshape %1408 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_325 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %1412 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1411, %1410 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_325 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %1413 = tosa.reshape %1412 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %1414 = tosa.sigmoid %1413 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %1415 = tosa.mul %1413, %1414 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %1416 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1417 = tosa.transpose %arg109, %1416 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %1418 = tosa.reshape %1408 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_326 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %1419 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1418, %1417 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_326 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %1420 = tosa.reshape %1419 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %1421 = tosa.mul %1415, %1420 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %1422 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1423 = tosa.transpose %arg110, %1422 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %1424 = tosa.reshape %1421 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_327 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1425 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1424, %1423 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_327 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1426 = tosa.reshape %1425 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1427 = tosa.add %1396, %1426 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1428 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_328 = arith.constant 2 : i32
    %1429 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1427 : tensor<1x40x4096xf32>) outs(%1428 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3745 = math.fpowi %in, %c2_i32_328 : f32, i32
      linalg.yield %3745 : f32
    } -> tensor<1x40x4096xf32>
    %1430 = tosa.reduce_sum %1429 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %1431 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1432 = tosa.reciprocal %1431 : (tensor<1xf32>) -> tensor<1xf32>
    %1433 = tosa.mul %1432, %1430 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1434 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %1435 = tosa.add %1433, %1434 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1436 = tosa.rsqrt %1435 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1437 = tosa.mul %1427, %1436 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %1438 = tosa.reshape %arg111 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %1439 = tosa.mul %1438, %1437 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1440 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1441 = tosa.transpose %arg112, %1440 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1442 = tosa.reshape %1439 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_329 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1443 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1442, %1441 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_329 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1444 = tosa.reshape %1443 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1445 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1446 = tosa.transpose %arg113, %1445 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1447 = tosa.reshape %1439 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_330 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1448 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1447, %1446 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_330 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1449 = tosa.reshape %1448 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1450 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1451 = tosa.transpose %arg114, %1450 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1452 = tosa.reshape %1439 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_331 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1453 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1452, %1451 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_331 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1454 = tosa.reshape %1453 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1455 = tosa.reshape %1444 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %1456 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1457 = tosa.transpose %1455, %1456 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %1458 = tosa.reshape %1449 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %1459 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1460 = tosa.transpose %1458, %1459 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %1461 = tosa.reshape %1454 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %1462 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1463 = tosa.transpose %1461, %1462 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %1464 = tosa.reshape %45 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %1465 = tosa.reshape %47 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %1466 = tosa.mul %1457, %1464 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_332 = tensor.extract_slice %1457[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_333 = tensor.extract_slice %1457[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %1467 = tensor.empty() : tensor<1x32x40x64xf32>
    %1468 = linalg.negf ins(%extracted_slice_333 : tensor<1x32x40x64xf32>) outs(%1467 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %1469 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_334 = tensor.insert_slice %1468 into %1469[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_335 = tensor.insert_slice %extracted_slice_332 into %inserted_slice_334[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %1470 = tosa.mul %inserted_slice_335, %1465 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1471 = tosa.add %1466, %1470 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1472 = tosa.mul %1460, %1464 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_336 = tensor.extract_slice %1460[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_337 = tensor.extract_slice %1460[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %1473 = tensor.empty() : tensor<1x32x40x64xf32>
    %1474 = linalg.negf ins(%extracted_slice_337 : tensor<1x32x40x64xf32>) outs(%1473 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %1475 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_338 = tensor.insert_slice %1474 into %1475[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_339 = tensor.insert_slice %extracted_slice_336 into %inserted_slice_338[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %1476 = tosa.mul %inserted_slice_339, %1465 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1477 = tosa.add %1472, %1476 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1478 = tosa.reshape %19 {new_shape = array<i64: 1, 40, 41>} : (tensor<40x41xf32>) -> tensor<1x40x41xf32>
    %1479 = tosa.reshape %1478 {new_shape = array<i64: 1, 1, 40, 41>} : (tensor<1x40x41xf32>) -> tensor<1x1x40x41xf32>
    %extracted_slice_340 = tensor.extract_slice %1479[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_341 = tensor.extract_slice %extracted_slice_340[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %1480 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x1x40x41xf32>}> : () -> tensor<1x1x40x41xf32>
    %1481 = tosa.add %extracted_slice_341, %1480 : (tensor<1x1x40x41xf32>, tensor<1x1x40x41xf32>) -> tensor<1x1x40x41xf32>
    %extracted_slice_342 = tensor.extract_slice %1481[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_343 = tensor.extract_slice %extracted_slice_342[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_344 = tensor.extract_slice %extracted_slice_343[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_345 = tensor.extract_slice %extracted_slice_344[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x40xf32>
    %cst_346 = arith.constant 0.000000e+00 : f32
    %splat_347 = tensor.splat %cst_346 : tensor<40x40xf32>
    %1482 = tosa.reshape %extracted_slice_345 {new_shape = array<i64: 40, 40>} : (tensor<1x1x40x40xf32>) -> tensor<40x40xf32>
    %1483 = tosa.add %splat_347, %1482 : (tensor<40x40xf32>, tensor<40x40xf32>) -> tensor<40x40xf32>
    %1484 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1485 = tosa.transpose %1477, %1484 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %1486 = tosa.reshape %1471 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %1487 = tosa.reshape %1485 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %1488 = tosa.matmul %1486, %1487 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %cst_348 = arith.constant 0.0883883461 : f32
    %splat_349 = tensor.splat %cst_348 : tensor<32x40x40xf32>
    %1489 = tosa.mul %1488, %splat_349 {shift = 0 : i8} : (tensor<32x40x40xf32>, tensor<32x40x40xf32>) -> tensor<32x40x40xf32>
    %1490 = tosa.add %1489, %1483 : (tensor<32x40x40xf32>, tensor<40x40xf32>) -> tensor<32x40x40xf32>
    %1491 = tosa.reduce_max %1490 {axis = 2 : i32} : (tensor<32x40x40xf32>) -> tensor<32x40x1xf32>
    %1492 = tosa.sub %1490, %1491 : (tensor<32x40x40xf32>, tensor<32x40x1xf32>) -> tensor<32x40x40xf32>
    %1493 = math.exp %1492 : tensor<32x40x40xf32>
    %1494 = tosa.reduce_sum %1493 {axis = 2 : i32} : (tensor<32x40x40xf32>) -> tensor<32x40x1xf32>
    %1495 = tosa.log %1494 : (tensor<32x40x1xf32>) -> tensor<32x40x1xf32>
    %1496 = tosa.add %1491, %1495 : (tensor<32x40x1xf32>, tensor<32x40x1xf32>) -> tensor<32x40x1xf32>
    %1497 = tosa.sub %1490, %1496 : (tensor<32x40x40xf32>, tensor<32x40x1xf32>) -> tensor<32x40x40xf32>
    %1498 = math.exp %1497 : tensor<32x40x40xf32>
    %1499 = tosa.reshape %1496 {new_shape = array<i64: 1, 32, 40>} : (tensor<32x40x1xf32>) -> tensor<1x32x40xf32>
    %1500 = tosa.reshape %1463 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %1501 = tosa.matmul %1498, %1500 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %1502 = tosa.reshape %1501 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1503 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1504 = tosa.transpose %1502, %1503 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %1505 = tosa.reshape %1504 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %1506 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1507 = tosa.transpose %arg115, %1506 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1508 = tosa.reshape %1505 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_350 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1509 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1508, %1507 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_350 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1510 = tosa.reshape %1509 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1511 = tosa.add %1427, %1510 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1512 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_351 = arith.constant 2 : i32
    %1513 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1511 : tensor<1x40x4096xf32>) outs(%1512 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3745 = math.fpowi %in, %c2_i32_351 : f32, i32
      linalg.yield %3745 : f32
    } -> tensor<1x40x4096xf32>
    %1514 = tosa.reduce_sum %1513 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %1515 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1516 = tosa.reciprocal %1515 : (tensor<1xf32>) -> tensor<1xf32>
    %1517 = tosa.mul %1516, %1514 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1518 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %1519 = tosa.add %1517, %1518 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1520 = tosa.rsqrt %1519 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1521 = tosa.mul %1511, %1520 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %1522 = tosa.reshape %arg116 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %1523 = tosa.mul %1522, %1521 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1524 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1525 = tosa.transpose %arg117, %1524 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %1526 = tosa.reshape %1523 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_352 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %1527 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1526, %1525 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_352 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %1528 = tosa.reshape %1527 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %1529 = tosa.sigmoid %1528 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %1530 = tosa.mul %1528, %1529 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %1531 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1532 = tosa.transpose %arg118, %1531 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %1533 = tosa.reshape %1523 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_353 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %1534 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1533, %1532 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_353 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %1535 = tosa.reshape %1534 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %1536 = tosa.mul %1530, %1535 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %1537 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1538 = tosa.transpose %arg119, %1537 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %1539 = tosa.reshape %1536 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_354 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1540 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1539, %1538 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_354 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1541 = tosa.reshape %1540 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1542 = tosa.add %1511, %1541 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1543 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_355 = arith.constant 2 : i32
    %1544 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1542 : tensor<1x40x4096xf32>) outs(%1543 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3745 = math.fpowi %in, %c2_i32_355 : f32, i32
      linalg.yield %3745 : f32
    } -> tensor<1x40x4096xf32>
    %1545 = tosa.reduce_sum %1544 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %1546 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1547 = tosa.reciprocal %1546 : (tensor<1xf32>) -> tensor<1xf32>
    %1548 = tosa.mul %1547, %1545 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1549 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %1550 = tosa.add %1548, %1549 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1551 = tosa.rsqrt %1550 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1552 = tosa.mul %1542, %1551 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %1553 = tosa.reshape %arg120 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %1554 = tosa.mul %1553, %1552 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1555 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1556 = tosa.transpose %arg121, %1555 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1557 = tosa.reshape %1554 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_356 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1558 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1557, %1556 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_356 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1559 = tosa.reshape %1558 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1560 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1561 = tosa.transpose %arg122, %1560 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1562 = tosa.reshape %1554 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_357 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1563 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1562, %1561 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_357 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1564 = tosa.reshape %1563 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1565 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1566 = tosa.transpose %arg123, %1565 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1567 = tosa.reshape %1554 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_358 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1568 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1567, %1566 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_358 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1569 = tosa.reshape %1568 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1570 = tosa.reshape %1559 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %1571 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1572 = tosa.transpose %1570, %1571 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %1573 = tosa.reshape %1564 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %1574 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1575 = tosa.transpose %1573, %1574 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %1576 = tosa.reshape %1569 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %1577 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1578 = tosa.transpose %1576, %1577 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %1579 = tosa.reshape %45 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %1580 = tosa.reshape %47 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %1581 = tosa.mul %1572, %1579 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_359 = tensor.extract_slice %1572[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_360 = tensor.extract_slice %1572[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %1582 = tensor.empty() : tensor<1x32x40x64xf32>
    %1583 = linalg.negf ins(%extracted_slice_360 : tensor<1x32x40x64xf32>) outs(%1582 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %1584 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_361 = tensor.insert_slice %1583 into %1584[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_362 = tensor.insert_slice %extracted_slice_359 into %inserted_slice_361[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %1585 = tosa.mul %inserted_slice_362, %1580 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1586 = tosa.add %1581, %1585 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1587 = tosa.mul %1575, %1579 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_363 = tensor.extract_slice %1575[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_364 = tensor.extract_slice %1575[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %1588 = tensor.empty() : tensor<1x32x40x64xf32>
    %1589 = linalg.negf ins(%extracted_slice_364 : tensor<1x32x40x64xf32>) outs(%1588 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %1590 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_365 = tensor.insert_slice %1589 into %1590[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_366 = tensor.insert_slice %extracted_slice_363 into %inserted_slice_365[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %1591 = tosa.mul %inserted_slice_366, %1580 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1592 = tosa.add %1587, %1591 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1593 = tosa.reshape %19 {new_shape = array<i64: 1, 40, 41>} : (tensor<40x41xf32>) -> tensor<1x40x41xf32>
    %1594 = tosa.reshape %1593 {new_shape = array<i64: 1, 1, 40, 41>} : (tensor<1x40x41xf32>) -> tensor<1x1x40x41xf32>
    %extracted_slice_367 = tensor.extract_slice %1594[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_368 = tensor.extract_slice %extracted_slice_367[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %1595 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x1x40x41xf32>}> : () -> tensor<1x1x40x41xf32>
    %1596 = tosa.add %extracted_slice_368, %1595 : (tensor<1x1x40x41xf32>, tensor<1x1x40x41xf32>) -> tensor<1x1x40x41xf32>
    %extracted_slice_369 = tensor.extract_slice %1596[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_370 = tensor.extract_slice %extracted_slice_369[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_371 = tensor.extract_slice %extracted_slice_370[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_372 = tensor.extract_slice %extracted_slice_371[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x40xf32>
    %cst_373 = arith.constant 0.000000e+00 : f32
    %splat_374 = tensor.splat %cst_373 : tensor<40x40xf32>
    %1597 = tosa.reshape %extracted_slice_372 {new_shape = array<i64: 40, 40>} : (tensor<1x1x40x40xf32>) -> tensor<40x40xf32>
    %1598 = tosa.add %splat_374, %1597 : (tensor<40x40xf32>, tensor<40x40xf32>) -> tensor<40x40xf32>
    %1599 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1600 = tosa.transpose %1592, %1599 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %1601 = tosa.reshape %1586 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %1602 = tosa.reshape %1600 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %1603 = tosa.matmul %1601, %1602 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %cst_375 = arith.constant 0.0883883461 : f32
    %splat_376 = tensor.splat %cst_375 : tensor<32x40x40xf32>
    %1604 = tosa.mul %1603, %splat_376 {shift = 0 : i8} : (tensor<32x40x40xf32>, tensor<32x40x40xf32>) -> tensor<32x40x40xf32>
    %1605 = tosa.add %1604, %1598 : (tensor<32x40x40xf32>, tensor<40x40xf32>) -> tensor<32x40x40xf32>
    %1606 = tosa.reduce_max %1605 {axis = 2 : i32} : (tensor<32x40x40xf32>) -> tensor<32x40x1xf32>
    %1607 = tosa.sub %1605, %1606 : (tensor<32x40x40xf32>, tensor<32x40x1xf32>) -> tensor<32x40x40xf32>
    %1608 = math.exp %1607 : tensor<32x40x40xf32>
    %1609 = tosa.reduce_sum %1608 {axis = 2 : i32} : (tensor<32x40x40xf32>) -> tensor<32x40x1xf32>
    %1610 = tosa.log %1609 : (tensor<32x40x1xf32>) -> tensor<32x40x1xf32>
    %1611 = tosa.add %1606, %1610 : (tensor<32x40x1xf32>, tensor<32x40x1xf32>) -> tensor<32x40x1xf32>
    %1612 = tosa.sub %1605, %1611 : (tensor<32x40x40xf32>, tensor<32x40x1xf32>) -> tensor<32x40x40xf32>
    %1613 = math.exp %1612 : tensor<32x40x40xf32>
    %1614 = tosa.reshape %1611 {new_shape = array<i64: 1, 32, 40>} : (tensor<32x40x1xf32>) -> tensor<1x32x40xf32>
    %1615 = tosa.reshape %1578 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %1616 = tosa.matmul %1613, %1615 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %1617 = tosa.reshape %1616 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1618 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1619 = tosa.transpose %1617, %1618 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %1620 = tosa.reshape %1619 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %1621 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1622 = tosa.transpose %arg124, %1621 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1623 = tosa.reshape %1620 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_377 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1624 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1623, %1622 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_377 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1625 = tosa.reshape %1624 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1626 = tosa.add %1542, %1625 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1627 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_378 = arith.constant 2 : i32
    %1628 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1626 : tensor<1x40x4096xf32>) outs(%1627 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3745 = math.fpowi %in, %c2_i32_378 : f32, i32
      linalg.yield %3745 : f32
    } -> tensor<1x40x4096xf32>
    %1629 = tosa.reduce_sum %1628 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %1630 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1631 = tosa.reciprocal %1630 : (tensor<1xf32>) -> tensor<1xf32>
    %1632 = tosa.mul %1631, %1629 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1633 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %1634 = tosa.add %1632, %1633 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1635 = tosa.rsqrt %1634 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1636 = tosa.mul %1626, %1635 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %1637 = tosa.reshape %arg125 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %1638 = tosa.mul %1637, %1636 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1639 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1640 = tosa.transpose %arg126, %1639 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %1641 = tosa.reshape %1638 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_379 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %1642 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1641, %1640 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_379 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %1643 = tosa.reshape %1642 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %1644 = tosa.sigmoid %1643 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %1645 = tosa.mul %1643, %1644 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %1646 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1647 = tosa.transpose %arg127, %1646 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %1648 = tosa.reshape %1638 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_380 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %1649 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1648, %1647 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_380 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %1650 = tosa.reshape %1649 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %1651 = tosa.mul %1645, %1650 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %1652 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1653 = tosa.transpose %arg128, %1652 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %1654 = tosa.reshape %1651 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_381 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1655 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1654, %1653 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_381 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1656 = tosa.reshape %1655 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1657 = tosa.add %1626, %1656 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1658 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_382 = arith.constant 2 : i32
    %1659 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1657 : tensor<1x40x4096xf32>) outs(%1658 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3745 = math.fpowi %in, %c2_i32_382 : f32, i32
      linalg.yield %3745 : f32
    } -> tensor<1x40x4096xf32>
    %1660 = tosa.reduce_sum %1659 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %1661 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1662 = tosa.reciprocal %1661 : (tensor<1xf32>) -> tensor<1xf32>
    %1663 = tosa.mul %1662, %1660 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1664 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %1665 = tosa.add %1663, %1664 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1666 = tosa.rsqrt %1665 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1667 = tosa.mul %1657, %1666 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %1668 = tosa.reshape %arg129 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %1669 = tosa.mul %1668, %1667 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1670 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1671 = tosa.transpose %arg130, %1670 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1672 = tosa.reshape %1669 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_383 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1673 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1672, %1671 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_383 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1674 = tosa.reshape %1673 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1675 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1676 = tosa.transpose %arg131, %1675 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1677 = tosa.reshape %1669 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_384 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1678 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1677, %1676 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_384 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1679 = tosa.reshape %1678 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1680 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1681 = tosa.transpose %arg132, %1680 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1682 = tosa.reshape %1669 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_385 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1683 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1682, %1681 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_385 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1684 = tosa.reshape %1683 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1685 = tosa.reshape %1674 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %1686 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1687 = tosa.transpose %1685, %1686 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %1688 = tosa.reshape %1679 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %1689 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1690 = tosa.transpose %1688, %1689 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %1691 = tosa.reshape %1684 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %1692 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1693 = tosa.transpose %1691, %1692 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %1694 = tosa.reshape %45 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %1695 = tosa.reshape %47 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %1696 = tosa.mul %1687, %1694 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_386 = tensor.extract_slice %1687[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_387 = tensor.extract_slice %1687[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %1697 = tensor.empty() : tensor<1x32x40x64xf32>
    %1698 = linalg.negf ins(%extracted_slice_387 : tensor<1x32x40x64xf32>) outs(%1697 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %1699 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_388 = tensor.insert_slice %1698 into %1699[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_389 = tensor.insert_slice %extracted_slice_386 into %inserted_slice_388[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %1700 = tosa.mul %inserted_slice_389, %1695 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1701 = tosa.add %1696, %1700 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1702 = tosa.mul %1690, %1694 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_390 = tensor.extract_slice %1690[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_391 = tensor.extract_slice %1690[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %1703 = tensor.empty() : tensor<1x32x40x64xf32>
    %1704 = linalg.negf ins(%extracted_slice_391 : tensor<1x32x40x64xf32>) outs(%1703 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %1705 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_392 = tensor.insert_slice %1704 into %1705[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_393 = tensor.insert_slice %extracted_slice_390 into %inserted_slice_392[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %1706 = tosa.mul %inserted_slice_393, %1695 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1707 = tosa.add %1702, %1706 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1708 = tosa.reshape %19 {new_shape = array<i64: 1, 40, 41>} : (tensor<40x41xf32>) -> tensor<1x40x41xf32>
    %1709 = tosa.reshape %1708 {new_shape = array<i64: 1, 1, 40, 41>} : (tensor<1x40x41xf32>) -> tensor<1x1x40x41xf32>
    %extracted_slice_394 = tensor.extract_slice %1709[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_395 = tensor.extract_slice %extracted_slice_394[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %1710 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x1x40x41xf32>}> : () -> tensor<1x1x40x41xf32>
    %1711 = tosa.add %extracted_slice_395, %1710 : (tensor<1x1x40x41xf32>, tensor<1x1x40x41xf32>) -> tensor<1x1x40x41xf32>
    %extracted_slice_396 = tensor.extract_slice %1711[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_397 = tensor.extract_slice %extracted_slice_396[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_398 = tensor.extract_slice %extracted_slice_397[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_399 = tensor.extract_slice %extracted_slice_398[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x40xf32>
    %cst_400 = arith.constant 0.000000e+00 : f32
    %splat_401 = tensor.splat %cst_400 : tensor<40x40xf32>
    %1712 = tosa.reshape %extracted_slice_399 {new_shape = array<i64: 40, 40>} : (tensor<1x1x40x40xf32>) -> tensor<40x40xf32>
    %1713 = tosa.add %splat_401, %1712 : (tensor<40x40xf32>, tensor<40x40xf32>) -> tensor<40x40xf32>
    %1714 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1715 = tosa.transpose %1707, %1714 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %1716 = tosa.reshape %1701 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %1717 = tosa.reshape %1715 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %1718 = tosa.matmul %1716, %1717 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %cst_402 = arith.constant 0.0883883461 : f32
    %splat_403 = tensor.splat %cst_402 : tensor<32x40x40xf32>
    %1719 = tosa.mul %1718, %splat_403 {shift = 0 : i8} : (tensor<32x40x40xf32>, tensor<32x40x40xf32>) -> tensor<32x40x40xf32>
    %1720 = tosa.add %1719, %1713 : (tensor<32x40x40xf32>, tensor<40x40xf32>) -> tensor<32x40x40xf32>
    %1721 = tosa.reduce_max %1720 {axis = 2 : i32} : (tensor<32x40x40xf32>) -> tensor<32x40x1xf32>
    %1722 = tosa.sub %1720, %1721 : (tensor<32x40x40xf32>, tensor<32x40x1xf32>) -> tensor<32x40x40xf32>
    %1723 = math.exp %1722 : tensor<32x40x40xf32>
    %1724 = tosa.reduce_sum %1723 {axis = 2 : i32} : (tensor<32x40x40xf32>) -> tensor<32x40x1xf32>
    %1725 = tosa.log %1724 : (tensor<32x40x1xf32>) -> tensor<32x40x1xf32>
    %1726 = tosa.add %1721, %1725 : (tensor<32x40x1xf32>, tensor<32x40x1xf32>) -> tensor<32x40x1xf32>
    %1727 = tosa.sub %1720, %1726 : (tensor<32x40x40xf32>, tensor<32x40x1xf32>) -> tensor<32x40x40xf32>
    %1728 = math.exp %1727 : tensor<32x40x40xf32>
    %1729 = tosa.reshape %1726 {new_shape = array<i64: 1, 32, 40>} : (tensor<32x40x1xf32>) -> tensor<1x32x40xf32>
    %1730 = tosa.reshape %1693 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %1731 = tosa.matmul %1728, %1730 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %1732 = tosa.reshape %1731 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1733 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1734 = tosa.transpose %1732, %1733 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %1735 = tosa.reshape %1734 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %1736 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1737 = tosa.transpose %arg133, %1736 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1738 = tosa.reshape %1735 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_404 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1739 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1738, %1737 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_404 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1740 = tosa.reshape %1739 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1741 = tosa.add %1657, %1740 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1742 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_405 = arith.constant 2 : i32
    %1743 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1741 : tensor<1x40x4096xf32>) outs(%1742 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3745 = math.fpowi %in, %c2_i32_405 : f32, i32
      linalg.yield %3745 : f32
    } -> tensor<1x40x4096xf32>
    %1744 = tosa.reduce_sum %1743 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %1745 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1746 = tosa.reciprocal %1745 : (tensor<1xf32>) -> tensor<1xf32>
    %1747 = tosa.mul %1746, %1744 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1748 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %1749 = tosa.add %1747, %1748 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1750 = tosa.rsqrt %1749 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1751 = tosa.mul %1741, %1750 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %1752 = tosa.reshape %arg134 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %1753 = tosa.mul %1752, %1751 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1754 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1755 = tosa.transpose %arg135, %1754 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %1756 = tosa.reshape %1753 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_406 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %1757 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1756, %1755 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_406 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %1758 = tosa.reshape %1757 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %1759 = tosa.sigmoid %1758 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %1760 = tosa.mul %1758, %1759 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %1761 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1762 = tosa.transpose %arg136, %1761 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %1763 = tosa.reshape %1753 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_407 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %1764 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1763, %1762 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_407 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %1765 = tosa.reshape %1764 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %1766 = tosa.mul %1760, %1765 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %1767 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1768 = tosa.transpose %arg137, %1767 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %1769 = tosa.reshape %1766 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_408 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1770 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1769, %1768 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_408 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1771 = tosa.reshape %1770 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1772 = tosa.add %1741, %1771 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1773 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_409 = arith.constant 2 : i32
    %1774 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1772 : tensor<1x40x4096xf32>) outs(%1773 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3745 = math.fpowi %in, %c2_i32_409 : f32, i32
      linalg.yield %3745 : f32
    } -> tensor<1x40x4096xf32>
    %1775 = tosa.reduce_sum %1774 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %1776 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1777 = tosa.reciprocal %1776 : (tensor<1xf32>) -> tensor<1xf32>
    %1778 = tosa.mul %1777, %1775 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1779 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %1780 = tosa.add %1778, %1779 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1781 = tosa.rsqrt %1780 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1782 = tosa.mul %1772, %1781 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %1783 = tosa.reshape %arg138 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %1784 = tosa.mul %1783, %1782 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1785 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1786 = tosa.transpose %arg139, %1785 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1787 = tosa.reshape %1784 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_410 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1788 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1787, %1786 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_410 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1789 = tosa.reshape %1788 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1790 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1791 = tosa.transpose %arg140, %1790 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1792 = tosa.reshape %1784 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_411 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1793 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1792, %1791 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_411 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1794 = tosa.reshape %1793 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1795 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1796 = tosa.transpose %arg141, %1795 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1797 = tosa.reshape %1784 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_412 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1798 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1797, %1796 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_412 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1799 = tosa.reshape %1798 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1800 = tosa.reshape %1789 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %1801 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1802 = tosa.transpose %1800, %1801 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %1803 = tosa.reshape %1794 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %1804 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1805 = tosa.transpose %1803, %1804 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %1806 = tosa.reshape %1799 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %1807 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1808 = tosa.transpose %1806, %1807 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %1809 = tosa.reshape %45 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %1810 = tosa.reshape %47 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %1811 = tosa.mul %1802, %1809 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_413 = tensor.extract_slice %1802[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_414 = tensor.extract_slice %1802[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %1812 = tensor.empty() : tensor<1x32x40x64xf32>
    %1813 = linalg.negf ins(%extracted_slice_414 : tensor<1x32x40x64xf32>) outs(%1812 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %1814 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_415 = tensor.insert_slice %1813 into %1814[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_416 = tensor.insert_slice %extracted_slice_413 into %inserted_slice_415[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %1815 = tosa.mul %inserted_slice_416, %1810 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1816 = tosa.add %1811, %1815 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1817 = tosa.mul %1805, %1809 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_417 = tensor.extract_slice %1805[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_418 = tensor.extract_slice %1805[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %1818 = tensor.empty() : tensor<1x32x40x64xf32>
    %1819 = linalg.negf ins(%extracted_slice_418 : tensor<1x32x40x64xf32>) outs(%1818 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %1820 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_419 = tensor.insert_slice %1819 into %1820[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_420 = tensor.insert_slice %extracted_slice_417 into %inserted_slice_419[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %1821 = tosa.mul %inserted_slice_420, %1810 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1822 = tosa.add %1817, %1821 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1823 = tosa.reshape %19 {new_shape = array<i64: 1, 40, 41>} : (tensor<40x41xf32>) -> tensor<1x40x41xf32>
    %1824 = tosa.reshape %1823 {new_shape = array<i64: 1, 1, 40, 41>} : (tensor<1x40x41xf32>) -> tensor<1x1x40x41xf32>
    %extracted_slice_421 = tensor.extract_slice %1824[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_422 = tensor.extract_slice %extracted_slice_421[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %1825 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x1x40x41xf32>}> : () -> tensor<1x1x40x41xf32>
    %1826 = tosa.add %extracted_slice_422, %1825 : (tensor<1x1x40x41xf32>, tensor<1x1x40x41xf32>) -> tensor<1x1x40x41xf32>
    %extracted_slice_423 = tensor.extract_slice %1826[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_424 = tensor.extract_slice %extracted_slice_423[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_425 = tensor.extract_slice %extracted_slice_424[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_426 = tensor.extract_slice %extracted_slice_425[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x40xf32>
    %cst_427 = arith.constant 0.000000e+00 : f32
    %splat_428 = tensor.splat %cst_427 : tensor<40x40xf32>
    %1827 = tosa.reshape %extracted_slice_426 {new_shape = array<i64: 40, 40>} : (tensor<1x1x40x40xf32>) -> tensor<40x40xf32>
    %1828 = tosa.add %splat_428, %1827 : (tensor<40x40xf32>, tensor<40x40xf32>) -> tensor<40x40xf32>
    %1829 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1830 = tosa.transpose %1822, %1829 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %1831 = tosa.reshape %1816 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %1832 = tosa.reshape %1830 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %1833 = tosa.matmul %1831, %1832 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %cst_429 = arith.constant 0.0883883461 : f32
    %splat_430 = tensor.splat %cst_429 : tensor<32x40x40xf32>
    %1834 = tosa.mul %1833, %splat_430 {shift = 0 : i8} : (tensor<32x40x40xf32>, tensor<32x40x40xf32>) -> tensor<32x40x40xf32>
    %1835 = tosa.add %1834, %1828 : (tensor<32x40x40xf32>, tensor<40x40xf32>) -> tensor<32x40x40xf32>
    %1836 = tosa.reduce_max %1835 {axis = 2 : i32} : (tensor<32x40x40xf32>) -> tensor<32x40x1xf32>
    %1837 = tosa.sub %1835, %1836 : (tensor<32x40x40xf32>, tensor<32x40x1xf32>) -> tensor<32x40x40xf32>
    %1838 = math.exp %1837 : tensor<32x40x40xf32>
    %1839 = tosa.reduce_sum %1838 {axis = 2 : i32} : (tensor<32x40x40xf32>) -> tensor<32x40x1xf32>
    %1840 = tosa.log %1839 : (tensor<32x40x1xf32>) -> tensor<32x40x1xf32>
    %1841 = tosa.add %1836, %1840 : (tensor<32x40x1xf32>, tensor<32x40x1xf32>) -> tensor<32x40x1xf32>
    %1842 = tosa.sub %1835, %1841 : (tensor<32x40x40xf32>, tensor<32x40x1xf32>) -> tensor<32x40x40xf32>
    %1843 = math.exp %1842 : tensor<32x40x40xf32>
    %1844 = tosa.reshape %1841 {new_shape = array<i64: 1, 32, 40>} : (tensor<32x40x1xf32>) -> tensor<1x32x40xf32>
    %1845 = tosa.reshape %1808 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %1846 = tosa.matmul %1843, %1845 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %1847 = tosa.reshape %1846 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1848 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1849 = tosa.transpose %1847, %1848 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %1850 = tosa.reshape %1849 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %1851 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1852 = tosa.transpose %arg142, %1851 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1853 = tosa.reshape %1850 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_431 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1854 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1853, %1852 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_431 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1855 = tosa.reshape %1854 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1856 = tosa.add %1772, %1855 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1857 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_432 = arith.constant 2 : i32
    %1858 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1856 : tensor<1x40x4096xf32>) outs(%1857 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3745 = math.fpowi %in, %c2_i32_432 : f32, i32
      linalg.yield %3745 : f32
    } -> tensor<1x40x4096xf32>
    %1859 = tosa.reduce_sum %1858 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %1860 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1861 = tosa.reciprocal %1860 : (tensor<1xf32>) -> tensor<1xf32>
    %1862 = tosa.mul %1861, %1859 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1863 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %1864 = tosa.add %1862, %1863 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1865 = tosa.rsqrt %1864 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1866 = tosa.mul %1856, %1865 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %1867 = tosa.reshape %arg143 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %1868 = tosa.mul %1867, %1866 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1869 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1870 = tosa.transpose %arg144, %1869 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %1871 = tosa.reshape %1868 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_433 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %1872 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1871, %1870 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_433 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %1873 = tosa.reshape %1872 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %1874 = tosa.sigmoid %1873 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %1875 = tosa.mul %1873, %1874 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %1876 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1877 = tosa.transpose %arg145, %1876 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %1878 = tosa.reshape %1868 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_434 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %1879 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1878, %1877 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_434 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %1880 = tosa.reshape %1879 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %1881 = tosa.mul %1875, %1880 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %1882 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1883 = tosa.transpose %arg146, %1882 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %1884 = tosa.reshape %1881 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_435 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1885 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1884, %1883 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_435 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1886 = tosa.reshape %1885 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1887 = tosa.add %1856, %1886 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1888 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_436 = arith.constant 2 : i32
    %1889 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1887 : tensor<1x40x4096xf32>) outs(%1888 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3745 = math.fpowi %in, %c2_i32_436 : f32, i32
      linalg.yield %3745 : f32
    } -> tensor<1x40x4096xf32>
    %1890 = tosa.reduce_sum %1889 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %1891 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1892 = tosa.reciprocal %1891 : (tensor<1xf32>) -> tensor<1xf32>
    %1893 = tosa.mul %1892, %1890 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1894 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %1895 = tosa.add %1893, %1894 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1896 = tosa.rsqrt %1895 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1897 = tosa.mul %1887, %1896 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %1898 = tosa.reshape %arg147 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %1899 = tosa.mul %1898, %1897 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1900 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1901 = tosa.transpose %arg148, %1900 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1902 = tosa.reshape %1899 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_437 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1903 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1902, %1901 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_437 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1904 = tosa.reshape %1903 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1905 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1906 = tosa.transpose %arg149, %1905 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1907 = tosa.reshape %1899 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_438 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1908 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1907, %1906 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_438 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1909 = tosa.reshape %1908 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1910 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1911 = tosa.transpose %arg150, %1910 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1912 = tosa.reshape %1899 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_439 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1913 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1912, %1911 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_439 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1914 = tosa.reshape %1913 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1915 = tosa.reshape %1904 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %1916 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1917 = tosa.transpose %1915, %1916 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %1918 = tosa.reshape %1909 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %1919 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1920 = tosa.transpose %1918, %1919 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %1921 = tosa.reshape %1914 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %1922 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1923 = tosa.transpose %1921, %1922 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %1924 = tosa.reshape %45 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %1925 = tosa.reshape %47 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %1926 = tosa.mul %1917, %1924 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_440 = tensor.extract_slice %1917[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_441 = tensor.extract_slice %1917[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %1927 = tensor.empty() : tensor<1x32x40x64xf32>
    %1928 = linalg.negf ins(%extracted_slice_441 : tensor<1x32x40x64xf32>) outs(%1927 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %1929 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_442 = tensor.insert_slice %1928 into %1929[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_443 = tensor.insert_slice %extracted_slice_440 into %inserted_slice_442[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %1930 = tosa.mul %inserted_slice_443, %1925 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1931 = tosa.add %1926, %1930 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1932 = tosa.mul %1920, %1924 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_444 = tensor.extract_slice %1920[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_445 = tensor.extract_slice %1920[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %1933 = tensor.empty() : tensor<1x32x40x64xf32>
    %1934 = linalg.negf ins(%extracted_slice_445 : tensor<1x32x40x64xf32>) outs(%1933 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %1935 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_446 = tensor.insert_slice %1934 into %1935[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_447 = tensor.insert_slice %extracted_slice_444 into %inserted_slice_446[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %1936 = tosa.mul %inserted_slice_447, %1925 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1937 = tosa.add %1932, %1936 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1938 = tosa.reshape %19 {new_shape = array<i64: 1, 40, 41>} : (tensor<40x41xf32>) -> tensor<1x40x41xf32>
    %1939 = tosa.reshape %1938 {new_shape = array<i64: 1, 1, 40, 41>} : (tensor<1x40x41xf32>) -> tensor<1x1x40x41xf32>
    %extracted_slice_448 = tensor.extract_slice %1939[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_449 = tensor.extract_slice %extracted_slice_448[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %1940 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x1x40x41xf32>}> : () -> tensor<1x1x40x41xf32>
    %1941 = tosa.add %extracted_slice_449, %1940 : (tensor<1x1x40x41xf32>, tensor<1x1x40x41xf32>) -> tensor<1x1x40x41xf32>
    %extracted_slice_450 = tensor.extract_slice %1941[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_451 = tensor.extract_slice %extracted_slice_450[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_452 = tensor.extract_slice %extracted_slice_451[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_453 = tensor.extract_slice %extracted_slice_452[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x40xf32>
    %cst_454 = arith.constant 0.000000e+00 : f32
    %splat_455 = tensor.splat %cst_454 : tensor<40x40xf32>
    %1942 = tosa.reshape %extracted_slice_453 {new_shape = array<i64: 40, 40>} : (tensor<1x1x40x40xf32>) -> tensor<40x40xf32>
    %1943 = tosa.add %splat_455, %1942 : (tensor<40x40xf32>, tensor<40x40xf32>) -> tensor<40x40xf32>
    %1944 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1945 = tosa.transpose %1937, %1944 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %1946 = tosa.reshape %1931 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %1947 = tosa.reshape %1945 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %1948 = tosa.matmul %1946, %1947 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %cst_456 = arith.constant 0.0883883461 : f32
    %splat_457 = tensor.splat %cst_456 : tensor<32x40x40xf32>
    %1949 = tosa.mul %1948, %splat_457 {shift = 0 : i8} : (tensor<32x40x40xf32>, tensor<32x40x40xf32>) -> tensor<32x40x40xf32>
    %1950 = tosa.add %1949, %1943 : (tensor<32x40x40xf32>, tensor<40x40xf32>) -> tensor<32x40x40xf32>
    %1951 = tosa.reduce_max %1950 {axis = 2 : i32} : (tensor<32x40x40xf32>) -> tensor<32x40x1xf32>
    %1952 = tosa.sub %1950, %1951 : (tensor<32x40x40xf32>, tensor<32x40x1xf32>) -> tensor<32x40x40xf32>
    %1953 = math.exp %1952 : tensor<32x40x40xf32>
    %1954 = tosa.reduce_sum %1953 {axis = 2 : i32} : (tensor<32x40x40xf32>) -> tensor<32x40x1xf32>
    %1955 = tosa.log %1954 : (tensor<32x40x1xf32>) -> tensor<32x40x1xf32>
    %1956 = tosa.add %1951, %1955 : (tensor<32x40x1xf32>, tensor<32x40x1xf32>) -> tensor<32x40x1xf32>
    %1957 = tosa.sub %1950, %1956 : (tensor<32x40x40xf32>, tensor<32x40x1xf32>) -> tensor<32x40x40xf32>
    %1958 = math.exp %1957 : tensor<32x40x40xf32>
    %1959 = tosa.reshape %1956 {new_shape = array<i64: 1, 32, 40>} : (tensor<32x40x1xf32>) -> tensor<1x32x40xf32>
    %1960 = tosa.reshape %1923 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %1961 = tosa.matmul %1958, %1960 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %1962 = tosa.reshape %1961 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1963 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1964 = tosa.transpose %1962, %1963 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %1965 = tosa.reshape %1964 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %1966 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1967 = tosa.transpose %arg151, %1966 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1968 = tosa.reshape %1965 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_458 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1969 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1968, %1967 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_458 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1970 = tosa.reshape %1969 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1971 = tosa.add %1887, %1970 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1972 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_459 = arith.constant 2 : i32
    %1973 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1971 : tensor<1x40x4096xf32>) outs(%1972 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3745 = math.fpowi %in, %c2_i32_459 : f32, i32
      linalg.yield %3745 : f32
    } -> tensor<1x40x4096xf32>
    %1974 = tosa.reduce_sum %1973 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %1975 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1976 = tosa.reciprocal %1975 : (tensor<1xf32>) -> tensor<1xf32>
    %1977 = tosa.mul %1976, %1974 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1978 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %1979 = tosa.add %1977, %1978 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1980 = tosa.rsqrt %1979 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1981 = tosa.mul %1971, %1980 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %1982 = tosa.reshape %arg152 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %1983 = tosa.mul %1982, %1981 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1984 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1985 = tosa.transpose %arg153, %1984 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %1986 = tosa.reshape %1983 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_460 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %1987 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1986, %1985 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_460 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %1988 = tosa.reshape %1987 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %1989 = tosa.sigmoid %1988 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %1990 = tosa.mul %1988, %1989 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %1991 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1992 = tosa.transpose %arg154, %1991 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %1993 = tosa.reshape %1983 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_461 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %1994 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1993, %1992 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_461 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %1995 = tosa.reshape %1994 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %1996 = tosa.mul %1990, %1995 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %1997 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1998 = tosa.transpose %arg155, %1997 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %1999 = tosa.reshape %1996 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_462 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2000 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1999, %1998 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_462 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2001 = tosa.reshape %2000 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2002 = tosa.add %1971, %2001 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2003 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_463 = arith.constant 2 : i32
    %2004 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2002 : tensor<1x40x4096xf32>) outs(%2003 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3745 = math.fpowi %in, %c2_i32_463 : f32, i32
      linalg.yield %3745 : f32
    } -> tensor<1x40x4096xf32>
    %2005 = tosa.reduce_sum %2004 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %2006 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2007 = tosa.reciprocal %2006 : (tensor<1xf32>) -> tensor<1xf32>
    %2008 = tosa.mul %2007, %2005 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2009 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %2010 = tosa.add %2008, %2009 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2011 = tosa.rsqrt %2010 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2012 = tosa.mul %2002, %2011 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %2013 = tosa.reshape %arg156 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %2014 = tosa.mul %2013, %2012 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2015 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2016 = tosa.transpose %arg157, %2015 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %2017 = tosa.reshape %2014 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_464 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2018 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2017, %2016 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_464 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2019 = tosa.reshape %2018 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2020 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2021 = tosa.transpose %arg158, %2020 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %2022 = tosa.reshape %2014 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_465 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2023 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2022, %2021 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_465 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2024 = tosa.reshape %2023 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2025 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2026 = tosa.transpose %arg159, %2025 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %2027 = tosa.reshape %2014 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_466 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2028 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2027, %2026 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_466 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2029 = tosa.reshape %2028 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2030 = tosa.reshape %2019 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %2031 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2032 = tosa.transpose %2030, %2031 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %2033 = tosa.reshape %2024 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %2034 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2035 = tosa.transpose %2033, %2034 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %2036 = tosa.reshape %2029 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %2037 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2038 = tosa.transpose %2036, %2037 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %2039 = tosa.reshape %45 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %2040 = tosa.reshape %47 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %2041 = tosa.mul %2032, %2039 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_467 = tensor.extract_slice %2032[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_468 = tensor.extract_slice %2032[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %2042 = tensor.empty() : tensor<1x32x40x64xf32>
    %2043 = linalg.negf ins(%extracted_slice_468 : tensor<1x32x40x64xf32>) outs(%2042 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %2044 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_469 = tensor.insert_slice %2043 into %2044[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_470 = tensor.insert_slice %extracted_slice_467 into %inserted_slice_469[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %2045 = tosa.mul %inserted_slice_470, %2040 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2046 = tosa.add %2041, %2045 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2047 = tosa.mul %2035, %2039 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_471 = tensor.extract_slice %2035[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_472 = tensor.extract_slice %2035[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %2048 = tensor.empty() : tensor<1x32x40x64xf32>
    %2049 = linalg.negf ins(%extracted_slice_472 : tensor<1x32x40x64xf32>) outs(%2048 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %2050 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_473 = tensor.insert_slice %2049 into %2050[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_474 = tensor.insert_slice %extracted_slice_471 into %inserted_slice_473[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %2051 = tosa.mul %inserted_slice_474, %2040 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2052 = tosa.add %2047, %2051 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2053 = tosa.reshape %19 {new_shape = array<i64: 1, 40, 41>} : (tensor<40x41xf32>) -> tensor<1x40x41xf32>
    %2054 = tosa.reshape %2053 {new_shape = array<i64: 1, 1, 40, 41>} : (tensor<1x40x41xf32>) -> tensor<1x1x40x41xf32>
    %extracted_slice_475 = tensor.extract_slice %2054[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_476 = tensor.extract_slice %extracted_slice_475[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %2055 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x1x40x41xf32>}> : () -> tensor<1x1x40x41xf32>
    %2056 = tosa.add %extracted_slice_476, %2055 : (tensor<1x1x40x41xf32>, tensor<1x1x40x41xf32>) -> tensor<1x1x40x41xf32>
    %extracted_slice_477 = tensor.extract_slice %2056[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_478 = tensor.extract_slice %extracted_slice_477[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_479 = tensor.extract_slice %extracted_slice_478[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_480 = tensor.extract_slice %extracted_slice_479[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x40xf32>
    %cst_481 = arith.constant 0.000000e+00 : f32
    %splat_482 = tensor.splat %cst_481 : tensor<40x40xf32>
    %2057 = tosa.reshape %extracted_slice_480 {new_shape = array<i64: 40, 40>} : (tensor<1x1x40x40xf32>) -> tensor<40x40xf32>
    %2058 = tosa.add %splat_482, %2057 : (tensor<40x40xf32>, tensor<40x40xf32>) -> tensor<40x40xf32>
    %2059 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2060 = tosa.transpose %2052, %2059 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %2061 = tosa.reshape %2046 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %2062 = tosa.reshape %2060 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %2063 = tosa.matmul %2061, %2062 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %cst_483 = arith.constant 0.0883883461 : f32
    %splat_484 = tensor.splat %cst_483 : tensor<32x40x40xf32>
    %2064 = tosa.mul %2063, %splat_484 {shift = 0 : i8} : (tensor<32x40x40xf32>, tensor<32x40x40xf32>) -> tensor<32x40x40xf32>
    %2065 = tosa.add %2064, %2058 : (tensor<32x40x40xf32>, tensor<40x40xf32>) -> tensor<32x40x40xf32>
    %2066 = tosa.reduce_max %2065 {axis = 2 : i32} : (tensor<32x40x40xf32>) -> tensor<32x40x1xf32>
    %2067 = tosa.sub %2065, %2066 : (tensor<32x40x40xf32>, tensor<32x40x1xf32>) -> tensor<32x40x40xf32>
    %2068 = math.exp %2067 : tensor<32x40x40xf32>
    %2069 = tosa.reduce_sum %2068 {axis = 2 : i32} : (tensor<32x40x40xf32>) -> tensor<32x40x1xf32>
    %2070 = tosa.log %2069 : (tensor<32x40x1xf32>) -> tensor<32x40x1xf32>
    %2071 = tosa.add %2066, %2070 : (tensor<32x40x1xf32>, tensor<32x40x1xf32>) -> tensor<32x40x1xf32>
    %2072 = tosa.sub %2065, %2071 : (tensor<32x40x40xf32>, tensor<32x40x1xf32>) -> tensor<32x40x40xf32>
    %2073 = math.exp %2072 : tensor<32x40x40xf32>
    %2074 = tosa.reshape %2071 {new_shape = array<i64: 1, 32, 40>} : (tensor<32x40x1xf32>) -> tensor<1x32x40xf32>
    %2075 = tosa.reshape %2038 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %2076 = tosa.matmul %2073, %2075 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %2077 = tosa.reshape %2076 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2078 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2079 = tosa.transpose %2077, %2078 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %2080 = tosa.reshape %2079 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %2081 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2082 = tosa.transpose %arg160, %2081 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %2083 = tosa.reshape %2080 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_485 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2084 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2083, %2082 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_485 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2085 = tosa.reshape %2084 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2086 = tosa.add %2002, %2085 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2087 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_486 = arith.constant 2 : i32
    %2088 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2086 : tensor<1x40x4096xf32>) outs(%2087 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3745 = math.fpowi %in, %c2_i32_486 : f32, i32
      linalg.yield %3745 : f32
    } -> tensor<1x40x4096xf32>
    %2089 = tosa.reduce_sum %2088 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %2090 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2091 = tosa.reciprocal %2090 : (tensor<1xf32>) -> tensor<1xf32>
    %2092 = tosa.mul %2091, %2089 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2093 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %2094 = tosa.add %2092, %2093 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2095 = tosa.rsqrt %2094 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2096 = tosa.mul %2086, %2095 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %2097 = tosa.reshape %arg161 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %2098 = tosa.mul %2097, %2096 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2099 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2100 = tosa.transpose %arg162, %2099 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %2101 = tosa.reshape %2098 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_487 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %2102 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2101, %2100 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_487 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %2103 = tosa.reshape %2102 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %2104 = tosa.sigmoid %2103 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %2105 = tosa.mul %2103, %2104 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %2106 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2107 = tosa.transpose %arg163, %2106 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %2108 = tosa.reshape %2098 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_488 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %2109 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2108, %2107 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_488 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %2110 = tosa.reshape %2109 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %2111 = tosa.mul %2105, %2110 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %2112 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2113 = tosa.transpose %arg164, %2112 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %2114 = tosa.reshape %2111 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_489 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2115 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2114, %2113 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_489 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2116 = tosa.reshape %2115 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2117 = tosa.add %2086, %2116 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2118 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_490 = arith.constant 2 : i32
    %2119 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2117 : tensor<1x40x4096xf32>) outs(%2118 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3745 = math.fpowi %in, %c2_i32_490 : f32, i32
      linalg.yield %3745 : f32
    } -> tensor<1x40x4096xf32>
    %2120 = tosa.reduce_sum %2119 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %2121 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2122 = tosa.reciprocal %2121 : (tensor<1xf32>) -> tensor<1xf32>
    %2123 = tosa.mul %2122, %2120 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2124 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %2125 = tosa.add %2123, %2124 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2126 = tosa.rsqrt %2125 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2127 = tosa.mul %2117, %2126 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %2128 = tosa.reshape %arg165 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %2129 = tosa.mul %2128, %2127 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2130 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2131 = tosa.transpose %arg166, %2130 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %2132 = tosa.reshape %2129 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_491 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2133 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2132, %2131 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_491 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2134 = tosa.reshape %2133 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2135 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2136 = tosa.transpose %arg167, %2135 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %2137 = tosa.reshape %2129 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_492 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2138 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2137, %2136 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_492 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2139 = tosa.reshape %2138 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2140 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2141 = tosa.transpose %arg168, %2140 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %2142 = tosa.reshape %2129 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_493 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2143 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2142, %2141 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_493 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2144 = tosa.reshape %2143 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2145 = tosa.reshape %2134 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %2146 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2147 = tosa.transpose %2145, %2146 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %2148 = tosa.reshape %2139 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %2149 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2150 = tosa.transpose %2148, %2149 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %2151 = tosa.reshape %2144 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %2152 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2153 = tosa.transpose %2151, %2152 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %2154 = tosa.reshape %45 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %2155 = tosa.reshape %47 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %2156 = tosa.mul %2147, %2154 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_494 = tensor.extract_slice %2147[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_495 = tensor.extract_slice %2147[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %2157 = tensor.empty() : tensor<1x32x40x64xf32>
    %2158 = linalg.negf ins(%extracted_slice_495 : tensor<1x32x40x64xf32>) outs(%2157 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %2159 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_496 = tensor.insert_slice %2158 into %2159[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_497 = tensor.insert_slice %extracted_slice_494 into %inserted_slice_496[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %2160 = tosa.mul %inserted_slice_497, %2155 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2161 = tosa.add %2156, %2160 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2162 = tosa.mul %2150, %2154 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_498 = tensor.extract_slice %2150[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_499 = tensor.extract_slice %2150[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %2163 = tensor.empty() : tensor<1x32x40x64xf32>
    %2164 = linalg.negf ins(%extracted_slice_499 : tensor<1x32x40x64xf32>) outs(%2163 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %2165 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_500 = tensor.insert_slice %2164 into %2165[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_501 = tensor.insert_slice %extracted_slice_498 into %inserted_slice_500[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %2166 = tosa.mul %inserted_slice_501, %2155 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2167 = tosa.add %2162, %2166 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2168 = tosa.reshape %19 {new_shape = array<i64: 1, 40, 41>} : (tensor<40x41xf32>) -> tensor<1x40x41xf32>
    %2169 = tosa.reshape %2168 {new_shape = array<i64: 1, 1, 40, 41>} : (tensor<1x40x41xf32>) -> tensor<1x1x40x41xf32>
    %extracted_slice_502 = tensor.extract_slice %2169[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_503 = tensor.extract_slice %extracted_slice_502[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %2170 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x1x40x41xf32>}> : () -> tensor<1x1x40x41xf32>
    %2171 = tosa.add %extracted_slice_503, %2170 : (tensor<1x1x40x41xf32>, tensor<1x1x40x41xf32>) -> tensor<1x1x40x41xf32>
    %extracted_slice_504 = tensor.extract_slice %2171[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_505 = tensor.extract_slice %extracted_slice_504[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_506 = tensor.extract_slice %extracted_slice_505[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_507 = tensor.extract_slice %extracted_slice_506[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x40xf32>
    %cst_508 = arith.constant 0.000000e+00 : f32
    %splat_509 = tensor.splat %cst_508 : tensor<40x40xf32>
    %2172 = tosa.reshape %extracted_slice_507 {new_shape = array<i64: 40, 40>} : (tensor<1x1x40x40xf32>) -> tensor<40x40xf32>
    %2173 = tosa.add %splat_509, %2172 : (tensor<40x40xf32>, tensor<40x40xf32>) -> tensor<40x40xf32>
    %2174 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2175 = tosa.transpose %2167, %2174 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %2176 = tosa.reshape %2161 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %2177 = tosa.reshape %2175 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %2178 = tosa.matmul %2176, %2177 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %cst_510 = arith.constant 0.0883883461 : f32
    %splat_511 = tensor.splat %cst_510 : tensor<32x40x40xf32>
    %2179 = tosa.mul %2178, %splat_511 {shift = 0 : i8} : (tensor<32x40x40xf32>, tensor<32x40x40xf32>) -> tensor<32x40x40xf32>
    %2180 = tosa.add %2179, %2173 : (tensor<32x40x40xf32>, tensor<40x40xf32>) -> tensor<32x40x40xf32>
    %2181 = tosa.reduce_max %2180 {axis = 2 : i32} : (tensor<32x40x40xf32>) -> tensor<32x40x1xf32>
    %2182 = tosa.sub %2180, %2181 : (tensor<32x40x40xf32>, tensor<32x40x1xf32>) -> tensor<32x40x40xf32>
    %2183 = math.exp %2182 : tensor<32x40x40xf32>
    %2184 = tosa.reduce_sum %2183 {axis = 2 : i32} : (tensor<32x40x40xf32>) -> tensor<32x40x1xf32>
    %2185 = tosa.log %2184 : (tensor<32x40x1xf32>) -> tensor<32x40x1xf32>
    %2186 = tosa.add %2181, %2185 : (tensor<32x40x1xf32>, tensor<32x40x1xf32>) -> tensor<32x40x1xf32>
    %2187 = tosa.sub %2180, %2186 : (tensor<32x40x40xf32>, tensor<32x40x1xf32>) -> tensor<32x40x40xf32>
    %2188 = math.exp %2187 : tensor<32x40x40xf32>
    %2189 = tosa.reshape %2186 {new_shape = array<i64: 1, 32, 40>} : (tensor<32x40x1xf32>) -> tensor<1x32x40xf32>
    %2190 = tosa.reshape %2153 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %2191 = tosa.matmul %2188, %2190 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %2192 = tosa.reshape %2191 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2193 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2194 = tosa.transpose %2192, %2193 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %2195 = tosa.reshape %2194 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %2196 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2197 = tosa.transpose %arg169, %2196 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %2198 = tosa.reshape %2195 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_512 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2199 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2198, %2197 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_512 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2200 = tosa.reshape %2199 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2201 = tosa.add %2117, %2200 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2202 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_513 = arith.constant 2 : i32
    %2203 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2201 : tensor<1x40x4096xf32>) outs(%2202 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3745 = math.fpowi %in, %c2_i32_513 : f32, i32
      linalg.yield %3745 : f32
    } -> tensor<1x40x4096xf32>
    %2204 = tosa.reduce_sum %2203 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %2205 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2206 = tosa.reciprocal %2205 : (tensor<1xf32>) -> tensor<1xf32>
    %2207 = tosa.mul %2206, %2204 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2208 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %2209 = tosa.add %2207, %2208 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2210 = tosa.rsqrt %2209 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2211 = tosa.mul %2201, %2210 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %2212 = tosa.reshape %arg170 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %2213 = tosa.mul %2212, %2211 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2214 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2215 = tosa.transpose %arg171, %2214 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %2216 = tosa.reshape %2213 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_514 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %2217 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2216, %2215 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_514 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %2218 = tosa.reshape %2217 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %2219 = tosa.sigmoid %2218 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %2220 = tosa.mul %2218, %2219 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %2221 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2222 = tosa.transpose %arg172, %2221 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %2223 = tosa.reshape %2213 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_515 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %2224 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2223, %2222 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_515 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %2225 = tosa.reshape %2224 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %2226 = tosa.mul %2220, %2225 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %2227 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2228 = tosa.transpose %arg173, %2227 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %2229 = tosa.reshape %2226 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_516 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2230 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2229, %2228 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_516 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2231 = tosa.reshape %2230 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2232 = tosa.add %2201, %2231 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2233 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_517 = arith.constant 2 : i32
    %2234 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2232 : tensor<1x40x4096xf32>) outs(%2233 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3745 = math.fpowi %in, %c2_i32_517 : f32, i32
      linalg.yield %3745 : f32
    } -> tensor<1x40x4096xf32>
    %2235 = tosa.reduce_sum %2234 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %2236 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2237 = tosa.reciprocal %2236 : (tensor<1xf32>) -> tensor<1xf32>
    %2238 = tosa.mul %2237, %2235 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2239 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %2240 = tosa.add %2238, %2239 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2241 = tosa.rsqrt %2240 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2242 = tosa.mul %2232, %2241 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %2243 = tosa.reshape %arg174 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %2244 = tosa.mul %2243, %2242 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2245 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2246 = tosa.transpose %arg175, %2245 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %2247 = tosa.reshape %2244 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_518 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2248 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2247, %2246 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_518 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2249 = tosa.reshape %2248 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2250 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2251 = tosa.transpose %arg176, %2250 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %2252 = tosa.reshape %2244 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_519 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2253 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2252, %2251 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_519 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2254 = tosa.reshape %2253 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2255 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2256 = tosa.transpose %arg177, %2255 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %2257 = tosa.reshape %2244 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_520 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2258 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2257, %2256 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_520 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2259 = tosa.reshape %2258 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2260 = tosa.reshape %2249 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %2261 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2262 = tosa.transpose %2260, %2261 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %2263 = tosa.reshape %2254 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %2264 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2265 = tosa.transpose %2263, %2264 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %2266 = tosa.reshape %2259 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %2267 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2268 = tosa.transpose %2266, %2267 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %2269 = tosa.reshape %45 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %2270 = tosa.reshape %47 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %2271 = tosa.mul %2262, %2269 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_521 = tensor.extract_slice %2262[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_522 = tensor.extract_slice %2262[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %2272 = tensor.empty() : tensor<1x32x40x64xf32>
    %2273 = linalg.negf ins(%extracted_slice_522 : tensor<1x32x40x64xf32>) outs(%2272 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %2274 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_523 = tensor.insert_slice %2273 into %2274[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_524 = tensor.insert_slice %extracted_slice_521 into %inserted_slice_523[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %2275 = tosa.mul %inserted_slice_524, %2270 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2276 = tosa.add %2271, %2275 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2277 = tosa.mul %2265, %2269 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_525 = tensor.extract_slice %2265[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_526 = tensor.extract_slice %2265[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %2278 = tensor.empty() : tensor<1x32x40x64xf32>
    %2279 = linalg.negf ins(%extracted_slice_526 : tensor<1x32x40x64xf32>) outs(%2278 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %2280 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_527 = tensor.insert_slice %2279 into %2280[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_528 = tensor.insert_slice %extracted_slice_525 into %inserted_slice_527[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %2281 = tosa.mul %inserted_slice_528, %2270 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2282 = tosa.add %2277, %2281 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2283 = tosa.reshape %19 {new_shape = array<i64: 1, 40, 41>} : (tensor<40x41xf32>) -> tensor<1x40x41xf32>
    %2284 = tosa.reshape %2283 {new_shape = array<i64: 1, 1, 40, 41>} : (tensor<1x40x41xf32>) -> tensor<1x1x40x41xf32>
    %extracted_slice_529 = tensor.extract_slice %2284[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_530 = tensor.extract_slice %extracted_slice_529[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %2285 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x1x40x41xf32>}> : () -> tensor<1x1x40x41xf32>
    %2286 = tosa.add %extracted_slice_530, %2285 : (tensor<1x1x40x41xf32>, tensor<1x1x40x41xf32>) -> tensor<1x1x40x41xf32>
    %extracted_slice_531 = tensor.extract_slice %2286[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_532 = tensor.extract_slice %extracted_slice_531[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_533 = tensor.extract_slice %extracted_slice_532[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_534 = tensor.extract_slice %extracted_slice_533[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x40xf32>
    %cst_535 = arith.constant 0.000000e+00 : f32
    %splat_536 = tensor.splat %cst_535 : tensor<40x40xf32>
    %2287 = tosa.reshape %extracted_slice_534 {new_shape = array<i64: 40, 40>} : (tensor<1x1x40x40xf32>) -> tensor<40x40xf32>
    %2288 = tosa.add %splat_536, %2287 : (tensor<40x40xf32>, tensor<40x40xf32>) -> tensor<40x40xf32>
    %2289 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2290 = tosa.transpose %2282, %2289 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %2291 = tosa.reshape %2276 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %2292 = tosa.reshape %2290 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %2293 = tosa.matmul %2291, %2292 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %cst_537 = arith.constant 0.0883883461 : f32
    %splat_538 = tensor.splat %cst_537 : tensor<32x40x40xf32>
    %2294 = tosa.mul %2293, %splat_538 {shift = 0 : i8} : (tensor<32x40x40xf32>, tensor<32x40x40xf32>) -> tensor<32x40x40xf32>
    %2295 = tosa.add %2294, %2288 : (tensor<32x40x40xf32>, tensor<40x40xf32>) -> tensor<32x40x40xf32>
    %2296 = tosa.reduce_max %2295 {axis = 2 : i32} : (tensor<32x40x40xf32>) -> tensor<32x40x1xf32>
    %2297 = tosa.sub %2295, %2296 : (tensor<32x40x40xf32>, tensor<32x40x1xf32>) -> tensor<32x40x40xf32>
    %2298 = math.exp %2297 : tensor<32x40x40xf32>
    %2299 = tosa.reduce_sum %2298 {axis = 2 : i32} : (tensor<32x40x40xf32>) -> tensor<32x40x1xf32>
    %2300 = tosa.log %2299 : (tensor<32x40x1xf32>) -> tensor<32x40x1xf32>
    %2301 = tosa.add %2296, %2300 : (tensor<32x40x1xf32>, tensor<32x40x1xf32>) -> tensor<32x40x1xf32>
    %2302 = tosa.sub %2295, %2301 : (tensor<32x40x40xf32>, tensor<32x40x1xf32>) -> tensor<32x40x40xf32>
    %2303 = math.exp %2302 : tensor<32x40x40xf32>
    %2304 = tosa.reshape %2301 {new_shape = array<i64: 1, 32, 40>} : (tensor<32x40x1xf32>) -> tensor<1x32x40xf32>
    %2305 = tosa.reshape %2268 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %2306 = tosa.matmul %2303, %2305 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %2307 = tosa.reshape %2306 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2308 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2309 = tosa.transpose %2307, %2308 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %2310 = tosa.reshape %2309 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %2311 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2312 = tosa.transpose %arg178, %2311 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %2313 = tosa.reshape %2310 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_539 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2314 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2313, %2312 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_539 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2315 = tosa.reshape %2314 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2316 = tosa.add %2232, %2315 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2317 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_540 = arith.constant 2 : i32
    %2318 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2316 : tensor<1x40x4096xf32>) outs(%2317 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3745 = math.fpowi %in, %c2_i32_540 : f32, i32
      linalg.yield %3745 : f32
    } -> tensor<1x40x4096xf32>
    %2319 = tosa.reduce_sum %2318 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %2320 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2321 = tosa.reciprocal %2320 : (tensor<1xf32>) -> tensor<1xf32>
    %2322 = tosa.mul %2321, %2319 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2323 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %2324 = tosa.add %2322, %2323 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2325 = tosa.rsqrt %2324 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2326 = tosa.mul %2316, %2325 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %2327 = tosa.reshape %arg179 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %2328 = tosa.mul %2327, %2326 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2329 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2330 = tosa.transpose %arg180, %2329 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %2331 = tosa.reshape %2328 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_541 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %2332 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2331, %2330 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_541 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %2333 = tosa.reshape %2332 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %2334 = tosa.sigmoid %2333 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %2335 = tosa.mul %2333, %2334 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %2336 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2337 = tosa.transpose %arg181, %2336 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %2338 = tosa.reshape %2328 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_542 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %2339 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2338, %2337 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_542 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %2340 = tosa.reshape %2339 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %2341 = tosa.mul %2335, %2340 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %2342 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2343 = tosa.transpose %arg182, %2342 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %2344 = tosa.reshape %2341 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_543 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2345 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2344, %2343 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_543 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2346 = tosa.reshape %2345 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2347 = tosa.add %2316, %2346 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2348 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_544 = arith.constant 2 : i32
    %2349 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2347 : tensor<1x40x4096xf32>) outs(%2348 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3745 = math.fpowi %in, %c2_i32_544 : f32, i32
      linalg.yield %3745 : f32
    } -> tensor<1x40x4096xf32>
    %2350 = tosa.reduce_sum %2349 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %2351 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2352 = tosa.reciprocal %2351 : (tensor<1xf32>) -> tensor<1xf32>
    %2353 = tosa.mul %2352, %2350 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2354 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %2355 = tosa.add %2353, %2354 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2356 = tosa.rsqrt %2355 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2357 = tosa.mul %2347, %2356 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %2358 = tosa.reshape %arg183 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %2359 = tosa.mul %2358, %2357 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2360 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2361 = tosa.transpose %arg184, %2360 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %2362 = tosa.reshape %2359 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_545 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2363 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2362, %2361 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_545 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2364 = tosa.reshape %2363 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2365 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2366 = tosa.transpose %arg185, %2365 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %2367 = tosa.reshape %2359 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_546 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2368 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2367, %2366 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_546 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2369 = tosa.reshape %2368 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2370 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2371 = tosa.transpose %arg186, %2370 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %2372 = tosa.reshape %2359 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_547 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2373 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2372, %2371 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_547 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2374 = tosa.reshape %2373 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2375 = tosa.reshape %2364 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %2376 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2377 = tosa.transpose %2375, %2376 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %2378 = tosa.reshape %2369 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %2379 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2380 = tosa.transpose %2378, %2379 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %2381 = tosa.reshape %2374 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %2382 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2383 = tosa.transpose %2381, %2382 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %2384 = tosa.reshape %45 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %2385 = tosa.reshape %47 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %2386 = tosa.mul %2377, %2384 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_548 = tensor.extract_slice %2377[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_549 = tensor.extract_slice %2377[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %2387 = tensor.empty() : tensor<1x32x40x64xf32>
    %2388 = linalg.negf ins(%extracted_slice_549 : tensor<1x32x40x64xf32>) outs(%2387 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %2389 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_550 = tensor.insert_slice %2388 into %2389[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_551 = tensor.insert_slice %extracted_slice_548 into %inserted_slice_550[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %2390 = tosa.mul %inserted_slice_551, %2385 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2391 = tosa.add %2386, %2390 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2392 = tosa.mul %2380, %2384 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_552 = tensor.extract_slice %2380[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_553 = tensor.extract_slice %2380[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %2393 = tensor.empty() : tensor<1x32x40x64xf32>
    %2394 = linalg.negf ins(%extracted_slice_553 : tensor<1x32x40x64xf32>) outs(%2393 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %2395 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_554 = tensor.insert_slice %2394 into %2395[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_555 = tensor.insert_slice %extracted_slice_552 into %inserted_slice_554[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %2396 = tosa.mul %inserted_slice_555, %2385 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2397 = tosa.add %2392, %2396 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2398 = tosa.reshape %19 {new_shape = array<i64: 1, 40, 41>} : (tensor<40x41xf32>) -> tensor<1x40x41xf32>
    %2399 = tosa.reshape %2398 {new_shape = array<i64: 1, 1, 40, 41>} : (tensor<1x40x41xf32>) -> tensor<1x1x40x41xf32>
    %extracted_slice_556 = tensor.extract_slice %2399[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_557 = tensor.extract_slice %extracted_slice_556[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %2400 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x1x40x41xf32>}> : () -> tensor<1x1x40x41xf32>
    %2401 = tosa.add %extracted_slice_557, %2400 : (tensor<1x1x40x41xf32>, tensor<1x1x40x41xf32>) -> tensor<1x1x40x41xf32>
    %extracted_slice_558 = tensor.extract_slice %2401[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_559 = tensor.extract_slice %extracted_slice_558[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_560 = tensor.extract_slice %extracted_slice_559[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_561 = tensor.extract_slice %extracted_slice_560[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x40xf32>
    %cst_562 = arith.constant 0.000000e+00 : f32
    %splat_563 = tensor.splat %cst_562 : tensor<40x40xf32>
    %2402 = tosa.reshape %extracted_slice_561 {new_shape = array<i64: 40, 40>} : (tensor<1x1x40x40xf32>) -> tensor<40x40xf32>
    %2403 = tosa.add %splat_563, %2402 : (tensor<40x40xf32>, tensor<40x40xf32>) -> tensor<40x40xf32>
    %2404 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2405 = tosa.transpose %2397, %2404 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %2406 = tosa.reshape %2391 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %2407 = tosa.reshape %2405 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %2408 = tosa.matmul %2406, %2407 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %cst_564 = arith.constant 0.0883883461 : f32
    %splat_565 = tensor.splat %cst_564 : tensor<32x40x40xf32>
    %2409 = tosa.mul %2408, %splat_565 {shift = 0 : i8} : (tensor<32x40x40xf32>, tensor<32x40x40xf32>) -> tensor<32x40x40xf32>
    %2410 = tosa.add %2409, %2403 : (tensor<32x40x40xf32>, tensor<40x40xf32>) -> tensor<32x40x40xf32>
    %2411 = tosa.reduce_max %2410 {axis = 2 : i32} : (tensor<32x40x40xf32>) -> tensor<32x40x1xf32>
    %2412 = tosa.sub %2410, %2411 : (tensor<32x40x40xf32>, tensor<32x40x1xf32>) -> tensor<32x40x40xf32>
    %2413 = math.exp %2412 : tensor<32x40x40xf32>
    %2414 = tosa.reduce_sum %2413 {axis = 2 : i32} : (tensor<32x40x40xf32>) -> tensor<32x40x1xf32>
    %2415 = tosa.log %2414 : (tensor<32x40x1xf32>) -> tensor<32x40x1xf32>
    %2416 = tosa.add %2411, %2415 : (tensor<32x40x1xf32>, tensor<32x40x1xf32>) -> tensor<32x40x1xf32>
    %2417 = tosa.sub %2410, %2416 : (tensor<32x40x40xf32>, tensor<32x40x1xf32>) -> tensor<32x40x40xf32>
    %2418 = math.exp %2417 : tensor<32x40x40xf32>
    %2419 = tosa.reshape %2416 {new_shape = array<i64: 1, 32, 40>} : (tensor<32x40x1xf32>) -> tensor<1x32x40xf32>
    %2420 = tosa.reshape %2383 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %2421 = tosa.matmul %2418, %2420 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %2422 = tosa.reshape %2421 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2423 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2424 = tosa.transpose %2422, %2423 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %2425 = tosa.reshape %2424 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %2426 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2427 = tosa.transpose %arg187, %2426 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %2428 = tosa.reshape %2425 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_566 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2429 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2428, %2427 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_566 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2430 = tosa.reshape %2429 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2431 = tosa.add %2347, %2430 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2432 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_567 = arith.constant 2 : i32
    %2433 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2431 : tensor<1x40x4096xf32>) outs(%2432 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3745 = math.fpowi %in, %c2_i32_567 : f32, i32
      linalg.yield %3745 : f32
    } -> tensor<1x40x4096xf32>
    %2434 = tosa.reduce_sum %2433 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %2435 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2436 = tosa.reciprocal %2435 : (tensor<1xf32>) -> tensor<1xf32>
    %2437 = tosa.mul %2436, %2434 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2438 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %2439 = tosa.add %2437, %2438 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2440 = tosa.rsqrt %2439 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2441 = tosa.mul %2431, %2440 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %2442 = tosa.reshape %arg188 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %2443 = tosa.mul %2442, %2441 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2444 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2445 = tosa.transpose %arg189, %2444 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %2446 = tosa.reshape %2443 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_568 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %2447 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2446, %2445 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_568 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %2448 = tosa.reshape %2447 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %2449 = tosa.sigmoid %2448 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %2450 = tosa.mul %2448, %2449 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %2451 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2452 = tosa.transpose %arg190, %2451 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %2453 = tosa.reshape %2443 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_569 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %2454 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2453, %2452 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_569 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %2455 = tosa.reshape %2454 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %2456 = tosa.mul %2450, %2455 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %2457 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2458 = tosa.transpose %arg191, %2457 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %2459 = tosa.reshape %2456 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_570 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2460 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2459, %2458 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_570 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2461 = tosa.reshape %2460 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2462 = tosa.add %2431, %2461 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2463 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_571 = arith.constant 2 : i32
    %2464 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2462 : tensor<1x40x4096xf32>) outs(%2463 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3745 = math.fpowi %in, %c2_i32_571 : f32, i32
      linalg.yield %3745 : f32
    } -> tensor<1x40x4096xf32>
    %2465 = tosa.reduce_sum %2464 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %2466 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2467 = tosa.reciprocal %2466 : (tensor<1xf32>) -> tensor<1xf32>
    %2468 = tosa.mul %2467, %2465 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2469 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %2470 = tosa.add %2468, %2469 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2471 = tosa.rsqrt %2470 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2472 = tosa.mul %2462, %2471 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %2473 = tosa.reshape %arg192 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %2474 = tosa.mul %2473, %2472 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2475 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2476 = tosa.transpose %arg193, %2475 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %2477 = tosa.reshape %2474 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_572 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2478 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2477, %2476 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_572 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2479 = tosa.reshape %2478 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2480 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2481 = tosa.transpose %arg194, %2480 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %2482 = tosa.reshape %2474 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_573 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2483 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2482, %2481 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_573 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2484 = tosa.reshape %2483 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2485 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2486 = tosa.transpose %arg195, %2485 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %2487 = tosa.reshape %2474 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_574 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2488 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2487, %2486 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_574 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2489 = tosa.reshape %2488 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2490 = tosa.reshape %2479 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %2491 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2492 = tosa.transpose %2490, %2491 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %2493 = tosa.reshape %2484 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %2494 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2495 = tosa.transpose %2493, %2494 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %2496 = tosa.reshape %2489 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %2497 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2498 = tosa.transpose %2496, %2497 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %2499 = tosa.reshape %45 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %2500 = tosa.reshape %47 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %2501 = tosa.mul %2492, %2499 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_575 = tensor.extract_slice %2492[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_576 = tensor.extract_slice %2492[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %2502 = tensor.empty() : tensor<1x32x40x64xf32>
    %2503 = linalg.negf ins(%extracted_slice_576 : tensor<1x32x40x64xf32>) outs(%2502 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %2504 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_577 = tensor.insert_slice %2503 into %2504[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_578 = tensor.insert_slice %extracted_slice_575 into %inserted_slice_577[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %2505 = tosa.mul %inserted_slice_578, %2500 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2506 = tosa.add %2501, %2505 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2507 = tosa.mul %2495, %2499 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_579 = tensor.extract_slice %2495[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_580 = tensor.extract_slice %2495[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %2508 = tensor.empty() : tensor<1x32x40x64xf32>
    %2509 = linalg.negf ins(%extracted_slice_580 : tensor<1x32x40x64xf32>) outs(%2508 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %2510 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_581 = tensor.insert_slice %2509 into %2510[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_582 = tensor.insert_slice %extracted_slice_579 into %inserted_slice_581[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %2511 = tosa.mul %inserted_slice_582, %2500 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2512 = tosa.add %2507, %2511 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2513 = tosa.reshape %19 {new_shape = array<i64: 1, 40, 41>} : (tensor<40x41xf32>) -> tensor<1x40x41xf32>
    %2514 = tosa.reshape %2513 {new_shape = array<i64: 1, 1, 40, 41>} : (tensor<1x40x41xf32>) -> tensor<1x1x40x41xf32>
    %extracted_slice_583 = tensor.extract_slice %2514[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_584 = tensor.extract_slice %extracted_slice_583[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %2515 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x1x40x41xf32>}> : () -> tensor<1x1x40x41xf32>
    %2516 = tosa.add %extracted_slice_584, %2515 : (tensor<1x1x40x41xf32>, tensor<1x1x40x41xf32>) -> tensor<1x1x40x41xf32>
    %extracted_slice_585 = tensor.extract_slice %2516[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_586 = tensor.extract_slice %extracted_slice_585[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_587 = tensor.extract_slice %extracted_slice_586[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_588 = tensor.extract_slice %extracted_slice_587[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x40xf32>
    %cst_589 = arith.constant 0.000000e+00 : f32
    %splat_590 = tensor.splat %cst_589 : tensor<40x40xf32>
    %2517 = tosa.reshape %extracted_slice_588 {new_shape = array<i64: 40, 40>} : (tensor<1x1x40x40xf32>) -> tensor<40x40xf32>
    %2518 = tosa.add %splat_590, %2517 : (tensor<40x40xf32>, tensor<40x40xf32>) -> tensor<40x40xf32>
    %2519 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2520 = tosa.transpose %2512, %2519 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %2521 = tosa.reshape %2506 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %2522 = tosa.reshape %2520 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %2523 = tosa.matmul %2521, %2522 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %cst_591 = arith.constant 0.0883883461 : f32
    %splat_592 = tensor.splat %cst_591 : tensor<32x40x40xf32>
    %2524 = tosa.mul %2523, %splat_592 {shift = 0 : i8} : (tensor<32x40x40xf32>, tensor<32x40x40xf32>) -> tensor<32x40x40xf32>
    %2525 = tosa.add %2524, %2518 : (tensor<32x40x40xf32>, tensor<40x40xf32>) -> tensor<32x40x40xf32>
    %2526 = tosa.reduce_max %2525 {axis = 2 : i32} : (tensor<32x40x40xf32>) -> tensor<32x40x1xf32>
    %2527 = tosa.sub %2525, %2526 : (tensor<32x40x40xf32>, tensor<32x40x1xf32>) -> tensor<32x40x40xf32>
    %2528 = math.exp %2527 : tensor<32x40x40xf32>
    %2529 = tosa.reduce_sum %2528 {axis = 2 : i32} : (tensor<32x40x40xf32>) -> tensor<32x40x1xf32>
    %2530 = tosa.log %2529 : (tensor<32x40x1xf32>) -> tensor<32x40x1xf32>
    %2531 = tosa.add %2526, %2530 : (tensor<32x40x1xf32>, tensor<32x40x1xf32>) -> tensor<32x40x1xf32>
    %2532 = tosa.sub %2525, %2531 : (tensor<32x40x40xf32>, tensor<32x40x1xf32>) -> tensor<32x40x40xf32>
    %2533 = math.exp %2532 : tensor<32x40x40xf32>
    %2534 = tosa.reshape %2531 {new_shape = array<i64: 1, 32, 40>} : (tensor<32x40x1xf32>) -> tensor<1x32x40xf32>
    %2535 = tosa.reshape %2498 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %2536 = tosa.matmul %2533, %2535 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %2537 = tosa.reshape %2536 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2538 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2539 = tosa.transpose %2537, %2538 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %2540 = tosa.reshape %2539 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %2541 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2542 = tosa.transpose %arg196, %2541 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %2543 = tosa.reshape %2540 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_593 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2544 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2543, %2542 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_593 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2545 = tosa.reshape %2544 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2546 = tosa.add %2462, %2545 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2547 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_594 = arith.constant 2 : i32
    %2548 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2546 : tensor<1x40x4096xf32>) outs(%2547 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3745 = math.fpowi %in, %c2_i32_594 : f32, i32
      linalg.yield %3745 : f32
    } -> tensor<1x40x4096xf32>
    %2549 = tosa.reduce_sum %2548 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %2550 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2551 = tosa.reciprocal %2550 : (tensor<1xf32>) -> tensor<1xf32>
    %2552 = tosa.mul %2551, %2549 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2553 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %2554 = tosa.add %2552, %2553 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2555 = tosa.rsqrt %2554 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2556 = tosa.mul %2546, %2555 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %2557 = tosa.reshape %arg197 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %2558 = tosa.mul %2557, %2556 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2559 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2560 = tosa.transpose %arg198, %2559 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %2561 = tosa.reshape %2558 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_595 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %2562 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2561, %2560 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_595 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %2563 = tosa.reshape %2562 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %2564 = tosa.sigmoid %2563 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %2565 = tosa.mul %2563, %2564 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %2566 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2567 = tosa.transpose %arg199, %2566 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %2568 = tosa.reshape %2558 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_596 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %2569 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2568, %2567 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_596 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %2570 = tosa.reshape %2569 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %2571 = tosa.mul %2565, %2570 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %2572 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2573 = tosa.transpose %arg200, %2572 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %2574 = tosa.reshape %2571 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_597 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2575 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2574, %2573 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_597 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2576 = tosa.reshape %2575 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2577 = tosa.add %2546, %2576 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2578 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_598 = arith.constant 2 : i32
    %2579 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2577 : tensor<1x40x4096xf32>) outs(%2578 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3745 = math.fpowi %in, %c2_i32_598 : f32, i32
      linalg.yield %3745 : f32
    } -> tensor<1x40x4096xf32>
    %2580 = tosa.reduce_sum %2579 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %2581 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2582 = tosa.reciprocal %2581 : (tensor<1xf32>) -> tensor<1xf32>
    %2583 = tosa.mul %2582, %2580 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2584 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %2585 = tosa.add %2583, %2584 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2586 = tosa.rsqrt %2585 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2587 = tosa.mul %2577, %2586 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %2588 = tosa.reshape %arg201 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %2589 = tosa.mul %2588, %2587 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2590 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2591 = tosa.transpose %arg202, %2590 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %2592 = tosa.reshape %2589 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_599 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2593 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2592, %2591 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_599 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2594 = tosa.reshape %2593 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2595 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2596 = tosa.transpose %arg203, %2595 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %2597 = tosa.reshape %2589 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_600 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2598 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2597, %2596 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_600 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2599 = tosa.reshape %2598 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2600 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2601 = tosa.transpose %arg204, %2600 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %2602 = tosa.reshape %2589 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_601 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2603 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2602, %2601 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_601 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2604 = tosa.reshape %2603 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2605 = tosa.reshape %2594 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %2606 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2607 = tosa.transpose %2605, %2606 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %2608 = tosa.reshape %2599 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %2609 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2610 = tosa.transpose %2608, %2609 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %2611 = tosa.reshape %2604 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %2612 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2613 = tosa.transpose %2611, %2612 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %2614 = tosa.reshape %45 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %2615 = tosa.reshape %47 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %2616 = tosa.mul %2607, %2614 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_602 = tensor.extract_slice %2607[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_603 = tensor.extract_slice %2607[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %2617 = tensor.empty() : tensor<1x32x40x64xf32>
    %2618 = linalg.negf ins(%extracted_slice_603 : tensor<1x32x40x64xf32>) outs(%2617 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %2619 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_604 = tensor.insert_slice %2618 into %2619[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_605 = tensor.insert_slice %extracted_slice_602 into %inserted_slice_604[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %2620 = tosa.mul %inserted_slice_605, %2615 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2621 = tosa.add %2616, %2620 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2622 = tosa.mul %2610, %2614 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_606 = tensor.extract_slice %2610[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_607 = tensor.extract_slice %2610[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %2623 = tensor.empty() : tensor<1x32x40x64xf32>
    %2624 = linalg.negf ins(%extracted_slice_607 : tensor<1x32x40x64xf32>) outs(%2623 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %2625 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_608 = tensor.insert_slice %2624 into %2625[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_609 = tensor.insert_slice %extracted_slice_606 into %inserted_slice_608[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %2626 = tosa.mul %inserted_slice_609, %2615 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2627 = tosa.add %2622, %2626 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2628 = tosa.reshape %19 {new_shape = array<i64: 1, 40, 41>} : (tensor<40x41xf32>) -> tensor<1x40x41xf32>
    %2629 = tosa.reshape %2628 {new_shape = array<i64: 1, 1, 40, 41>} : (tensor<1x40x41xf32>) -> tensor<1x1x40x41xf32>
    %extracted_slice_610 = tensor.extract_slice %2629[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_611 = tensor.extract_slice %extracted_slice_610[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %2630 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x1x40x41xf32>}> : () -> tensor<1x1x40x41xf32>
    %2631 = tosa.add %extracted_slice_611, %2630 : (tensor<1x1x40x41xf32>, tensor<1x1x40x41xf32>) -> tensor<1x1x40x41xf32>
    %extracted_slice_612 = tensor.extract_slice %2631[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_613 = tensor.extract_slice %extracted_slice_612[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_614 = tensor.extract_slice %extracted_slice_613[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_615 = tensor.extract_slice %extracted_slice_614[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x40xf32>
    %cst_616 = arith.constant 0.000000e+00 : f32
    %splat_617 = tensor.splat %cst_616 : tensor<40x40xf32>
    %2632 = tosa.reshape %extracted_slice_615 {new_shape = array<i64: 40, 40>} : (tensor<1x1x40x40xf32>) -> tensor<40x40xf32>
    %2633 = tosa.add %splat_617, %2632 : (tensor<40x40xf32>, tensor<40x40xf32>) -> tensor<40x40xf32>
    %2634 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2635 = tosa.transpose %2627, %2634 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %2636 = tosa.reshape %2621 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %2637 = tosa.reshape %2635 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %2638 = tosa.matmul %2636, %2637 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %cst_618 = arith.constant 0.0883883461 : f32
    %splat_619 = tensor.splat %cst_618 : tensor<32x40x40xf32>
    %2639 = tosa.mul %2638, %splat_619 {shift = 0 : i8} : (tensor<32x40x40xf32>, tensor<32x40x40xf32>) -> tensor<32x40x40xf32>
    %2640 = tosa.add %2639, %2633 : (tensor<32x40x40xf32>, tensor<40x40xf32>) -> tensor<32x40x40xf32>
    %2641 = tosa.reduce_max %2640 {axis = 2 : i32} : (tensor<32x40x40xf32>) -> tensor<32x40x1xf32>
    %2642 = tosa.sub %2640, %2641 : (tensor<32x40x40xf32>, tensor<32x40x1xf32>) -> tensor<32x40x40xf32>
    %2643 = math.exp %2642 : tensor<32x40x40xf32>
    %2644 = tosa.reduce_sum %2643 {axis = 2 : i32} : (tensor<32x40x40xf32>) -> tensor<32x40x1xf32>
    %2645 = tosa.log %2644 : (tensor<32x40x1xf32>) -> tensor<32x40x1xf32>
    %2646 = tosa.add %2641, %2645 : (tensor<32x40x1xf32>, tensor<32x40x1xf32>) -> tensor<32x40x1xf32>
    %2647 = tosa.sub %2640, %2646 : (tensor<32x40x40xf32>, tensor<32x40x1xf32>) -> tensor<32x40x40xf32>
    %2648 = math.exp %2647 : tensor<32x40x40xf32>
    %2649 = tosa.reshape %2646 {new_shape = array<i64: 1, 32, 40>} : (tensor<32x40x1xf32>) -> tensor<1x32x40xf32>
    %2650 = tosa.reshape %2613 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %2651 = tosa.matmul %2648, %2650 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %2652 = tosa.reshape %2651 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2653 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2654 = tosa.transpose %2652, %2653 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %2655 = tosa.reshape %2654 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %2656 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2657 = tosa.transpose %arg205, %2656 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %2658 = tosa.reshape %2655 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_620 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2659 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2658, %2657 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_620 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2660 = tosa.reshape %2659 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2661 = tosa.add %2577, %2660 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2662 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_621 = arith.constant 2 : i32
    %2663 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2661 : tensor<1x40x4096xf32>) outs(%2662 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3745 = math.fpowi %in, %c2_i32_621 : f32, i32
      linalg.yield %3745 : f32
    } -> tensor<1x40x4096xf32>
    %2664 = tosa.reduce_sum %2663 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %2665 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2666 = tosa.reciprocal %2665 : (tensor<1xf32>) -> tensor<1xf32>
    %2667 = tosa.mul %2666, %2664 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2668 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %2669 = tosa.add %2667, %2668 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2670 = tosa.rsqrt %2669 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2671 = tosa.mul %2661, %2670 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %2672 = tosa.reshape %arg206 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %2673 = tosa.mul %2672, %2671 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2674 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2675 = tosa.transpose %arg207, %2674 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %2676 = tosa.reshape %2673 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_622 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %2677 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2676, %2675 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_622 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %2678 = tosa.reshape %2677 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %2679 = tosa.sigmoid %2678 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %2680 = tosa.mul %2678, %2679 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %2681 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2682 = tosa.transpose %arg208, %2681 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %2683 = tosa.reshape %2673 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_623 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %2684 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2683, %2682 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_623 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %2685 = tosa.reshape %2684 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %2686 = tosa.mul %2680, %2685 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %2687 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2688 = tosa.transpose %arg209, %2687 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %2689 = tosa.reshape %2686 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_624 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2690 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2689, %2688 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_624 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2691 = tosa.reshape %2690 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2692 = tosa.add %2661, %2691 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2693 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_625 = arith.constant 2 : i32
    %2694 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2692 : tensor<1x40x4096xf32>) outs(%2693 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3745 = math.fpowi %in, %c2_i32_625 : f32, i32
      linalg.yield %3745 : f32
    } -> tensor<1x40x4096xf32>
    %2695 = tosa.reduce_sum %2694 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %2696 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2697 = tosa.reciprocal %2696 : (tensor<1xf32>) -> tensor<1xf32>
    %2698 = tosa.mul %2697, %2695 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2699 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %2700 = tosa.add %2698, %2699 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2701 = tosa.rsqrt %2700 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2702 = tosa.mul %2692, %2701 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %2703 = tosa.reshape %arg210 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %2704 = tosa.mul %2703, %2702 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2705 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2706 = tosa.transpose %arg211, %2705 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %2707 = tosa.reshape %2704 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_626 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2708 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2707, %2706 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_626 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2709 = tosa.reshape %2708 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2710 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2711 = tosa.transpose %arg212, %2710 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %2712 = tosa.reshape %2704 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_627 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2713 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2712, %2711 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_627 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2714 = tosa.reshape %2713 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2715 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2716 = tosa.transpose %arg213, %2715 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %2717 = tosa.reshape %2704 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_628 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2718 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2717, %2716 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_628 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2719 = tosa.reshape %2718 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2720 = tosa.reshape %2709 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %2721 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2722 = tosa.transpose %2720, %2721 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %2723 = tosa.reshape %2714 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %2724 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2725 = tosa.transpose %2723, %2724 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %2726 = tosa.reshape %2719 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %2727 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2728 = tosa.transpose %2726, %2727 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %2729 = tosa.reshape %45 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %2730 = tosa.reshape %47 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %2731 = tosa.mul %2722, %2729 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_629 = tensor.extract_slice %2722[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_630 = tensor.extract_slice %2722[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %2732 = tensor.empty() : tensor<1x32x40x64xf32>
    %2733 = linalg.negf ins(%extracted_slice_630 : tensor<1x32x40x64xf32>) outs(%2732 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %2734 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_631 = tensor.insert_slice %2733 into %2734[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_632 = tensor.insert_slice %extracted_slice_629 into %inserted_slice_631[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %2735 = tosa.mul %inserted_slice_632, %2730 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2736 = tosa.add %2731, %2735 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2737 = tosa.mul %2725, %2729 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_633 = tensor.extract_slice %2725[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_634 = tensor.extract_slice %2725[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %2738 = tensor.empty() : tensor<1x32x40x64xf32>
    %2739 = linalg.negf ins(%extracted_slice_634 : tensor<1x32x40x64xf32>) outs(%2738 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %2740 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_635 = tensor.insert_slice %2739 into %2740[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_636 = tensor.insert_slice %extracted_slice_633 into %inserted_slice_635[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %2741 = tosa.mul %inserted_slice_636, %2730 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2742 = tosa.add %2737, %2741 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2743 = tosa.reshape %19 {new_shape = array<i64: 1, 40, 41>} : (tensor<40x41xf32>) -> tensor<1x40x41xf32>
    %2744 = tosa.reshape %2743 {new_shape = array<i64: 1, 1, 40, 41>} : (tensor<1x40x41xf32>) -> tensor<1x1x40x41xf32>
    %extracted_slice_637 = tensor.extract_slice %2744[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_638 = tensor.extract_slice %extracted_slice_637[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %2745 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x1x40x41xf32>}> : () -> tensor<1x1x40x41xf32>
    %2746 = tosa.add %extracted_slice_638, %2745 : (tensor<1x1x40x41xf32>, tensor<1x1x40x41xf32>) -> tensor<1x1x40x41xf32>
    %extracted_slice_639 = tensor.extract_slice %2746[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_640 = tensor.extract_slice %extracted_slice_639[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_641 = tensor.extract_slice %extracted_slice_640[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_642 = tensor.extract_slice %extracted_slice_641[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x40xf32>
    %cst_643 = arith.constant 0.000000e+00 : f32
    %splat_644 = tensor.splat %cst_643 : tensor<40x40xf32>
    %2747 = tosa.reshape %extracted_slice_642 {new_shape = array<i64: 40, 40>} : (tensor<1x1x40x40xf32>) -> tensor<40x40xf32>
    %2748 = tosa.add %splat_644, %2747 : (tensor<40x40xf32>, tensor<40x40xf32>) -> tensor<40x40xf32>
    %2749 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2750 = tosa.transpose %2742, %2749 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %2751 = tosa.reshape %2736 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %2752 = tosa.reshape %2750 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %2753 = tosa.matmul %2751, %2752 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %cst_645 = arith.constant 0.0883883461 : f32
    %splat_646 = tensor.splat %cst_645 : tensor<32x40x40xf32>
    %2754 = tosa.mul %2753, %splat_646 {shift = 0 : i8} : (tensor<32x40x40xf32>, tensor<32x40x40xf32>) -> tensor<32x40x40xf32>
    %2755 = tosa.add %2754, %2748 : (tensor<32x40x40xf32>, tensor<40x40xf32>) -> tensor<32x40x40xf32>
    %2756 = tosa.reduce_max %2755 {axis = 2 : i32} : (tensor<32x40x40xf32>) -> tensor<32x40x1xf32>
    %2757 = tosa.sub %2755, %2756 : (tensor<32x40x40xf32>, tensor<32x40x1xf32>) -> tensor<32x40x40xf32>
    %2758 = math.exp %2757 : tensor<32x40x40xf32>
    %2759 = tosa.reduce_sum %2758 {axis = 2 : i32} : (tensor<32x40x40xf32>) -> tensor<32x40x1xf32>
    %2760 = tosa.log %2759 : (tensor<32x40x1xf32>) -> tensor<32x40x1xf32>
    %2761 = tosa.add %2756, %2760 : (tensor<32x40x1xf32>, tensor<32x40x1xf32>) -> tensor<32x40x1xf32>
    %2762 = tosa.sub %2755, %2761 : (tensor<32x40x40xf32>, tensor<32x40x1xf32>) -> tensor<32x40x40xf32>
    %2763 = math.exp %2762 : tensor<32x40x40xf32>
    %2764 = tosa.reshape %2761 {new_shape = array<i64: 1, 32, 40>} : (tensor<32x40x1xf32>) -> tensor<1x32x40xf32>
    %2765 = tosa.reshape %2728 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %2766 = tosa.matmul %2763, %2765 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %2767 = tosa.reshape %2766 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2768 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2769 = tosa.transpose %2767, %2768 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %2770 = tosa.reshape %2769 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %2771 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2772 = tosa.transpose %arg214, %2771 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %2773 = tosa.reshape %2770 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_647 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2774 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2773, %2772 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_647 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2775 = tosa.reshape %2774 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2776 = tosa.add %2692, %2775 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2777 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_648 = arith.constant 2 : i32
    %2778 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2776 : tensor<1x40x4096xf32>) outs(%2777 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3745 = math.fpowi %in, %c2_i32_648 : f32, i32
      linalg.yield %3745 : f32
    } -> tensor<1x40x4096xf32>
    %2779 = tosa.reduce_sum %2778 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %2780 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2781 = tosa.reciprocal %2780 : (tensor<1xf32>) -> tensor<1xf32>
    %2782 = tosa.mul %2781, %2779 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2783 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %2784 = tosa.add %2782, %2783 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2785 = tosa.rsqrt %2784 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2786 = tosa.mul %2776, %2785 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %2787 = tosa.reshape %arg215 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %2788 = tosa.mul %2787, %2786 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2789 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2790 = tosa.transpose %arg216, %2789 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %2791 = tosa.reshape %2788 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_649 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %2792 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2791, %2790 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_649 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %2793 = tosa.reshape %2792 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %2794 = tosa.sigmoid %2793 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %2795 = tosa.mul %2793, %2794 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %2796 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2797 = tosa.transpose %arg217, %2796 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %2798 = tosa.reshape %2788 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_650 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %2799 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2798, %2797 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_650 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %2800 = tosa.reshape %2799 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %2801 = tosa.mul %2795, %2800 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %2802 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2803 = tosa.transpose %arg218, %2802 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %2804 = tosa.reshape %2801 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_651 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2805 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2804, %2803 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_651 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2806 = tosa.reshape %2805 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2807 = tosa.add %2776, %2806 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2808 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_652 = arith.constant 2 : i32
    %2809 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2807 : tensor<1x40x4096xf32>) outs(%2808 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3745 = math.fpowi %in, %c2_i32_652 : f32, i32
      linalg.yield %3745 : f32
    } -> tensor<1x40x4096xf32>
    %2810 = tosa.reduce_sum %2809 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %2811 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2812 = tosa.reciprocal %2811 : (tensor<1xf32>) -> tensor<1xf32>
    %2813 = tosa.mul %2812, %2810 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2814 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %2815 = tosa.add %2813, %2814 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2816 = tosa.rsqrt %2815 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2817 = tosa.mul %2807, %2816 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %2818 = tosa.reshape %arg219 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %2819 = tosa.mul %2818, %2817 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2820 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2821 = tosa.transpose %arg220, %2820 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %2822 = tosa.reshape %2819 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_653 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2823 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2822, %2821 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_653 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2824 = tosa.reshape %2823 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2825 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2826 = tosa.transpose %arg221, %2825 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %2827 = tosa.reshape %2819 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_654 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2828 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2827, %2826 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_654 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2829 = tosa.reshape %2828 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2830 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2831 = tosa.transpose %arg222, %2830 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %2832 = tosa.reshape %2819 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_655 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2833 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2832, %2831 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_655 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2834 = tosa.reshape %2833 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2835 = tosa.reshape %2824 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %2836 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2837 = tosa.transpose %2835, %2836 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %2838 = tosa.reshape %2829 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %2839 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2840 = tosa.transpose %2838, %2839 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %2841 = tosa.reshape %2834 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %2842 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2843 = tosa.transpose %2841, %2842 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %2844 = tosa.reshape %45 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %2845 = tosa.reshape %47 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %2846 = tosa.mul %2837, %2844 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_656 = tensor.extract_slice %2837[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_657 = tensor.extract_slice %2837[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %2847 = tensor.empty() : tensor<1x32x40x64xf32>
    %2848 = linalg.negf ins(%extracted_slice_657 : tensor<1x32x40x64xf32>) outs(%2847 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %2849 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_658 = tensor.insert_slice %2848 into %2849[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_659 = tensor.insert_slice %extracted_slice_656 into %inserted_slice_658[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %2850 = tosa.mul %inserted_slice_659, %2845 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2851 = tosa.add %2846, %2850 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2852 = tosa.mul %2840, %2844 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_660 = tensor.extract_slice %2840[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_661 = tensor.extract_slice %2840[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %2853 = tensor.empty() : tensor<1x32x40x64xf32>
    %2854 = linalg.negf ins(%extracted_slice_661 : tensor<1x32x40x64xf32>) outs(%2853 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %2855 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_662 = tensor.insert_slice %2854 into %2855[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_663 = tensor.insert_slice %extracted_slice_660 into %inserted_slice_662[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %2856 = tosa.mul %inserted_slice_663, %2845 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2857 = tosa.add %2852, %2856 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2858 = tosa.reshape %19 {new_shape = array<i64: 1, 40, 41>} : (tensor<40x41xf32>) -> tensor<1x40x41xf32>
    %2859 = tosa.reshape %2858 {new_shape = array<i64: 1, 1, 40, 41>} : (tensor<1x40x41xf32>) -> tensor<1x1x40x41xf32>
    %extracted_slice_664 = tensor.extract_slice %2859[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_665 = tensor.extract_slice %extracted_slice_664[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %2860 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x1x40x41xf32>}> : () -> tensor<1x1x40x41xf32>
    %2861 = tosa.add %extracted_slice_665, %2860 : (tensor<1x1x40x41xf32>, tensor<1x1x40x41xf32>) -> tensor<1x1x40x41xf32>
    %extracted_slice_666 = tensor.extract_slice %2861[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_667 = tensor.extract_slice %extracted_slice_666[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_668 = tensor.extract_slice %extracted_slice_667[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_669 = tensor.extract_slice %extracted_slice_668[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x40xf32>
    %cst_670 = arith.constant 0.000000e+00 : f32
    %splat_671 = tensor.splat %cst_670 : tensor<40x40xf32>
    %2862 = tosa.reshape %extracted_slice_669 {new_shape = array<i64: 40, 40>} : (tensor<1x1x40x40xf32>) -> tensor<40x40xf32>
    %2863 = tosa.add %splat_671, %2862 : (tensor<40x40xf32>, tensor<40x40xf32>) -> tensor<40x40xf32>
    %2864 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2865 = tosa.transpose %2857, %2864 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %2866 = tosa.reshape %2851 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %2867 = tosa.reshape %2865 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %2868 = tosa.matmul %2866, %2867 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %cst_672 = arith.constant 0.0883883461 : f32
    %splat_673 = tensor.splat %cst_672 : tensor<32x40x40xf32>
    %2869 = tosa.mul %2868, %splat_673 {shift = 0 : i8} : (tensor<32x40x40xf32>, tensor<32x40x40xf32>) -> tensor<32x40x40xf32>
    %2870 = tosa.add %2869, %2863 : (tensor<32x40x40xf32>, tensor<40x40xf32>) -> tensor<32x40x40xf32>
    %2871 = tosa.reduce_max %2870 {axis = 2 : i32} : (tensor<32x40x40xf32>) -> tensor<32x40x1xf32>
    %2872 = tosa.sub %2870, %2871 : (tensor<32x40x40xf32>, tensor<32x40x1xf32>) -> tensor<32x40x40xf32>
    %2873 = math.exp %2872 : tensor<32x40x40xf32>
    %2874 = tosa.reduce_sum %2873 {axis = 2 : i32} : (tensor<32x40x40xf32>) -> tensor<32x40x1xf32>
    %2875 = tosa.log %2874 : (tensor<32x40x1xf32>) -> tensor<32x40x1xf32>
    %2876 = tosa.add %2871, %2875 : (tensor<32x40x1xf32>, tensor<32x40x1xf32>) -> tensor<32x40x1xf32>
    %2877 = tosa.sub %2870, %2876 : (tensor<32x40x40xf32>, tensor<32x40x1xf32>) -> tensor<32x40x40xf32>
    %2878 = math.exp %2877 : tensor<32x40x40xf32>
    %2879 = tosa.reshape %2876 {new_shape = array<i64: 1, 32, 40>} : (tensor<32x40x1xf32>) -> tensor<1x32x40xf32>
    %2880 = tosa.reshape %2843 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %2881 = tosa.matmul %2878, %2880 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %2882 = tosa.reshape %2881 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2883 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2884 = tosa.transpose %2882, %2883 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %2885 = tosa.reshape %2884 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %2886 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2887 = tosa.transpose %arg223, %2886 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %2888 = tosa.reshape %2885 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_674 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2889 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2888, %2887 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_674 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2890 = tosa.reshape %2889 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2891 = tosa.add %2807, %2890 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2892 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_675 = arith.constant 2 : i32
    %2893 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2891 : tensor<1x40x4096xf32>) outs(%2892 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3745 = math.fpowi %in, %c2_i32_675 : f32, i32
      linalg.yield %3745 : f32
    } -> tensor<1x40x4096xf32>
    %2894 = tosa.reduce_sum %2893 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %2895 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2896 = tosa.reciprocal %2895 : (tensor<1xf32>) -> tensor<1xf32>
    %2897 = tosa.mul %2896, %2894 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2898 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %2899 = tosa.add %2897, %2898 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2900 = tosa.rsqrt %2899 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2901 = tosa.mul %2891, %2900 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %2902 = tosa.reshape %arg224 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %2903 = tosa.mul %2902, %2901 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2904 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2905 = tosa.transpose %arg225, %2904 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %2906 = tosa.reshape %2903 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_676 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %2907 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2906, %2905 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_676 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %2908 = tosa.reshape %2907 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %2909 = tosa.sigmoid %2908 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %2910 = tosa.mul %2908, %2909 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %2911 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2912 = tosa.transpose %arg226, %2911 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %2913 = tosa.reshape %2903 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_677 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %2914 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2913, %2912 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_677 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %2915 = tosa.reshape %2914 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %2916 = tosa.mul %2910, %2915 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %2917 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2918 = tosa.transpose %arg227, %2917 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %2919 = tosa.reshape %2916 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_678 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2920 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2919, %2918 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_678 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2921 = tosa.reshape %2920 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2922 = tosa.add %2891, %2921 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2923 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_679 = arith.constant 2 : i32
    %2924 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2922 : tensor<1x40x4096xf32>) outs(%2923 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3745 = math.fpowi %in, %c2_i32_679 : f32, i32
      linalg.yield %3745 : f32
    } -> tensor<1x40x4096xf32>
    %2925 = tosa.reduce_sum %2924 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %2926 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2927 = tosa.reciprocal %2926 : (tensor<1xf32>) -> tensor<1xf32>
    %2928 = tosa.mul %2927, %2925 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2929 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %2930 = tosa.add %2928, %2929 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2931 = tosa.rsqrt %2930 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2932 = tosa.mul %2922, %2931 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %2933 = tosa.reshape %arg228 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %2934 = tosa.mul %2933, %2932 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2935 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2936 = tosa.transpose %arg229, %2935 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %2937 = tosa.reshape %2934 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_680 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2938 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2937, %2936 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_680 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2939 = tosa.reshape %2938 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2940 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2941 = tosa.transpose %arg230, %2940 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %2942 = tosa.reshape %2934 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_681 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2943 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2942, %2941 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_681 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2944 = tosa.reshape %2943 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2945 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2946 = tosa.transpose %arg231, %2945 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %2947 = tosa.reshape %2934 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_682 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2948 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2947, %2946 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_682 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2949 = tosa.reshape %2948 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2950 = tosa.reshape %2939 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %2951 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2952 = tosa.transpose %2950, %2951 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %2953 = tosa.reshape %2944 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %2954 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2955 = tosa.transpose %2953, %2954 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %2956 = tosa.reshape %2949 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %2957 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2958 = tosa.transpose %2956, %2957 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %2959 = tosa.reshape %45 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %2960 = tosa.reshape %47 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %2961 = tosa.mul %2952, %2959 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_683 = tensor.extract_slice %2952[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_684 = tensor.extract_slice %2952[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %2962 = tensor.empty() : tensor<1x32x40x64xf32>
    %2963 = linalg.negf ins(%extracted_slice_684 : tensor<1x32x40x64xf32>) outs(%2962 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %2964 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_685 = tensor.insert_slice %2963 into %2964[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_686 = tensor.insert_slice %extracted_slice_683 into %inserted_slice_685[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %2965 = tosa.mul %inserted_slice_686, %2960 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2966 = tosa.add %2961, %2965 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2967 = tosa.mul %2955, %2959 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_687 = tensor.extract_slice %2955[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_688 = tensor.extract_slice %2955[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %2968 = tensor.empty() : tensor<1x32x40x64xf32>
    %2969 = linalg.negf ins(%extracted_slice_688 : tensor<1x32x40x64xf32>) outs(%2968 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %2970 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_689 = tensor.insert_slice %2969 into %2970[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_690 = tensor.insert_slice %extracted_slice_687 into %inserted_slice_689[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %2971 = tosa.mul %inserted_slice_690, %2960 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2972 = tosa.add %2967, %2971 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2973 = tosa.reshape %19 {new_shape = array<i64: 1, 40, 41>} : (tensor<40x41xf32>) -> tensor<1x40x41xf32>
    %2974 = tosa.reshape %2973 {new_shape = array<i64: 1, 1, 40, 41>} : (tensor<1x40x41xf32>) -> tensor<1x1x40x41xf32>
    %extracted_slice_691 = tensor.extract_slice %2974[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_692 = tensor.extract_slice %extracted_slice_691[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %2975 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x1x40x41xf32>}> : () -> tensor<1x1x40x41xf32>
    %2976 = tosa.add %extracted_slice_692, %2975 : (tensor<1x1x40x41xf32>, tensor<1x1x40x41xf32>) -> tensor<1x1x40x41xf32>
    %extracted_slice_693 = tensor.extract_slice %2976[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_694 = tensor.extract_slice %extracted_slice_693[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_695 = tensor.extract_slice %extracted_slice_694[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_696 = tensor.extract_slice %extracted_slice_695[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x40xf32>
    %cst_697 = arith.constant 0.000000e+00 : f32
    %splat_698 = tensor.splat %cst_697 : tensor<40x40xf32>
    %2977 = tosa.reshape %extracted_slice_696 {new_shape = array<i64: 40, 40>} : (tensor<1x1x40x40xf32>) -> tensor<40x40xf32>
    %2978 = tosa.add %splat_698, %2977 : (tensor<40x40xf32>, tensor<40x40xf32>) -> tensor<40x40xf32>
    %2979 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2980 = tosa.transpose %2972, %2979 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %2981 = tosa.reshape %2966 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %2982 = tosa.reshape %2980 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %2983 = tosa.matmul %2981, %2982 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %cst_699 = arith.constant 0.0883883461 : f32
    %splat_700 = tensor.splat %cst_699 : tensor<32x40x40xf32>
    %2984 = tosa.mul %2983, %splat_700 {shift = 0 : i8} : (tensor<32x40x40xf32>, tensor<32x40x40xf32>) -> tensor<32x40x40xf32>
    %2985 = tosa.add %2984, %2978 : (tensor<32x40x40xf32>, tensor<40x40xf32>) -> tensor<32x40x40xf32>
    %2986 = tosa.reduce_max %2985 {axis = 2 : i32} : (tensor<32x40x40xf32>) -> tensor<32x40x1xf32>
    %2987 = tosa.sub %2985, %2986 : (tensor<32x40x40xf32>, tensor<32x40x1xf32>) -> tensor<32x40x40xf32>
    %2988 = math.exp %2987 : tensor<32x40x40xf32>
    %2989 = tosa.reduce_sum %2988 {axis = 2 : i32} : (tensor<32x40x40xf32>) -> tensor<32x40x1xf32>
    %2990 = tosa.log %2989 : (tensor<32x40x1xf32>) -> tensor<32x40x1xf32>
    %2991 = tosa.add %2986, %2990 : (tensor<32x40x1xf32>, tensor<32x40x1xf32>) -> tensor<32x40x1xf32>
    %2992 = tosa.sub %2985, %2991 : (tensor<32x40x40xf32>, tensor<32x40x1xf32>) -> tensor<32x40x40xf32>
    %2993 = math.exp %2992 : tensor<32x40x40xf32>
    %2994 = tosa.reshape %2991 {new_shape = array<i64: 1, 32, 40>} : (tensor<32x40x1xf32>) -> tensor<1x32x40xf32>
    %2995 = tosa.reshape %2958 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %2996 = tosa.matmul %2993, %2995 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %2997 = tosa.reshape %2996 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2998 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2999 = tosa.transpose %2997, %2998 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %3000 = tosa.reshape %2999 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %3001 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3002 = tosa.transpose %arg232, %3001 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %3003 = tosa.reshape %3000 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_701 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3004 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3003, %3002 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_701 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3005 = tosa.reshape %3004 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3006 = tosa.add %2922, %3005 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %3007 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_702 = arith.constant 2 : i32
    %3008 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3006 : tensor<1x40x4096xf32>) outs(%3007 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3745 = math.fpowi %in, %c2_i32_702 : f32, i32
      linalg.yield %3745 : f32
    } -> tensor<1x40x4096xf32>
    %3009 = tosa.reduce_sum %3008 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %3010 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3011 = tosa.reciprocal %3010 : (tensor<1xf32>) -> tensor<1xf32>
    %3012 = tosa.mul %3011, %3009 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3013 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %3014 = tosa.add %3012, %3013 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3015 = tosa.rsqrt %3014 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3016 = tosa.mul %3006, %3015 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %3017 = tosa.reshape %arg233 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %3018 = tosa.mul %3017, %3016 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %3019 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3020 = tosa.transpose %arg234, %3019 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %3021 = tosa.reshape %3018 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_703 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %3022 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3021, %3020 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_703 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %3023 = tosa.reshape %3022 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %3024 = tosa.sigmoid %3023 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %3025 = tosa.mul %3023, %3024 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %3026 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3027 = tosa.transpose %arg235, %3026 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %3028 = tosa.reshape %3018 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_704 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %3029 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3028, %3027 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_704 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %3030 = tosa.reshape %3029 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %3031 = tosa.mul %3025, %3030 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %3032 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3033 = tosa.transpose %arg236, %3032 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %3034 = tosa.reshape %3031 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_705 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3035 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3034, %3033 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_705 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3036 = tosa.reshape %3035 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3037 = tosa.add %3006, %3036 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %3038 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_706 = arith.constant 2 : i32
    %3039 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3037 : tensor<1x40x4096xf32>) outs(%3038 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3745 = math.fpowi %in, %c2_i32_706 : f32, i32
      linalg.yield %3745 : f32
    } -> tensor<1x40x4096xf32>
    %3040 = tosa.reduce_sum %3039 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %3041 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3042 = tosa.reciprocal %3041 : (tensor<1xf32>) -> tensor<1xf32>
    %3043 = tosa.mul %3042, %3040 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3044 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %3045 = tosa.add %3043, %3044 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3046 = tosa.rsqrt %3045 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3047 = tosa.mul %3037, %3046 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %3048 = tosa.reshape %arg237 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %3049 = tosa.mul %3048, %3047 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %3050 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3051 = tosa.transpose %arg238, %3050 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %3052 = tosa.reshape %3049 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_707 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3053 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3052, %3051 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_707 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3054 = tosa.reshape %3053 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3055 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3056 = tosa.transpose %arg239, %3055 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %3057 = tosa.reshape %3049 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_708 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3058 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3057, %3056 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_708 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3059 = tosa.reshape %3058 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3060 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3061 = tosa.transpose %arg240, %3060 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %3062 = tosa.reshape %3049 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_709 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3063 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3062, %3061 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_709 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3064 = tosa.reshape %3063 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3065 = tosa.reshape %3054 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %3066 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3067 = tosa.transpose %3065, %3066 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %3068 = tosa.reshape %3059 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %3069 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3070 = tosa.transpose %3068, %3069 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %3071 = tosa.reshape %3064 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %3072 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3073 = tosa.transpose %3071, %3072 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %3074 = tosa.reshape %45 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %3075 = tosa.reshape %47 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %3076 = tosa.mul %3067, %3074 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_710 = tensor.extract_slice %3067[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_711 = tensor.extract_slice %3067[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %3077 = tensor.empty() : tensor<1x32x40x64xf32>
    %3078 = linalg.negf ins(%extracted_slice_711 : tensor<1x32x40x64xf32>) outs(%3077 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %3079 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_712 = tensor.insert_slice %3078 into %3079[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_713 = tensor.insert_slice %extracted_slice_710 into %inserted_slice_712[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %3080 = tosa.mul %inserted_slice_713, %3075 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3081 = tosa.add %3076, %3080 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3082 = tosa.mul %3070, %3074 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_714 = tensor.extract_slice %3070[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_715 = tensor.extract_slice %3070[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %3083 = tensor.empty() : tensor<1x32x40x64xf32>
    %3084 = linalg.negf ins(%extracted_slice_715 : tensor<1x32x40x64xf32>) outs(%3083 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %3085 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_716 = tensor.insert_slice %3084 into %3085[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_717 = tensor.insert_slice %extracted_slice_714 into %inserted_slice_716[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %3086 = tosa.mul %inserted_slice_717, %3075 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3087 = tosa.add %3082, %3086 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3088 = tosa.reshape %19 {new_shape = array<i64: 1, 40, 41>} : (tensor<40x41xf32>) -> tensor<1x40x41xf32>
    %3089 = tosa.reshape %3088 {new_shape = array<i64: 1, 1, 40, 41>} : (tensor<1x40x41xf32>) -> tensor<1x1x40x41xf32>
    %extracted_slice_718 = tensor.extract_slice %3089[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_719 = tensor.extract_slice %extracted_slice_718[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %3090 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x1x40x41xf32>}> : () -> tensor<1x1x40x41xf32>
    %3091 = tosa.add %extracted_slice_719, %3090 : (tensor<1x1x40x41xf32>, tensor<1x1x40x41xf32>) -> tensor<1x1x40x41xf32>
    %extracted_slice_720 = tensor.extract_slice %3091[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_721 = tensor.extract_slice %extracted_slice_720[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_722 = tensor.extract_slice %extracted_slice_721[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_723 = tensor.extract_slice %extracted_slice_722[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x40xf32>
    %cst_724 = arith.constant 0.000000e+00 : f32
    %splat_725 = tensor.splat %cst_724 : tensor<40x40xf32>
    %3092 = tosa.reshape %extracted_slice_723 {new_shape = array<i64: 40, 40>} : (tensor<1x1x40x40xf32>) -> tensor<40x40xf32>
    %3093 = tosa.add %splat_725, %3092 : (tensor<40x40xf32>, tensor<40x40xf32>) -> tensor<40x40xf32>
    %3094 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3095 = tosa.transpose %3087, %3094 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %3096 = tosa.reshape %3081 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %3097 = tosa.reshape %3095 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %3098 = tosa.matmul %3096, %3097 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %cst_726 = arith.constant 0.0883883461 : f32
    %splat_727 = tensor.splat %cst_726 : tensor<32x40x40xf32>
    %3099 = tosa.mul %3098, %splat_727 {shift = 0 : i8} : (tensor<32x40x40xf32>, tensor<32x40x40xf32>) -> tensor<32x40x40xf32>
    %3100 = tosa.add %3099, %3093 : (tensor<32x40x40xf32>, tensor<40x40xf32>) -> tensor<32x40x40xf32>
    %3101 = tosa.reduce_max %3100 {axis = 2 : i32} : (tensor<32x40x40xf32>) -> tensor<32x40x1xf32>
    %3102 = tosa.sub %3100, %3101 : (tensor<32x40x40xf32>, tensor<32x40x1xf32>) -> tensor<32x40x40xf32>
    %3103 = math.exp %3102 : tensor<32x40x40xf32>
    %3104 = tosa.reduce_sum %3103 {axis = 2 : i32} : (tensor<32x40x40xf32>) -> tensor<32x40x1xf32>
    %3105 = tosa.log %3104 : (tensor<32x40x1xf32>) -> tensor<32x40x1xf32>
    %3106 = tosa.add %3101, %3105 : (tensor<32x40x1xf32>, tensor<32x40x1xf32>) -> tensor<32x40x1xf32>
    %3107 = tosa.sub %3100, %3106 : (tensor<32x40x40xf32>, tensor<32x40x1xf32>) -> tensor<32x40x40xf32>
    %3108 = math.exp %3107 : tensor<32x40x40xf32>
    %3109 = tosa.reshape %3106 {new_shape = array<i64: 1, 32, 40>} : (tensor<32x40x1xf32>) -> tensor<1x32x40xf32>
    %3110 = tosa.reshape %3073 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %3111 = tosa.matmul %3108, %3110 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %3112 = tosa.reshape %3111 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3113 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3114 = tosa.transpose %3112, %3113 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %3115 = tosa.reshape %3114 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %3116 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3117 = tosa.transpose %arg241, %3116 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %3118 = tosa.reshape %3115 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_728 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3119 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3118, %3117 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_728 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3120 = tosa.reshape %3119 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3121 = tosa.add %3037, %3120 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %3122 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_729 = arith.constant 2 : i32
    %3123 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3121 : tensor<1x40x4096xf32>) outs(%3122 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3745 = math.fpowi %in, %c2_i32_729 : f32, i32
      linalg.yield %3745 : f32
    } -> tensor<1x40x4096xf32>
    %3124 = tosa.reduce_sum %3123 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %3125 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3126 = tosa.reciprocal %3125 : (tensor<1xf32>) -> tensor<1xf32>
    %3127 = tosa.mul %3126, %3124 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3128 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %3129 = tosa.add %3127, %3128 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3130 = tosa.rsqrt %3129 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3131 = tosa.mul %3121, %3130 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %3132 = tosa.reshape %arg242 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %3133 = tosa.mul %3132, %3131 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %3134 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3135 = tosa.transpose %arg243, %3134 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %3136 = tosa.reshape %3133 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_730 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %3137 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3136, %3135 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_730 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %3138 = tosa.reshape %3137 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %3139 = tosa.sigmoid %3138 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %3140 = tosa.mul %3138, %3139 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %3141 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3142 = tosa.transpose %arg244, %3141 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %3143 = tosa.reshape %3133 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_731 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %3144 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3143, %3142 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_731 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %3145 = tosa.reshape %3144 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %3146 = tosa.mul %3140, %3145 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %3147 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3148 = tosa.transpose %arg245, %3147 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %3149 = tosa.reshape %3146 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_732 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3150 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3149, %3148 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_732 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3151 = tosa.reshape %3150 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3152 = tosa.add %3121, %3151 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %3153 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_733 = arith.constant 2 : i32
    %3154 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3152 : tensor<1x40x4096xf32>) outs(%3153 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3745 = math.fpowi %in, %c2_i32_733 : f32, i32
      linalg.yield %3745 : f32
    } -> tensor<1x40x4096xf32>
    %3155 = tosa.reduce_sum %3154 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %3156 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3157 = tosa.reciprocal %3156 : (tensor<1xf32>) -> tensor<1xf32>
    %3158 = tosa.mul %3157, %3155 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3159 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %3160 = tosa.add %3158, %3159 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3161 = tosa.rsqrt %3160 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3162 = tosa.mul %3152, %3161 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %3163 = tosa.reshape %arg246 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %3164 = tosa.mul %3163, %3162 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %3165 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3166 = tosa.transpose %arg247, %3165 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %3167 = tosa.reshape %3164 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_734 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3168 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3167, %3166 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_734 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3169 = tosa.reshape %3168 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3170 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3171 = tosa.transpose %arg248, %3170 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %3172 = tosa.reshape %3164 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_735 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3173 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3172, %3171 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_735 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3174 = tosa.reshape %3173 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3175 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3176 = tosa.transpose %arg249, %3175 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %3177 = tosa.reshape %3164 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_736 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3178 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3177, %3176 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_736 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3179 = tosa.reshape %3178 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3180 = tosa.reshape %3169 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %3181 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3182 = tosa.transpose %3180, %3181 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %3183 = tosa.reshape %3174 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %3184 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3185 = tosa.transpose %3183, %3184 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %3186 = tosa.reshape %3179 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %3187 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3188 = tosa.transpose %3186, %3187 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %3189 = tosa.reshape %45 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %3190 = tosa.reshape %47 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %3191 = tosa.mul %3182, %3189 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_737 = tensor.extract_slice %3182[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_738 = tensor.extract_slice %3182[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %3192 = tensor.empty() : tensor<1x32x40x64xf32>
    %3193 = linalg.negf ins(%extracted_slice_738 : tensor<1x32x40x64xf32>) outs(%3192 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %3194 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_739 = tensor.insert_slice %3193 into %3194[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_740 = tensor.insert_slice %extracted_slice_737 into %inserted_slice_739[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %3195 = tosa.mul %inserted_slice_740, %3190 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3196 = tosa.add %3191, %3195 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3197 = tosa.mul %3185, %3189 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_741 = tensor.extract_slice %3185[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_742 = tensor.extract_slice %3185[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %3198 = tensor.empty() : tensor<1x32x40x64xf32>
    %3199 = linalg.negf ins(%extracted_slice_742 : tensor<1x32x40x64xf32>) outs(%3198 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %3200 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_743 = tensor.insert_slice %3199 into %3200[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_744 = tensor.insert_slice %extracted_slice_741 into %inserted_slice_743[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %3201 = tosa.mul %inserted_slice_744, %3190 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3202 = tosa.add %3197, %3201 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3203 = tosa.reshape %19 {new_shape = array<i64: 1, 40, 41>} : (tensor<40x41xf32>) -> tensor<1x40x41xf32>
    %3204 = tosa.reshape %3203 {new_shape = array<i64: 1, 1, 40, 41>} : (tensor<1x40x41xf32>) -> tensor<1x1x40x41xf32>
    %extracted_slice_745 = tensor.extract_slice %3204[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_746 = tensor.extract_slice %extracted_slice_745[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %3205 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x1x40x41xf32>}> : () -> tensor<1x1x40x41xf32>
    %3206 = tosa.add %extracted_slice_746, %3205 : (tensor<1x1x40x41xf32>, tensor<1x1x40x41xf32>) -> tensor<1x1x40x41xf32>
    %extracted_slice_747 = tensor.extract_slice %3206[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_748 = tensor.extract_slice %extracted_slice_747[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_749 = tensor.extract_slice %extracted_slice_748[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_750 = tensor.extract_slice %extracted_slice_749[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x40xf32>
    %cst_751 = arith.constant 0.000000e+00 : f32
    %splat_752 = tensor.splat %cst_751 : tensor<40x40xf32>
    %3207 = tosa.reshape %extracted_slice_750 {new_shape = array<i64: 40, 40>} : (tensor<1x1x40x40xf32>) -> tensor<40x40xf32>
    %3208 = tosa.add %splat_752, %3207 : (tensor<40x40xf32>, tensor<40x40xf32>) -> tensor<40x40xf32>
    %3209 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3210 = tosa.transpose %3202, %3209 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %3211 = tosa.reshape %3196 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %3212 = tosa.reshape %3210 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %3213 = tosa.matmul %3211, %3212 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %cst_753 = arith.constant 0.0883883461 : f32
    %splat_754 = tensor.splat %cst_753 : tensor<32x40x40xf32>
    %3214 = tosa.mul %3213, %splat_754 {shift = 0 : i8} : (tensor<32x40x40xf32>, tensor<32x40x40xf32>) -> tensor<32x40x40xf32>
    %3215 = tosa.add %3214, %3208 : (tensor<32x40x40xf32>, tensor<40x40xf32>) -> tensor<32x40x40xf32>
    %3216 = tosa.reduce_max %3215 {axis = 2 : i32} : (tensor<32x40x40xf32>) -> tensor<32x40x1xf32>
    %3217 = tosa.sub %3215, %3216 : (tensor<32x40x40xf32>, tensor<32x40x1xf32>) -> tensor<32x40x40xf32>
    %3218 = math.exp %3217 : tensor<32x40x40xf32>
    %3219 = tosa.reduce_sum %3218 {axis = 2 : i32} : (tensor<32x40x40xf32>) -> tensor<32x40x1xf32>
    %3220 = tosa.log %3219 : (tensor<32x40x1xf32>) -> tensor<32x40x1xf32>
    %3221 = tosa.add %3216, %3220 : (tensor<32x40x1xf32>, tensor<32x40x1xf32>) -> tensor<32x40x1xf32>
    %3222 = tosa.sub %3215, %3221 : (tensor<32x40x40xf32>, tensor<32x40x1xf32>) -> tensor<32x40x40xf32>
    %3223 = math.exp %3222 : tensor<32x40x40xf32>
    %3224 = tosa.reshape %3221 {new_shape = array<i64: 1, 32, 40>} : (tensor<32x40x1xf32>) -> tensor<1x32x40xf32>
    %3225 = tosa.reshape %3188 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %3226 = tosa.matmul %3223, %3225 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %3227 = tosa.reshape %3226 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3228 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3229 = tosa.transpose %3227, %3228 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %3230 = tosa.reshape %3229 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %3231 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3232 = tosa.transpose %arg250, %3231 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %3233 = tosa.reshape %3230 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_755 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3234 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3233, %3232 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_755 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3235 = tosa.reshape %3234 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3236 = tosa.add %3152, %3235 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %3237 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_756 = arith.constant 2 : i32
    %3238 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3236 : tensor<1x40x4096xf32>) outs(%3237 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3745 = math.fpowi %in, %c2_i32_756 : f32, i32
      linalg.yield %3745 : f32
    } -> tensor<1x40x4096xf32>
    %3239 = tosa.reduce_sum %3238 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %3240 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3241 = tosa.reciprocal %3240 : (tensor<1xf32>) -> tensor<1xf32>
    %3242 = tosa.mul %3241, %3239 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3243 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %3244 = tosa.add %3242, %3243 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3245 = tosa.rsqrt %3244 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3246 = tosa.mul %3236, %3245 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %3247 = tosa.reshape %arg251 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %3248 = tosa.mul %3247, %3246 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %3249 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3250 = tosa.transpose %arg252, %3249 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %3251 = tosa.reshape %3248 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_757 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %3252 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3251, %3250 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_757 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %3253 = tosa.reshape %3252 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %3254 = tosa.sigmoid %3253 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %3255 = tosa.mul %3253, %3254 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %3256 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3257 = tosa.transpose %arg253, %3256 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %3258 = tosa.reshape %3248 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_758 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %3259 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3258, %3257 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_758 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %3260 = tosa.reshape %3259 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %3261 = tosa.mul %3255, %3260 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %3262 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3263 = tosa.transpose %arg254, %3262 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %3264 = tosa.reshape %3261 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_759 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3265 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3264, %3263 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_759 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3266 = tosa.reshape %3265 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3267 = tosa.add %3236, %3266 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %3268 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_760 = arith.constant 2 : i32
    %3269 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3267 : tensor<1x40x4096xf32>) outs(%3268 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3745 = math.fpowi %in, %c2_i32_760 : f32, i32
      linalg.yield %3745 : f32
    } -> tensor<1x40x4096xf32>
    %3270 = tosa.reduce_sum %3269 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %3271 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3272 = tosa.reciprocal %3271 : (tensor<1xf32>) -> tensor<1xf32>
    %3273 = tosa.mul %3272, %3270 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3274 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %3275 = tosa.add %3273, %3274 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3276 = tosa.rsqrt %3275 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3277 = tosa.mul %3267, %3276 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %3278 = tosa.reshape %arg255 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %3279 = tosa.mul %3278, %3277 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %3280 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3281 = tosa.transpose %arg256, %3280 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %3282 = tosa.reshape %3279 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_761 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3283 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3282, %3281 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_761 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3284 = tosa.reshape %3283 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3285 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3286 = tosa.transpose %arg257, %3285 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %3287 = tosa.reshape %3279 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_762 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3288 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3287, %3286 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_762 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3289 = tosa.reshape %3288 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3290 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3291 = tosa.transpose %arg258, %3290 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %3292 = tosa.reshape %3279 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_763 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3293 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3292, %3291 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_763 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3294 = tosa.reshape %3293 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3295 = tosa.reshape %3284 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %3296 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3297 = tosa.transpose %3295, %3296 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %3298 = tosa.reshape %3289 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %3299 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3300 = tosa.transpose %3298, %3299 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %3301 = tosa.reshape %3294 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %3302 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3303 = tosa.transpose %3301, %3302 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %3304 = tosa.reshape %45 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %3305 = tosa.reshape %47 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %3306 = tosa.mul %3297, %3304 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_764 = tensor.extract_slice %3297[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_765 = tensor.extract_slice %3297[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %3307 = tensor.empty() : tensor<1x32x40x64xf32>
    %3308 = linalg.negf ins(%extracted_slice_765 : tensor<1x32x40x64xf32>) outs(%3307 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %3309 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_766 = tensor.insert_slice %3308 into %3309[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_767 = tensor.insert_slice %extracted_slice_764 into %inserted_slice_766[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %3310 = tosa.mul %inserted_slice_767, %3305 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3311 = tosa.add %3306, %3310 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3312 = tosa.mul %3300, %3304 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_768 = tensor.extract_slice %3300[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_769 = tensor.extract_slice %3300[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %3313 = tensor.empty() : tensor<1x32x40x64xf32>
    %3314 = linalg.negf ins(%extracted_slice_769 : tensor<1x32x40x64xf32>) outs(%3313 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %3315 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_770 = tensor.insert_slice %3314 into %3315[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_771 = tensor.insert_slice %extracted_slice_768 into %inserted_slice_770[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %3316 = tosa.mul %inserted_slice_771, %3305 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3317 = tosa.add %3312, %3316 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3318 = tosa.reshape %19 {new_shape = array<i64: 1, 40, 41>} : (tensor<40x41xf32>) -> tensor<1x40x41xf32>
    %3319 = tosa.reshape %3318 {new_shape = array<i64: 1, 1, 40, 41>} : (tensor<1x40x41xf32>) -> tensor<1x1x40x41xf32>
    %extracted_slice_772 = tensor.extract_slice %3319[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_773 = tensor.extract_slice %extracted_slice_772[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %3320 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x1x40x41xf32>}> : () -> tensor<1x1x40x41xf32>
    %3321 = tosa.add %extracted_slice_773, %3320 : (tensor<1x1x40x41xf32>, tensor<1x1x40x41xf32>) -> tensor<1x1x40x41xf32>
    %extracted_slice_774 = tensor.extract_slice %3321[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_775 = tensor.extract_slice %extracted_slice_774[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_776 = tensor.extract_slice %extracted_slice_775[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_777 = tensor.extract_slice %extracted_slice_776[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x40xf32>
    %cst_778 = arith.constant 0.000000e+00 : f32
    %splat_779 = tensor.splat %cst_778 : tensor<40x40xf32>
    %3322 = tosa.reshape %extracted_slice_777 {new_shape = array<i64: 40, 40>} : (tensor<1x1x40x40xf32>) -> tensor<40x40xf32>
    %3323 = tosa.add %splat_779, %3322 : (tensor<40x40xf32>, tensor<40x40xf32>) -> tensor<40x40xf32>
    %3324 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3325 = tosa.transpose %3317, %3324 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %3326 = tosa.reshape %3311 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %3327 = tosa.reshape %3325 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %3328 = tosa.matmul %3326, %3327 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %cst_780 = arith.constant 0.0883883461 : f32
    %splat_781 = tensor.splat %cst_780 : tensor<32x40x40xf32>
    %3329 = tosa.mul %3328, %splat_781 {shift = 0 : i8} : (tensor<32x40x40xf32>, tensor<32x40x40xf32>) -> tensor<32x40x40xf32>
    %3330 = tosa.add %3329, %3323 : (tensor<32x40x40xf32>, tensor<40x40xf32>) -> tensor<32x40x40xf32>
    %3331 = tosa.reduce_max %3330 {axis = 2 : i32} : (tensor<32x40x40xf32>) -> tensor<32x40x1xf32>
    %3332 = tosa.sub %3330, %3331 : (tensor<32x40x40xf32>, tensor<32x40x1xf32>) -> tensor<32x40x40xf32>
    %3333 = math.exp %3332 : tensor<32x40x40xf32>
    %3334 = tosa.reduce_sum %3333 {axis = 2 : i32} : (tensor<32x40x40xf32>) -> tensor<32x40x1xf32>
    %3335 = tosa.log %3334 : (tensor<32x40x1xf32>) -> tensor<32x40x1xf32>
    %3336 = tosa.add %3331, %3335 : (tensor<32x40x1xf32>, tensor<32x40x1xf32>) -> tensor<32x40x1xf32>
    %3337 = tosa.sub %3330, %3336 : (tensor<32x40x40xf32>, tensor<32x40x1xf32>) -> tensor<32x40x40xf32>
    %3338 = math.exp %3337 : tensor<32x40x40xf32>
    %3339 = tosa.reshape %3336 {new_shape = array<i64: 1, 32, 40>} : (tensor<32x40x1xf32>) -> tensor<1x32x40xf32>
    %3340 = tosa.reshape %3303 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %3341 = tosa.matmul %3338, %3340 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %3342 = tosa.reshape %3341 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3343 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3344 = tosa.transpose %3342, %3343 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %3345 = tosa.reshape %3344 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %3346 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3347 = tosa.transpose %arg259, %3346 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %3348 = tosa.reshape %3345 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_782 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3349 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3348, %3347 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_782 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3350 = tosa.reshape %3349 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3351 = tosa.add %3267, %3350 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %3352 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_783 = arith.constant 2 : i32
    %3353 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3351 : tensor<1x40x4096xf32>) outs(%3352 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3745 = math.fpowi %in, %c2_i32_783 : f32, i32
      linalg.yield %3745 : f32
    } -> tensor<1x40x4096xf32>
    %3354 = tosa.reduce_sum %3353 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %3355 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3356 = tosa.reciprocal %3355 : (tensor<1xf32>) -> tensor<1xf32>
    %3357 = tosa.mul %3356, %3354 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3358 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %3359 = tosa.add %3357, %3358 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3360 = tosa.rsqrt %3359 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3361 = tosa.mul %3351, %3360 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %3362 = tosa.reshape %arg260 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %3363 = tosa.mul %3362, %3361 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %3364 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3365 = tosa.transpose %arg261, %3364 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %3366 = tosa.reshape %3363 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_784 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %3367 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3366, %3365 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_784 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %3368 = tosa.reshape %3367 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %3369 = tosa.sigmoid %3368 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %3370 = tosa.mul %3368, %3369 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %3371 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3372 = tosa.transpose %arg262, %3371 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %3373 = tosa.reshape %3363 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_785 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %3374 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3373, %3372 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_785 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %3375 = tosa.reshape %3374 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %3376 = tosa.mul %3370, %3375 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %3377 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3378 = tosa.transpose %arg263, %3377 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %3379 = tosa.reshape %3376 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_786 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3380 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3379, %3378 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_786 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3381 = tosa.reshape %3380 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3382 = tosa.add %3351, %3381 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %3383 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_787 = arith.constant 2 : i32
    %3384 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3382 : tensor<1x40x4096xf32>) outs(%3383 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3745 = math.fpowi %in, %c2_i32_787 : f32, i32
      linalg.yield %3745 : f32
    } -> tensor<1x40x4096xf32>
    %3385 = tosa.reduce_sum %3384 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %3386 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3387 = tosa.reciprocal %3386 : (tensor<1xf32>) -> tensor<1xf32>
    %3388 = tosa.mul %3387, %3385 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3389 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %3390 = tosa.add %3388, %3389 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3391 = tosa.rsqrt %3390 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3392 = tosa.mul %3382, %3391 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %3393 = tosa.reshape %arg264 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %3394 = tosa.mul %3393, %3392 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %3395 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3396 = tosa.transpose %arg265, %3395 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %3397 = tosa.reshape %3394 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_788 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3398 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3397, %3396 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_788 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3399 = tosa.reshape %3398 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3400 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3401 = tosa.transpose %arg266, %3400 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %3402 = tosa.reshape %3394 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_789 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3403 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3402, %3401 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_789 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3404 = tosa.reshape %3403 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3405 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3406 = tosa.transpose %arg267, %3405 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %3407 = tosa.reshape %3394 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_790 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3408 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3407, %3406 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_790 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3409 = tosa.reshape %3408 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3410 = tosa.reshape %3399 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %3411 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3412 = tosa.transpose %3410, %3411 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %3413 = tosa.reshape %3404 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %3414 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3415 = tosa.transpose %3413, %3414 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %3416 = tosa.reshape %3409 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %3417 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3418 = tosa.transpose %3416, %3417 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %3419 = tosa.reshape %45 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %3420 = tosa.reshape %47 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %3421 = tosa.mul %3412, %3419 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_791 = tensor.extract_slice %3412[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_792 = tensor.extract_slice %3412[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %3422 = tensor.empty() : tensor<1x32x40x64xf32>
    %3423 = linalg.negf ins(%extracted_slice_792 : tensor<1x32x40x64xf32>) outs(%3422 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %3424 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_793 = tensor.insert_slice %3423 into %3424[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_794 = tensor.insert_slice %extracted_slice_791 into %inserted_slice_793[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %3425 = tosa.mul %inserted_slice_794, %3420 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3426 = tosa.add %3421, %3425 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3427 = tosa.mul %3415, %3419 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_795 = tensor.extract_slice %3415[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_796 = tensor.extract_slice %3415[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %3428 = tensor.empty() : tensor<1x32x40x64xf32>
    %3429 = linalg.negf ins(%extracted_slice_796 : tensor<1x32x40x64xf32>) outs(%3428 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %3430 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_797 = tensor.insert_slice %3429 into %3430[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_798 = tensor.insert_slice %extracted_slice_795 into %inserted_slice_797[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %3431 = tosa.mul %inserted_slice_798, %3420 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3432 = tosa.add %3427, %3431 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3433 = tosa.reshape %19 {new_shape = array<i64: 1, 40, 41>} : (tensor<40x41xf32>) -> tensor<1x40x41xf32>
    %3434 = tosa.reshape %3433 {new_shape = array<i64: 1, 1, 40, 41>} : (tensor<1x40x41xf32>) -> tensor<1x1x40x41xf32>
    %extracted_slice_799 = tensor.extract_slice %3434[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_800 = tensor.extract_slice %extracted_slice_799[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %3435 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x1x40x41xf32>}> : () -> tensor<1x1x40x41xf32>
    %3436 = tosa.add %extracted_slice_800, %3435 : (tensor<1x1x40x41xf32>, tensor<1x1x40x41xf32>) -> tensor<1x1x40x41xf32>
    %extracted_slice_801 = tensor.extract_slice %3436[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_802 = tensor.extract_slice %extracted_slice_801[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_803 = tensor.extract_slice %extracted_slice_802[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_804 = tensor.extract_slice %extracted_slice_803[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x40xf32>
    %cst_805 = arith.constant 0.000000e+00 : f32
    %splat_806 = tensor.splat %cst_805 : tensor<40x40xf32>
    %3437 = tosa.reshape %extracted_slice_804 {new_shape = array<i64: 40, 40>} : (tensor<1x1x40x40xf32>) -> tensor<40x40xf32>
    %3438 = tosa.add %splat_806, %3437 : (tensor<40x40xf32>, tensor<40x40xf32>) -> tensor<40x40xf32>
    %3439 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3440 = tosa.transpose %3432, %3439 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %3441 = tosa.reshape %3426 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %3442 = tosa.reshape %3440 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %3443 = tosa.matmul %3441, %3442 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %cst_807 = arith.constant 0.0883883461 : f32
    %splat_808 = tensor.splat %cst_807 : tensor<32x40x40xf32>
    %3444 = tosa.mul %3443, %splat_808 {shift = 0 : i8} : (tensor<32x40x40xf32>, tensor<32x40x40xf32>) -> tensor<32x40x40xf32>
    %3445 = tosa.add %3444, %3438 : (tensor<32x40x40xf32>, tensor<40x40xf32>) -> tensor<32x40x40xf32>
    %3446 = tosa.reduce_max %3445 {axis = 2 : i32} : (tensor<32x40x40xf32>) -> tensor<32x40x1xf32>
    %3447 = tosa.sub %3445, %3446 : (tensor<32x40x40xf32>, tensor<32x40x1xf32>) -> tensor<32x40x40xf32>
    %3448 = math.exp %3447 : tensor<32x40x40xf32>
    %3449 = tosa.reduce_sum %3448 {axis = 2 : i32} : (tensor<32x40x40xf32>) -> tensor<32x40x1xf32>
    %3450 = tosa.log %3449 : (tensor<32x40x1xf32>) -> tensor<32x40x1xf32>
    %3451 = tosa.add %3446, %3450 : (tensor<32x40x1xf32>, tensor<32x40x1xf32>) -> tensor<32x40x1xf32>
    %3452 = tosa.sub %3445, %3451 : (tensor<32x40x40xf32>, tensor<32x40x1xf32>) -> tensor<32x40x40xf32>
    %3453 = math.exp %3452 : tensor<32x40x40xf32>
    %3454 = tosa.reshape %3451 {new_shape = array<i64: 1, 32, 40>} : (tensor<32x40x1xf32>) -> tensor<1x32x40xf32>
    %3455 = tosa.reshape %3418 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %3456 = tosa.matmul %3453, %3455 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %3457 = tosa.reshape %3456 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3458 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3459 = tosa.transpose %3457, %3458 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %3460 = tosa.reshape %3459 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %3461 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3462 = tosa.transpose %arg268, %3461 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %3463 = tosa.reshape %3460 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_809 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3464 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3463, %3462 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_809 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3465 = tosa.reshape %3464 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3466 = tosa.add %3382, %3465 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %3467 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_810 = arith.constant 2 : i32
    %3468 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3466 : tensor<1x40x4096xf32>) outs(%3467 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3745 = math.fpowi %in, %c2_i32_810 : f32, i32
      linalg.yield %3745 : f32
    } -> tensor<1x40x4096xf32>
    %3469 = tosa.reduce_sum %3468 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %3470 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3471 = tosa.reciprocal %3470 : (tensor<1xf32>) -> tensor<1xf32>
    %3472 = tosa.mul %3471, %3469 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3473 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %3474 = tosa.add %3472, %3473 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3475 = tosa.rsqrt %3474 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3476 = tosa.mul %3466, %3475 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %3477 = tosa.reshape %arg269 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %3478 = tosa.mul %3477, %3476 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %3479 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3480 = tosa.transpose %arg270, %3479 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %3481 = tosa.reshape %3478 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_811 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %3482 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3481, %3480 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_811 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %3483 = tosa.reshape %3482 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %3484 = tosa.sigmoid %3483 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %3485 = tosa.mul %3483, %3484 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %3486 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3487 = tosa.transpose %arg271, %3486 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %3488 = tosa.reshape %3478 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_812 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %3489 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3488, %3487 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_812 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %3490 = tosa.reshape %3489 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %3491 = tosa.mul %3485, %3490 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %3492 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3493 = tosa.transpose %arg272, %3492 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %3494 = tosa.reshape %3491 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_813 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3495 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3494, %3493 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_813 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3496 = tosa.reshape %3495 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3497 = tosa.add %3466, %3496 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %3498 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_814 = arith.constant 2 : i32
    %3499 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3497 : tensor<1x40x4096xf32>) outs(%3498 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3745 = math.fpowi %in, %c2_i32_814 : f32, i32
      linalg.yield %3745 : f32
    } -> tensor<1x40x4096xf32>
    %3500 = tosa.reduce_sum %3499 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %3501 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3502 = tosa.reciprocal %3501 : (tensor<1xf32>) -> tensor<1xf32>
    %3503 = tosa.mul %3502, %3500 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3504 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %3505 = tosa.add %3503, %3504 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3506 = tosa.rsqrt %3505 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3507 = tosa.mul %3497, %3506 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %3508 = tosa.reshape %arg273 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %3509 = tosa.mul %3508, %3507 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %3510 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3511 = tosa.transpose %arg274, %3510 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %3512 = tosa.reshape %3509 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_815 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3513 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3512, %3511 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_815 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3514 = tosa.reshape %3513 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3515 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3516 = tosa.transpose %arg275, %3515 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %3517 = tosa.reshape %3509 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_816 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3518 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3517, %3516 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_816 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3519 = tosa.reshape %3518 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3520 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3521 = tosa.transpose %arg276, %3520 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %3522 = tosa.reshape %3509 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_817 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3523 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3522, %3521 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_817 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3524 = tosa.reshape %3523 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3525 = tosa.reshape %3514 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %3526 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3527 = tosa.transpose %3525, %3526 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %3528 = tosa.reshape %3519 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %3529 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3530 = tosa.transpose %3528, %3529 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %3531 = tosa.reshape %3524 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %3532 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3533 = tosa.transpose %3531, %3532 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %3534 = tosa.reshape %45 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %3535 = tosa.reshape %47 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %3536 = tosa.mul %3527, %3534 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_818 = tensor.extract_slice %3527[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_819 = tensor.extract_slice %3527[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %3537 = tensor.empty() : tensor<1x32x40x64xf32>
    %3538 = linalg.negf ins(%extracted_slice_819 : tensor<1x32x40x64xf32>) outs(%3537 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %3539 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_820 = tensor.insert_slice %3538 into %3539[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_821 = tensor.insert_slice %extracted_slice_818 into %inserted_slice_820[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %3540 = tosa.mul %inserted_slice_821, %3535 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3541 = tosa.add %3536, %3540 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3542 = tosa.mul %3530, %3534 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_822 = tensor.extract_slice %3530[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_823 = tensor.extract_slice %3530[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %3543 = tensor.empty() : tensor<1x32x40x64xf32>
    %3544 = linalg.negf ins(%extracted_slice_823 : tensor<1x32x40x64xf32>) outs(%3543 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %3545 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_824 = tensor.insert_slice %3544 into %3545[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_825 = tensor.insert_slice %extracted_slice_822 into %inserted_slice_824[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %3546 = tosa.mul %inserted_slice_825, %3535 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3547 = tosa.add %3542, %3546 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3548 = tosa.reshape %19 {new_shape = array<i64: 1, 40, 41>} : (tensor<40x41xf32>) -> tensor<1x40x41xf32>
    %3549 = tosa.reshape %3548 {new_shape = array<i64: 1, 1, 40, 41>} : (tensor<1x40x41xf32>) -> tensor<1x1x40x41xf32>
    %extracted_slice_826 = tensor.extract_slice %3549[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_827 = tensor.extract_slice %extracted_slice_826[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %3550 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x1x40x41xf32>}> : () -> tensor<1x1x40x41xf32>
    %3551 = tosa.add %extracted_slice_827, %3550 : (tensor<1x1x40x41xf32>, tensor<1x1x40x41xf32>) -> tensor<1x1x40x41xf32>
    %extracted_slice_828 = tensor.extract_slice %3551[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_829 = tensor.extract_slice %extracted_slice_828[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_830 = tensor.extract_slice %extracted_slice_829[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_831 = tensor.extract_slice %extracted_slice_830[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x40xf32>
    %cst_832 = arith.constant 0.000000e+00 : f32
    %splat_833 = tensor.splat %cst_832 : tensor<40x40xf32>
    %3552 = tosa.reshape %extracted_slice_831 {new_shape = array<i64: 40, 40>} : (tensor<1x1x40x40xf32>) -> tensor<40x40xf32>
    %3553 = tosa.add %splat_833, %3552 : (tensor<40x40xf32>, tensor<40x40xf32>) -> tensor<40x40xf32>
    %3554 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3555 = tosa.transpose %3547, %3554 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %3556 = tosa.reshape %3541 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %3557 = tosa.reshape %3555 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %3558 = tosa.matmul %3556, %3557 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %cst_834 = arith.constant 0.0883883461 : f32
    %splat_835 = tensor.splat %cst_834 : tensor<32x40x40xf32>
    %3559 = tosa.mul %3558, %splat_835 {shift = 0 : i8} : (tensor<32x40x40xf32>, tensor<32x40x40xf32>) -> tensor<32x40x40xf32>
    %3560 = tosa.add %3559, %3553 : (tensor<32x40x40xf32>, tensor<40x40xf32>) -> tensor<32x40x40xf32>
    %3561 = tosa.reduce_max %3560 {axis = 2 : i32} : (tensor<32x40x40xf32>) -> tensor<32x40x1xf32>
    %3562 = tosa.sub %3560, %3561 : (tensor<32x40x40xf32>, tensor<32x40x1xf32>) -> tensor<32x40x40xf32>
    %3563 = math.exp %3562 : tensor<32x40x40xf32>
    %3564 = tosa.reduce_sum %3563 {axis = 2 : i32} : (tensor<32x40x40xf32>) -> tensor<32x40x1xf32>
    %3565 = tosa.log %3564 : (tensor<32x40x1xf32>) -> tensor<32x40x1xf32>
    %3566 = tosa.add %3561, %3565 : (tensor<32x40x1xf32>, tensor<32x40x1xf32>) -> tensor<32x40x1xf32>
    %3567 = tosa.sub %3560, %3566 : (tensor<32x40x40xf32>, tensor<32x40x1xf32>) -> tensor<32x40x40xf32>
    %3568 = math.exp %3567 : tensor<32x40x40xf32>
    %3569 = tosa.reshape %3566 {new_shape = array<i64: 1, 32, 40>} : (tensor<32x40x1xf32>) -> tensor<1x32x40xf32>
    %3570 = tosa.reshape %3533 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %3571 = tosa.matmul %3568, %3570 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %3572 = tosa.reshape %3571 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3573 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3574 = tosa.transpose %3572, %3573 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %3575 = tosa.reshape %3574 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %3576 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3577 = tosa.transpose %arg277, %3576 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %3578 = tosa.reshape %3575 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_836 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3579 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3578, %3577 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_836 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3580 = tosa.reshape %3579 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3581 = tosa.add %3497, %3580 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %3582 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_837 = arith.constant 2 : i32
    %3583 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3581 : tensor<1x40x4096xf32>) outs(%3582 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3745 = math.fpowi %in, %c2_i32_837 : f32, i32
      linalg.yield %3745 : f32
    } -> tensor<1x40x4096xf32>
    %3584 = tosa.reduce_sum %3583 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %3585 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3586 = tosa.reciprocal %3585 : (tensor<1xf32>) -> tensor<1xf32>
    %3587 = tosa.mul %3586, %3584 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3588 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %3589 = tosa.add %3587, %3588 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3590 = tosa.rsqrt %3589 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3591 = tosa.mul %3581, %3590 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %3592 = tosa.reshape %arg278 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %3593 = tosa.mul %3592, %3591 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %3594 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3595 = tosa.transpose %arg279, %3594 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %3596 = tosa.reshape %3593 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_838 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %3597 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3596, %3595 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_838 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %3598 = tosa.reshape %3597 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %3599 = tosa.sigmoid %3598 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %3600 = tosa.mul %3598, %3599 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %3601 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3602 = tosa.transpose %arg280, %3601 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %3603 = tosa.reshape %3593 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_839 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %3604 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3603, %3602 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_839 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %3605 = tosa.reshape %3604 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %3606 = tosa.mul %3600, %3605 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %3607 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3608 = tosa.transpose %arg281, %3607 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %3609 = tosa.reshape %3606 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_840 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3610 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3609, %3608 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_840 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3611 = tosa.reshape %3610 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3612 = tosa.add %3581, %3611 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %3613 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_841 = arith.constant 2 : i32
    %3614 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3612 : tensor<1x40x4096xf32>) outs(%3613 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3745 = math.fpowi %in, %c2_i32_841 : f32, i32
      linalg.yield %3745 : f32
    } -> tensor<1x40x4096xf32>
    %3615 = tosa.reduce_sum %3614 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %3616 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3617 = tosa.reciprocal %3616 : (tensor<1xf32>) -> tensor<1xf32>
    %3618 = tosa.mul %3617, %3615 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3619 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %3620 = tosa.add %3618, %3619 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3621 = tosa.rsqrt %3620 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3622 = tosa.mul %3612, %3621 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %3623 = tosa.reshape %arg282 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %3624 = tosa.mul %3623, %3622 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %3625 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3626 = tosa.transpose %arg283, %3625 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %3627 = tosa.reshape %3624 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_842 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3628 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3627, %3626 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_842 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3629 = tosa.reshape %3628 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3630 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3631 = tosa.transpose %arg284, %3630 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %3632 = tosa.reshape %3624 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_843 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3633 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3632, %3631 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_843 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3634 = tosa.reshape %3633 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3635 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3636 = tosa.transpose %arg285, %3635 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %3637 = tosa.reshape %3624 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_844 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3638 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3637, %3636 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_844 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3639 = tosa.reshape %3638 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3640 = tosa.reshape %3629 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %3641 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3642 = tosa.transpose %3640, %3641 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %3643 = tosa.reshape %3634 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %3644 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3645 = tosa.transpose %3643, %3644 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %3646 = tosa.reshape %3639 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %3647 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3648 = tosa.transpose %3646, %3647 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %3649 = tosa.reshape %45 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %3650 = tosa.reshape %47 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %3651 = tosa.mul %3642, %3649 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_845 = tensor.extract_slice %3642[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_846 = tensor.extract_slice %3642[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %3652 = tensor.empty() : tensor<1x32x40x64xf32>
    %3653 = linalg.negf ins(%extracted_slice_846 : tensor<1x32x40x64xf32>) outs(%3652 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %3654 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_847 = tensor.insert_slice %3653 into %3654[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_848 = tensor.insert_slice %extracted_slice_845 into %inserted_slice_847[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %3655 = tosa.mul %inserted_slice_848, %3650 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3656 = tosa.add %3651, %3655 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3657 = tosa.mul %3645, %3649 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_849 = tensor.extract_slice %3645[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_850 = tensor.extract_slice %3645[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %3658 = tensor.empty() : tensor<1x32x40x64xf32>
    %3659 = linalg.negf ins(%extracted_slice_850 : tensor<1x32x40x64xf32>) outs(%3658 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %3660 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_851 = tensor.insert_slice %3659 into %3660[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_852 = tensor.insert_slice %extracted_slice_849 into %inserted_slice_851[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %3661 = tosa.mul %inserted_slice_852, %3650 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3662 = tosa.add %3657, %3661 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3663 = tosa.reshape %19 {new_shape = array<i64: 1, 40, 41>} : (tensor<40x41xf32>) -> tensor<1x40x41xf32>
    %3664 = tosa.reshape %3663 {new_shape = array<i64: 1, 1, 40, 41>} : (tensor<1x40x41xf32>) -> tensor<1x1x40x41xf32>
    %extracted_slice_853 = tensor.extract_slice %3664[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_854 = tensor.extract_slice %extracted_slice_853[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %3665 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x1x40x41xf32>}> : () -> tensor<1x1x40x41xf32>
    %3666 = tosa.add %extracted_slice_854, %3665 : (tensor<1x1x40x41xf32>, tensor<1x1x40x41xf32>) -> tensor<1x1x40x41xf32>
    %extracted_slice_855 = tensor.extract_slice %3666[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_856 = tensor.extract_slice %extracted_slice_855[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_857 = tensor.extract_slice %extracted_slice_856[0, 0, 0, 0] [1, 1, 40, 41] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x41xf32>
    %extracted_slice_858 = tensor.extract_slice %extracted_slice_857[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x41xf32> to tensor<1x1x40x40xf32>
    %cst_859 = arith.constant 0.000000e+00 : f32
    %splat_860 = tensor.splat %cst_859 : tensor<40x40xf32>
    %3667 = tosa.reshape %extracted_slice_858 {new_shape = array<i64: 40, 40>} : (tensor<1x1x40x40xf32>) -> tensor<40x40xf32>
    %3668 = tosa.add %splat_860, %3667 : (tensor<40x40xf32>, tensor<40x40xf32>) -> tensor<40x40xf32>
    %3669 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3670 = tosa.transpose %3662, %3669 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %3671 = tosa.reshape %3656 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %3672 = tosa.reshape %3670 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %3673 = tosa.matmul %3671, %3672 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %cst_861 = arith.constant 0.0883883461 : f32
    %splat_862 = tensor.splat %cst_861 : tensor<32x40x40xf32>
    %3674 = tosa.mul %3673, %splat_862 {shift = 0 : i8} : (tensor<32x40x40xf32>, tensor<32x40x40xf32>) -> tensor<32x40x40xf32>
    %3675 = tosa.add %3674, %3668 : (tensor<32x40x40xf32>, tensor<40x40xf32>) -> tensor<32x40x40xf32>
    %3676 = tosa.reduce_max %3675 {axis = 2 : i32} : (tensor<32x40x40xf32>) -> tensor<32x40x1xf32>
    %3677 = tosa.sub %3675, %3676 : (tensor<32x40x40xf32>, tensor<32x40x1xf32>) -> tensor<32x40x40xf32>
    %3678 = math.exp %3677 : tensor<32x40x40xf32>
    %3679 = tosa.reduce_sum %3678 {axis = 2 : i32} : (tensor<32x40x40xf32>) -> tensor<32x40x1xf32>
    %3680 = tosa.log %3679 : (tensor<32x40x1xf32>) -> tensor<32x40x1xf32>
    %3681 = tosa.add %3676, %3680 : (tensor<32x40x1xf32>, tensor<32x40x1xf32>) -> tensor<32x40x1xf32>
    %3682 = tosa.sub %3675, %3681 : (tensor<32x40x40xf32>, tensor<32x40x1xf32>) -> tensor<32x40x40xf32>
    %3683 = math.exp %3682 : tensor<32x40x40xf32>
    %3684 = tosa.reshape %3681 {new_shape = array<i64: 1, 32, 40>} : (tensor<32x40x1xf32>) -> tensor<1x32x40xf32>
    %3685 = tosa.reshape %3648 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %3686 = tosa.matmul %3683, %3685 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %3687 = tosa.reshape %3686 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3688 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3689 = tosa.transpose %3687, %3688 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %3690 = tosa.reshape %3689 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %3691 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3692 = tosa.transpose %arg286, %3691 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %3693 = tosa.reshape %3690 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_863 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3694 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3693, %3692 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_863 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3695 = tosa.reshape %3694 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3696 = tosa.add %3612, %3695 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %3697 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_864 = arith.constant 2 : i32
    %3698 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3696 : tensor<1x40x4096xf32>) outs(%3697 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3745 = math.fpowi %in, %c2_i32_864 : f32, i32
      linalg.yield %3745 : f32
    } -> tensor<1x40x4096xf32>
    %3699 = tosa.reduce_sum %3698 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %3700 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3701 = tosa.reciprocal %3700 : (tensor<1xf32>) -> tensor<1xf32>
    %3702 = tosa.mul %3701, %3699 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3703 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %3704 = tosa.add %3702, %3703 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3705 = tosa.rsqrt %3704 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3706 = tosa.mul %3696, %3705 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %3707 = tosa.reshape %arg287 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %3708 = tosa.mul %3707, %3706 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %3709 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3710 = tosa.transpose %arg288, %3709 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %3711 = tosa.reshape %3708 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_865 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %3712 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3711, %3710 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_865 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %3713 = tosa.reshape %3712 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %3714 = tosa.sigmoid %3713 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %3715 = tosa.mul %3713, %3714 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %3716 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3717 = tosa.transpose %arg289, %3716 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %3718 = tosa.reshape %3708 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_866 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %3719 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3718, %3717 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_866 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %3720 = tosa.reshape %3719 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %3721 = tosa.mul %3715, %3720 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %3722 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3723 = tosa.transpose %arg290, %3722 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %3724 = tosa.reshape %3721 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_867 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3725 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3724, %3723 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_867 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3726 = tosa.reshape %3725 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3727 = tosa.add %3696, %3726 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %3728 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_868 = arith.constant 2 : i32
    %3729 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3727 : tensor<1x40x4096xf32>) outs(%3728 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3745 = math.fpowi %in, %c2_i32_868 : f32, i32
      linalg.yield %3745 : f32
    } -> tensor<1x40x4096xf32>
    %3730 = tosa.reduce_sum %3729 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %3731 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3732 = tosa.reciprocal %3731 : (tensor<1xf32>) -> tensor<1xf32>
    %3733 = tosa.mul %3732, %3730 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3734 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %3735 = tosa.add %3733, %3734 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3736 = tosa.rsqrt %3735 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3737 = tosa.mul %3727, %3736 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %3738 = tosa.reshape %arg291 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %3739 = tosa.mul %3738, %3737 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %extracted_slice_869 = tensor.extract_slice %3739[0, 0, 0] [1, 40, 4096] [1, 1, 1] : tensor<1x40x4096xf32> to tensor<1x40x4096xf32>
    %extracted_slice_870 = tensor.extract_slice %extracted_slice_869[0, 0, 0] [1, 40, 4096] [1, 1, 1] : tensor<1x40x4096xf32> to tensor<1x40x4096xf32>
    %extracted_slice_871 = tensor.extract_slice %extracted_slice_870[0, 0, 0] [1, 40, 4096] [1, 1, 1] : tensor<1x40x4096xf32> to tensor<1x40x4096xf32>
    %3740 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3741 = tosa.transpose %arg292, %3740 : (tensor<32000x4096xf32>, tensor<2xi32>) -> tensor<4096x32000xf32>
    %3742 = tosa.reshape %extracted_slice_871 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_872 = arith.constant dense<0.000000e+00> : tensor<40x32000xf32>
    %3743 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3742, %3741 : tensor<40x4096xf32>, tensor<4096x32000xf32>) outs(%cst_872 : tensor<40x32000xf32>) -> tensor<40x32000xf32>
    %3744 = tosa.reshape %3743 {new_shape = array<i64: 1, 40, 32000>} : (tensor<40x32000xf32>) -> tensor<1x40x32000xf32>
    return %3744 : tensor<1x40x32000xf32>
  }
}

