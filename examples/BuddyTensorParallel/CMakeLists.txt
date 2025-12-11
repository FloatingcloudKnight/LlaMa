
set(BUFFERIZE_FULL_OPTS "unknown-type-conversion=identity-layout-map function-boundary-type-conversion=identity-layout-map bufferize-function-boundaries")
set(BUFFERIZE_SIMPLE_OPTS "bufferize-function-boundaries")
set(TOSA_PIPELINE "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))")


set(FORWARD_PASSES
 -eliminate-empty-tensors
 -empty-tensor-to-alloc-tensor
 -one-shot-bufferize=${BUFFERIZE_SIMPLE_OPTS}
 -expand-strided-metadata
 -ownership-based-buffer-deallocation
 -buffer-deallocation-simplification
 -bufferization-lower-deallocations
 -matmul-vectorization-blis
 -batchmatmul-optimize
 -convert-linalg-to-affine-loops
 -affine-loop-fusion
 -affine-parallelize
 -convert-vector-to-scf
 -lower-affine
 -convert-scf-to-openmp=num-threads=32
 -cse
 -memref-expand
 -arith-expand
 -convert-vector-to-llvm
 -convert-arith-to-llvm
 -finalize-memref-to-llvm
 -convert-scf-to-cf
 -convert-cf-to-llvm
 -llvm-request-c-wrappers
 -convert-openmp-to-llvm
 -convert-arith-to-llvm
 -convert-math-to-llvm
 -convert-math-to-libm
  -convert-func-to-llvm
  -reconcile-unrealized-casts
)

set(SUBGRAPH_PASSES
  -eliminate-empty-tensors
  -empty-tensor-to-alloc-tensor
  -convert-elementwise-to-linalg
  -one-shot-bufferize=${BUFFERIZE_SIMPLE_OPTS}
  -expand-strided-metadata
  -ownership-based-buffer-deallocation
  -buffer-deallocation-simplification
  -bufferization-lower-deallocations
  -matmul-vectorization-blis
  -batchmatmul-optimize
  -convert-linalg-to-affine-loops
  -affine-loop-fusion
  -affine-parallelize
  -convert-vector-to-scf
  -lower-affine
  -convert-scf-to-openmp=num-threads=32
  -func-bufferize-dynamic-offset
  -cse
  -memref-expand
  -arith-expand
  -convert-vector-to-llvm
  -convert-arith-to-llvm
  -finalize-memref-to-llvm
  -convert-scf-to-cf
  -convert-cf-to-llvm
  -llvm-request-c-wrappers
  -convert-openmp-to-llvm
  -convert-arith-to-llvm
  -convert-math-to-llvm
  -convert-math-to-libm
  -convert-func-to-llvm
  -reconcile-unrealized-casts
)

# string(REPLACE ";" " " FORWARD_PASSES_STR "${FORWARD_PASSES_LIST}")
# string(REPLACE ";" " " SUBGRAPH_PASSES_STR "${SUBGRAPH_PASSES}")

function(build_forward_prefill N)
  add_custom_command(
    OUTPUT forward${N}_prefill.o
    COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_SOURCE_DIR}/arch/dstest/forward${N}_prefill.mlir
            -simplify-tosa-reshape |
            ${LLVM_TOOLS_BINARY_DIR}/mlir-opt -pass-pipeline ${TOSA_PIPELINE}
            | ${BUDDY_BINARY_DIR}/buddy-opt ${FORWARD_PASSES}
            | ${LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir
            | ${LLVM_TOOLS_BINARY_DIR}/llvm-as
            | ${LLVM_TOOLS_BINARY_DIR}/llc -filetype=obj -relocation-model=pic -O3
              -o ${CMAKE_CURRENT_BINARY_DIR}/forward${N}_prefill.o
    DEPENDS buddy-opt ${CMAKE_CURRENT_SOURCE_DIR}/arch/dstest/forward${N}_prefill.mlir
    COMMENT "Building forward${N}_prefill.o"
    VERBATIM)
endfunction()

function(build_subgraph_prefill N)
  add_custom_command(
    OUTPUT subgraph${N}_prefill.o
    COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_SOURCE_DIR}/arch/dstest/subgraph${N}_prefill.mlir
            -simplify-tosa-reshape |
            ${LLVM_TOOLS_BINARY_DIR}/mlir-opt -pass-pipeline ${TOSA_PIPELINE}
            | ${BUDDY_BINARY_DIR}/buddy-opt ${SUBGRAPH_PASSES}
            | ${LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir
            | ${LLVM_TOOLS_BINARY_DIR}/llvm-as
            | ${LLVM_TOOLS_BINARY_DIR}/llc -filetype=obj -relocation-model=pic -O3
              -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph${N}_prefill.o
    DEPENDS buddy-opt ${CMAKE_CURRENT_SOURCE_DIR}/arch/dstest/subgraph${N}_prefill.mlir
    COMMENT "Building subgraph${N}_prefill.o"
    VERBATIM)
endfunction()

foreach(N 0 1 2 3 5 169)
  build_forward_prefill(${N})
  build_subgraph_prefill(${N})
  add_library(DISLLAMA${N}_prefill STATIC forward${N}_prefill.o subgraph${N}_prefill.o)
endforeach()

function(build_forward_decode N)
  add_custom_command(
    OUTPUT forward${N}_decode.o
    COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_SOURCE_DIR}/arch/dstest/forward${N}_decode.mlir
            -simplify-tosa-reshape |
            ${LLVM_TOOLS_BINARY_DIR}/mlir-opt -pass-pipeline ${TOSA_PIPELINE}
            | ${BUDDY_BINARY_DIR}/buddy-opt ${FORWARD_PASSES}
            | ${LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir
            | ${LLVM_TOOLS_BINARY_DIR}/llvm-as
            | ${LLVM_TOOLS_BINARY_DIR}/llc -filetype=obj -relocation-model=pic -O3
              -o ${CMAKE_CURRENT_BINARY_DIR}/forward${N}_decode.o
    DEPENDS buddy-opt ${CMAKE_CURRENT_SOURCE_DIR}/arch/dstest/forward${N}_decode.mlir
    COMMENT "Building forward${N}_decode.o"
    VERBATIM)
endfunction()

function(build_subgraph_decode N)
  add_custom_command(
    OUTPUT subgraph${N}_decode.o
    COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_SOURCE_DIR}/arch/dstest/subgraph${N}_decode.mlir
            -simplify-tosa-reshape |
            ${LLVM_TOOLS_BINARY_DIR}/mlir-opt -pass-pipeline ${TOSA_PIPELINE}
            | ${BUDDY_BINARY_DIR}/buddy-opt ${SUBGRAPH_PASSES}
            | ${LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir
            | ${LLVM_TOOLS_BINARY_DIR}/llvm-as
            | ${LLVM_TOOLS_BINARY_DIR}/llc -filetype=obj -relocation-model=pic -O3
              -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph${N}_decode.o
    DEPENDS buddy-opt ${CMAKE_CURRENT_SOURCE_DIR}/arch/dstest/subgraph${N}_decode.mlir
    COMMENT "Building subgraph${N}_decode.o"
    VERBATIM)
endfunction()

foreach(N 0 1 2 3 5 169)
  build_forward_decode(${N})
  build_subgraph_decode(${N})
  add_library(DISLLAMA${N}_decode STATIC forward${N}_decode.o subgraph${N}_decode.o)
endforeach()



SET_SOURCE_FILES_PROPERTIES(
  template.o
  PROPERTIES
  EXTERNAL_OBJECT true
  GENERATED true)

SET_TARGET_PROPERTIES(
  DISLLAMA0_prefill
  DISLLAMA1_prefill
  DISLLAMA2_prefill
  DISLLAMA3_prefill
  DISLLAMA5_prefill
  DISLLAMA169_prefill
  DISLLAMA0_decode
  DISLLAMA1_decode
  DISLLAMA2_decode
  DISLLAMA3_decode
  DISLLAMA5_decode
  DISLLAMA169_decode
  PROPERTIES
  LINKER_LANGUAGE C)

add_executable(buddy-dis-llama-run llama-main.cpp)

set(LLAMA_DIS_EXAMPLE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
set(LLAMA_EXAMPLE_BUILD_PATH "${CMAKE_CURRENT_SOURCE_DIR}/arch/tempGraphPrefill")

target_compile_definitions(buddy-dis-llama-run PRIVATE 
  LLAMA_DIS_EXAMPLE_PATH="${LLAMA_DIS_EXAMPLE_PATH}" 
  LLAMA_EXAMPLE_BUILD_PATH="${LLAMA_EXAMPLE_BUILD_PATH}"
)

target_link_directories(buddy-dis-llama-run PRIVATE ${LLVM_LIBRARY_DIR})

set(BUDDY_DIS_LLAMA_LIBS
  DISLLAMA0_prefill
  DISLLAMA1_prefill
  DISLLAMA2_prefill
  DISLLAMA3_prefill
  DISLLAMA5_prefill
  DISLLAMA169_prefill
  DISLLAMA0_decode
  DISLLAMA1_decode
  DISLLAMA2_decode
  DISLLAMA3_decode
  DISLLAMA5_decode
  DISLLAMA169_decode
  mlir_c_runner_utils
  omp
)

target_link_libraries(buddy-dis-llama-run ${BUDDY_DIS_LLAMA_LIBS})

