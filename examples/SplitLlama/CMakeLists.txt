# -*- coding: utf-8 -*-
# @Time : 2025-10-24 10:00:00
# @Author : gaoxu
# @Project : buddy compiler llama

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/forward0.mlir 
         ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.mlir 
         ${CMAKE_CURRENT_BINARY_DIR}/forward1.mlir 
         ${CMAKE_CURRENT_BINARY_DIR}/subgraph1.mlir 
         ${CMAKE_CURRENT_BINARY_DIR}/forward2.mlir 
         ${CMAKE_CURRENT_BINARY_DIR}/subgraph2.mlir
         ${CMAKE_CURRENT_BINARY_DIR}/forward3.mlir 
         ${CMAKE_CURRENT_BINARY_DIR}/subgraph3.mlir
         ${CMAKE_CURRENT_BINARY_DIR}/forward5.mlir 
         ${CMAKE_CURRENT_BINARY_DIR}/subgraph5.mlir
         ${CMAKE_CURRENT_BINARY_DIR}/forward193.mlir 
         ${CMAKE_CURRENT_BINARY_DIR}/subgraph193.mlir 
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/llama-import.py 
          --output-dir ${CMAKE_CURRENT_BINARY_DIR} 
  COMMENT "Generating forward.mlir, subgraph.mlir and arg0.data..."
)

add_custom_command(
  OUTPUT forward0.o
  COMMAND ${LLVM_TOOLS_BINARY_DIR}/mlir-opt ${CMAKE_CURRENT_BINARY_DIR}/forward0.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -arith-expand
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -one-shot-bufferize
            -matmul-parallel-vectorization-optimize
            -batchmatmul-optimize
            -convert-linalg-to-affine-loops
            -affine-loop-fusion
            -affine-parallelize
            -lower-affine
            -convert-scf-to-openmp
            -func-bufferize
            -arith-bufferize
            -tensor-bufferize
            -buffer-deallocation
            -finalizing-bufferize
            -convert-vector-to-scf
            -expand-strided-metadata
            -convert-vector-to-llvm
            -memref-expand
            -arith-expand
            -convert-arith-to-llvm
            -finalize-memref-to-llvm
            -convert-scf-to-cf
            -llvm-request-c-wrappers
            -convert-openmp-to-llvm
            -convert-arith-to-llvm
            -convert-math-to-llvm
            -convert-math-to-libm 
            -convert-func-to-llvm
            -reconcile-unrealized-casts |
        ${LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
        ${LLVM_TOOLS_BINARY_DIR}/llvm-as |
        ${LLVM_TOOLS_BINARY_DIR}/llc -filetype=obj -relocation-model=pic -O3
          -o ${CMAKE_CURRENT_BINARY_DIR}/forward0.o
  DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward0.mlir
  COMMENT "Building forward0.o "
  VERBATIM)

add_custom_command(
    OUTPUT subgraph0.o
    COMMAND ${LLVM_TOOLS_BINARY_DIR}/mlir-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.mlir 
              -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" |
            ${BUDDY_BINARY_DIR}/buddy-opt
            -convert-elementwise-to-linalg
            -arith-expand
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -one-shot-bufferize
            -func-bufferize-dynamic-offset
            -tensor-bufferize
            -arith-bufferize
            -buffer-deallocation
            -finalizing-bufferize
            -matmul-parallel-vectorization-optimize
            -batchmatmul-optimize
            -convert-linalg-to-affine-loops
            -affine-loop-fusion
            -affine-parallelize
            -lower-affine
            -convert-scf-to-openmp
            -convert-vector-to-scf
            -expand-strided-metadata
            -cse
            -convert-vector-to-llvm
            -memref-expand
            -arith-expand
            -convert-arith-to-llvm
            -finalize-memref-to-llvm
            -convert-scf-to-cf
            -llvm-request-c-wrappers
            -convert-openmp-to-llvm
            -convert-arith-to-llvm
            -convert-math-to-llvm
            -convert-math-to-libm 
            -convert-func-to-llvm
            -reconcile-unrealized-casts |
          ${LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
          ${LLVM_TOOLS_BINARY_DIR}/llvm-as |
          ${LLVM_TOOLS_BINARY_DIR}/llc -filetype=obj -relocation-model=pic -O3
            -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.o
    DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.mlir
    COMMENT "Building subgraph0.o "
    VERBATIM)

add_library(SPLITLLAMA0 STATIC forward0.o subgraph0.o)

add_custom_command(
  OUTPUT forward1.o
  COMMAND ${LLVM_TOOLS_BINARY_DIR}/mlir-opt ${CMAKE_CURRENT_BINARY_DIR}/forward1.mlir 
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -arith-expand
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -one-shot-bufferize
            -matmul-parallel-vectorization-optimize
            -batchmatmul-optimize
            -convert-linalg-to-affine-loops
            -affine-loop-fusion
            -affine-parallelize
            -lower-affine
            -convert-scf-to-openmp
            -func-bufferize
            -arith-bufferize
            -tensor-bufferize
            -buffer-deallocation
            -finalizing-bufferize
            -convert-vector-to-scf
            -expand-strided-metadata
            -convert-vector-to-llvm
            -memref-expand
            -arith-expand
            -convert-arith-to-llvm
            -finalize-memref-to-llvm
            -convert-scf-to-cf
            -llvm-request-c-wrappers
            -convert-openmp-to-llvm
            -convert-arith-to-llvm
            -convert-math-to-llvm
            -convert-math-to-libm 
            -convert-func-to-llvm
            -reconcile-unrealized-casts |
        ${LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
        ${LLVM_TOOLS_BINARY_DIR}/llvm-as |
        ${LLVM_TOOLS_BINARY_DIR}/llc -filetype=obj -relocation-model=pic -O3
          -o ${CMAKE_CURRENT_BINARY_DIR}/forward1.o
  DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward1.mlir
  COMMENT "Building forward1.o "
  VERBATIM)

add_custom_command(
    OUTPUT subgraph1.o
    COMMAND ${LLVM_TOOLS_BINARY_DIR}/mlir-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph1.mlir 
              -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" |
            ${BUDDY_BINARY_DIR}/buddy-opt
            -convert-elementwise-to-linalg
            -arith-expand
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -one-shot-bufferize
            -func-bufferize-dynamic-offset
            -tensor-bufferize
            -arith-bufferize
            -buffer-deallocation
            -finalizing-bufferize
            -matmul-parallel-vectorization-optimize
            -batchmatmul-optimize
            -convert-linalg-to-affine-loops
            -affine-loop-fusion
            -affine-parallelize
            -lower-affine
            -convert-scf-to-openmp
            -convert-vector-to-scf
            -expand-strided-metadata
            -cse
            -convert-vector-to-llvm
            -memref-expand
            -arith-expand
            -convert-arith-to-llvm
            -finalize-memref-to-llvm
            -convert-scf-to-cf
            -llvm-request-c-wrappers
            -convert-openmp-to-llvm
            -convert-arith-to-llvm
            -convert-math-to-llvm
            -convert-math-to-libm 
            -convert-func-to-llvm
            -reconcile-unrealized-casts |
          ${LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
          ${LLVM_TOOLS_BINARY_DIR}/llvm-as |
          ${LLVM_TOOLS_BINARY_DIR}/llc -filetype=obj -relocation-model=pic -O3
            -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph1.o
    DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph1.mlir
    COMMENT "Building subgraph1.o "
    VERBATIM)

add_library(SPLITLLAMA1 STATIC forward1.o subgraph1.o)

add_custom_command(
  OUTPUT forward2.o
  COMMAND ${LLVM_TOOLS_BINARY_DIR}/mlir-opt ${CMAKE_CURRENT_BINARY_DIR}/forward2.mlir 
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -arith-expand
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -one-shot-bufferize
            -matmul-parallel-vectorization-optimize
            -batchmatmul-optimize
            -convert-linalg-to-affine-loops
            -affine-loop-fusion
            -affine-parallelize
            -lower-affine
            -convert-scf-to-openmp
            -func-bufferize
            -arith-bufferize
            -tensor-bufferize
            -buffer-deallocation
            -finalizing-bufferize
            -convert-vector-to-scf
            -expand-strided-metadata
            -convert-vector-to-llvm
            -memref-expand
            -arith-expand
            -convert-arith-to-llvm
            -finalize-memref-to-llvm
            -convert-scf-to-cf
            -llvm-request-c-wrappers
            -convert-openmp-to-llvm
            -convert-arith-to-llvm
            -convert-math-to-llvm
            -convert-math-to-libm 
            -convert-func-to-llvm
            -reconcile-unrealized-casts |
        ${LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
        ${LLVM_TOOLS_BINARY_DIR}/llvm-as |
        ${LLVM_TOOLS_BINARY_DIR}/llc -filetype=obj -relocation-model=pic -O3
          -o ${CMAKE_CURRENT_BINARY_DIR}/forward2.o
  DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward2.mlir
  COMMENT "Building forward2.o "
  VERBATIM)

add_custom_command(
    OUTPUT subgraph2.o
    COMMAND ${LLVM_TOOLS_BINARY_DIR}/mlir-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph2.mlir 
              -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" |
            ${BUDDY_BINARY_DIR}/buddy-opt
            -convert-elementwise-to-linalg
            -arith-expand
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -one-shot-bufferize
            -func-bufferize-dynamic-offset
            -tensor-bufferize
            -arith-bufferize
            -buffer-deallocation
            -finalizing-bufferize
            -matmul-parallel-vectorization-optimize
            -batchmatmul-optimize
            -convert-linalg-to-affine-loops
            -affine-loop-fusion
            -affine-parallelize
            -lower-affine
            -convert-scf-to-openmp
            -convert-vector-to-scf
            -expand-strided-metadata
            -cse
            -convert-vector-to-llvm
            -memref-expand
            -arith-expand
            -convert-arith-to-llvm
            -finalize-memref-to-llvm
            -convert-scf-to-cf
            -llvm-request-c-wrappers
            -convert-openmp-to-llvm
            -convert-arith-to-llvm
            -convert-math-to-llvm
            -convert-math-to-libm 
            -convert-func-to-llvm
            -reconcile-unrealized-casts |
          ${LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
          ${LLVM_TOOLS_BINARY_DIR}/llvm-as |
          ${LLVM_TOOLS_BINARY_DIR}/llc -filetype=obj -relocation-model=pic -O3
            -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph2.o
    DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph2.mlir
    COMMENT "Building subgraph2.o "
    VERBATIM)

add_library(SPLITLLAMA2 STATIC forward2.o subgraph2.o)

add_custom_command(
  OUTPUT forward3.o
  COMMAND ${LLVM_TOOLS_BINARY_DIR}/mlir-opt ${CMAKE_CURRENT_BINARY_DIR}/forward3.mlir 
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -arith-expand
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -one-shot-bufferize
            -matmul-parallel-vectorization-optimize
            -batchmatmul-optimize
            -convert-linalg-to-affine-loops
            -affine-loop-fusion
            -affine-parallelize
            -lower-affine
            -convert-scf-to-openmp
            -func-bufferize
            -arith-bufferize
            -tensor-bufferize
            -buffer-deallocation
            -finalizing-bufferize
            -convert-vector-to-scf
            -expand-strided-metadata
            -convert-vector-to-llvm
            -memref-expand
            -arith-expand
            -convert-arith-to-llvm
            -finalize-memref-to-llvm
            -convert-scf-to-cf
            -llvm-request-c-wrappers
            -convert-openmp-to-llvm
            -convert-arith-to-llvm
            -convert-math-to-llvm
            -convert-math-to-libm 
            -convert-func-to-llvm
            -reconcile-unrealized-casts |
        ${LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
        ${LLVM_TOOLS_BINARY_DIR}/llvm-as |
        ${LLVM_TOOLS_BINARY_DIR}/llc -filetype=obj -relocation-model=pic -O3
          -o ${CMAKE_CURRENT_BINARY_DIR}/forward3.o
  DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward3.mlir
  COMMENT "Building forward3.o "
  VERBATIM)

add_custom_command(
    OUTPUT subgraph3.o
    COMMAND ${LLVM_TOOLS_BINARY_DIR}/mlir-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph3.mlir 
              -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" |
            ${BUDDY_BINARY_DIR}/buddy-opt
            -convert-elementwise-to-linalg
            -arith-expand
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -one-shot-bufferize
            -func-bufferize-dynamic-offset
            -tensor-bufferize
            -arith-bufferize
            -buffer-deallocation
            -finalizing-bufferize
            -matmul-parallel-vectorization-optimize
            -batchmatmul-optimize
            -convert-linalg-to-affine-loops
            -affine-loop-fusion
            -affine-parallelize
            -lower-affine
            -convert-scf-to-openmp
            -convert-vector-to-scf
            -expand-strided-metadata
            -cse
            -convert-vector-to-llvm
            -memref-expand
            -arith-expand
            -convert-arith-to-llvm
            -finalize-memref-to-llvm
            -convert-scf-to-cf
            -llvm-request-c-wrappers
            -convert-openmp-to-llvm
            -convert-arith-to-llvm
            -convert-math-to-llvm
            -convert-math-to-libm 
            -convert-func-to-llvm
            -reconcile-unrealized-casts |
          ${LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
          ${LLVM_TOOLS_BINARY_DIR}/llvm-as |
          ${LLVM_TOOLS_BINARY_DIR}/llc -filetype=obj -relocation-model=pic -O3
            -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph3.o
    DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph3.mlir
    COMMENT "Building subgraph3.o "
    VERBATIM)

add_library(SPLITLLAMA3 STATIC forward3.o subgraph3.o)

add_custom_command(
  OUTPUT forward5.o
  COMMAND ${LLVM_TOOLS_BINARY_DIR}/mlir-opt ${CMAKE_CURRENT_BINARY_DIR}/forward5.mlir 
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -arith-expand
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -one-shot-bufferize
            -matmul-parallel-vectorization-optimize
            -batchmatmul-optimize
            -convert-linalg-to-affine-loops
            -affine-loop-fusion
            -affine-parallelize
            -lower-affine
            -convert-scf-to-openmp
            -func-bufferize
            -arith-bufferize
            -tensor-bufferize
            -buffer-deallocation
            -finalizing-bufferize
            -convert-vector-to-scf
            -expand-strided-metadata
            -convert-vector-to-llvm
            -memref-expand
            -arith-expand
            -convert-arith-to-llvm
            -finalize-memref-to-llvm
            -convert-scf-to-cf
            -llvm-request-c-wrappers
            -convert-openmp-to-llvm
            -convert-arith-to-llvm
            -convert-math-to-llvm
            -convert-math-to-libm 
            -convert-func-to-llvm
            -reconcile-unrealized-casts |
        ${LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
        ${LLVM_TOOLS_BINARY_DIR}/llvm-as |
        ${LLVM_TOOLS_BINARY_DIR}/llc -filetype=obj -relocation-model=pic -O3
          -o ${CMAKE_CURRENT_BINARY_DIR}/forward5.o
  DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward5.mlir
  COMMENT "Building forward5.o "
  VERBATIM)

add_custom_command(
    OUTPUT subgraph5.o
    COMMAND ${LLVM_TOOLS_BINARY_DIR}/mlir-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph5.mlir 
              -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" |
            ${BUDDY_BINARY_DIR}/buddy-opt
            -convert-elementwise-to-linalg
            -arith-expand
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -one-shot-bufferize
            -func-bufferize-dynamic-offset
            -tensor-bufferize
            -arith-bufferize
            -buffer-deallocation
            -finalizing-bufferize
            -matmul-parallel-vectorization-optimize
            -batchmatmul-optimize
            -convert-linalg-to-affine-loops
            -affine-loop-fusion
            -affine-parallelize
            -lower-affine
            -convert-scf-to-openmp
            -convert-vector-to-scf
            -expand-strided-metadata
            -cse
            -convert-vector-to-llvm
            -memref-expand
            -arith-expand
            -convert-arith-to-llvm
            -finalize-memref-to-llvm
            -convert-scf-to-cf
            -llvm-request-c-wrappers
            -convert-openmp-to-llvm
            -convert-arith-to-llvm
            -convert-math-to-llvm
            -convert-math-to-libm 
            -convert-func-to-llvm
            -reconcile-unrealized-casts |
          ${LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
          ${LLVM_TOOLS_BINARY_DIR}/llvm-as |
          ${LLVM_TOOLS_BINARY_DIR}/llc -filetype=obj -relocation-model=pic -O3
            -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph5.o
    DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph5.mlir
    COMMENT "Building subgraph5.o "
    VERBATIM)

add_library(SPLITLLAMA4 STATIC forward5.o subgraph5.o)

add_custom_command(
  OUTPUT forward193.o
  COMMAND ${LLVM_TOOLS_BINARY_DIR}/mlir-opt ${CMAKE_CURRENT_BINARY_DIR}/forward193.mlir 
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -arith-expand
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -one-shot-bufferize
            -matmul-parallel-vectorization-optimize
            -batchmatmul-optimize
            -convert-linalg-to-affine-loops
            -affine-loop-fusion
            -affine-parallelize
            -lower-affine
            -convert-scf-to-openmp
            -func-bufferize
            -arith-bufferize
            -tensor-bufferize
            -buffer-deallocation
            -finalizing-bufferize
            -convert-vector-to-scf
            -expand-strided-metadata
            -convert-vector-to-llvm
            -memref-expand
            -arith-expand
            -convert-arith-to-llvm
            -finalize-memref-to-llvm
            -convert-scf-to-cf
            -llvm-request-c-wrappers
            -convert-openmp-to-llvm
            -convert-arith-to-llvm
            -convert-math-to-llvm
            -convert-math-to-libm 
            -convert-func-to-llvm
            -reconcile-unrealized-casts |
        ${LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
        ${LLVM_TOOLS_BINARY_DIR}/llvm-as |
        ${LLVM_TOOLS_BINARY_DIR}/llc -filetype=obj -relocation-model=pic -O3
          -o ${CMAKE_CURRENT_BINARY_DIR}/forward193.o
  DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward193.mlir
  COMMENT "Building forward193.o "
  VERBATIM)

add_custom_command(
    OUTPUT subgraph193.o
    COMMAND ${LLVM_TOOLS_BINARY_DIR}/mlir-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph193.mlir 
              -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" |
            ${BUDDY_BINARY_DIR}/buddy-opt
            -convert-elementwise-to-linalg
            -arith-expand
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -one-shot-bufferize
            -func-bufferize-dynamic-offset
            -tensor-bufferize
            -arith-bufferize
            -buffer-deallocation
            -finalizing-bufferize
            -matmul-parallel-vectorization-optimize
            -batchmatmul-optimize
            -convert-linalg-to-affine-loops
            -affine-loop-fusion
            -affine-parallelize
            -lower-affine
            -convert-scf-to-openmp
            -convert-vector-to-scf
            -expand-strided-metadata
            -cse
            -convert-vector-to-llvm
            -memref-expand
            -arith-expand
            -convert-arith-to-llvm
            -finalize-memref-to-llvm
            -convert-scf-to-cf
            -llvm-request-c-wrappers
            -convert-openmp-to-llvm
            -convert-arith-to-llvm
            -convert-math-to-llvm
            -convert-math-to-libm 
            -convert-func-to-llvm
            -reconcile-unrealized-casts |
          ${LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
          ${LLVM_TOOLS_BINARY_DIR}/llvm-as |
          ${LLVM_TOOLS_BINARY_DIR}/llc -filetype=obj -relocation-model=pic -O3
            -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph193.o
    DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph193.mlir
    COMMENT "Building subgraph193.o "
    VERBATIM)

# set(Boost_INCLUDE_DIR "/home/chenweiwei/boost_1_86_0/include")
add_library(SPLITLLAMA5 STATIC forward193.o subgraph193.o)

# 查找 Boost 库
find_package(Boost REQUIRED COMPONENTS system thread)

# 包含 Boost 头文件

include_directories(${Boost_INCLUDE_DIRS})

# 添加 websocketpp 头文件路径
include_directories(/home/chenweiwei/llvm/websocketpp)

SET_SOURCE_FILES_PROPERTIES(
  template.o
  PROPERTIES
  EXTERNAL_OBJECT true
  GENERATED true)

SET_TARGET_PROPERTIES(
  SPLITLLAMA0
  SPLITLLAMA1
  SPLITLLAMA2
  SPLITLLAMA3
  SPLITLLAMA4
  SPLITLLAMA5
  PROPERTIES
  LINKER_LANGUAGE C)

set(LLAMA_SPLIT_EXAMPLE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
set(LLAMA_EXAMPLE_BUILD_PATH ${CMAKE_CURRENT_BINARY_DIR})

set(BUDDY_SPLIT_LLAMA_LIBS
  SPLITLLAMA0
  SPLITLLAMA1
  SPLITLLAMA2
  SPLITLLAMA3
  SPLITLLAMA4
  SPLITLLAMA5
  mlir_c_runner_utils
  omp
)

add_executable(buddy-llama-input-run llama-input.cpp)

target_compile_definitions(buddy-llama-input-run PRIVATE 
  LLAMA_SPLIT_EXAMPLE_PATH="${LLAMA_SPLIT_EXAMPLE_PATH}" 
  LLAMA_EXAMPLE_BUILD_PATH="${LLAMA_EXAMPLE_BUILD_PATH}"
)

target_link_directories(buddy-llama-input-run PRIVATE ${LLVM_LIBRARY_DIR})


target_link_libraries(buddy-llama-input-run 
  PRIVATE
  ${BUDDY_SPLIT_LLAMA_LIBS}
  ${Boost_LIBRARIES} 
  pthread
  ${CMAKE_THREAD_LIBS_INIT}  # 添加线程库
)

add_executable(buddy-llama-rms-mha-0-run llama-rms-mha.cpp)

target_compile_definitions(buddy-llama-rms-mha-0-run PRIVATE 
  LLAMA_SPLIT_EXAMPLE_PATH="${LLAMA_SPLIT_EXAMPLE_PATH}" 
  LLAMA_EXAMPLE_BUILD_PATH="${LLAMA_EXAMPLE_BUILD_PATH}"
)

target_link_directories(buddy-llama-rms-mha-0-run PRIVATE ${LLVM_LIBRARY_DIR})

target_link_libraries(buddy-llama-rms-mha-0-run 
  PRIVATE
  ${BUDDY_SPLIT_LLAMA_LIBS}
  ${Boost_LIBRARIES} 
  pthread
  ${CMAKE_THREAD_LIBS_INIT}  # 添加线程库
)

add_executable(buddy-llama-rms-mha-1-run llama-rms-mha-0.cpp)

target_compile_definitions(buddy-llama-rms-mha-1-run PRIVATE 
  LLAMA_SPLIT_EXAMPLE_PATH="${LLAMA_SPLIT_EXAMPLE_PATH}" 
  LLAMA_EXAMPLE_BUILD_PATH="${LLAMA_EXAMPLE_BUILD_PATH}"
)

target_link_directories(buddy-llama-rms-mha-1-run PRIVATE ${LLVM_LIBRARY_DIR})

target_link_libraries(buddy-llama-rms-mha-1-run 
  PRIVATE
  ${BUDDY_SPLIT_LLAMA_LIBS}
  ${Boost_LIBRARIES} 
  pthread
  ${CMAKE_THREAD_LIBS_INIT}  # 添加线程库
)

add_executable(buddy-llama-mha-0-run llama-mha.cpp)

target_compile_definitions(buddy-llama-mha-0-run PRIVATE 
  LLAMA_SPLIT_EXAMPLE_PATH="${LLAMA_SPLIT_EXAMPLE_PATH}" 
  LLAMA_EXAMPLE_BUILD_PATH="${LLAMA_EXAMPLE_BUILD_PATH}"
)

target_link_directories(buddy-llama-mha-0-run PRIVATE ${LLVM_LIBRARY_DIR})

target_link_libraries(buddy-llama-mha-0-run 
  PRIVATE
  ${BUDDY_SPLIT_LLAMA_LIBS}
  ${Boost_LIBRARIES} 
  pthread
  ${CMAKE_THREAD_LIBS_INIT}  # 添加线程库
)

add_executable(buddy-llama-mha-1-run llama-mha-0.cpp)

target_compile_definitions(buddy-llama-mha-1-run PRIVATE 
  LLAMA_SPLIT_EXAMPLE_PATH="${LLAMA_SPLIT_EXAMPLE_PATH}" 
  LLAMA_EXAMPLE_BUILD_PATH="${LLAMA_EXAMPLE_BUILD_PATH}"
)

target_link_directories(buddy-llama-mha-1-run PRIVATE ${LLVM_LIBRARY_DIR})

target_link_libraries(buddy-llama-mha-1-run 
  PRIVATE
  ${BUDDY_SPLIT_LLAMA_LIBS}
  ${Boost_LIBRARIES} 
  pthread
  ${CMAKE_THREAD_LIBS_INIT}  # 添加线程库
)

add_executable(buddy-llama-add-mha-0-run llama-add-mha.cpp)

target_compile_definitions(buddy-llama-add-mha-0-run PRIVATE 
  LLAMA_SPLIT_EXAMPLE_PATH="${LLAMA_SPLIT_EXAMPLE_PATH}" 
  LLAMA_EXAMPLE_BUILD_PATH="${LLAMA_EXAMPLE_BUILD_PATH}"
)

target_link_directories(buddy-llama-add-mha-0-run PRIVATE ${LLVM_LIBRARY_DIR})

target_link_libraries(buddy-llama-add-mha-0-run 
  PRIVATE
  ${BUDDY_SPLIT_LLAMA_LIBS}
  ${Boost_LIBRARIES} 
  pthread
  ${CMAKE_THREAD_LIBS_INIT}  # 添加线程库
)

add_executable(buddy-llama-add-mha-1-run llama-add-mha-0.cpp)

target_compile_definitions(buddy-llama-add-mha-1-run PRIVATE 
  LLAMA_SPLIT_EXAMPLE_PATH="${LLAMA_SPLIT_EXAMPLE_PATH}" 
  LLAMA_EXAMPLE_BUILD_PATH="${LLAMA_EXAMPLE_BUILD_PATH}"
)

target_link_directories(buddy-llama-add-mha-1-run PRIVATE ${LLVM_LIBRARY_DIR})

target_link_libraries(buddy-llama-add-mha-1-run 
  PRIVATE
  ${BUDDY_SPLIT_LLAMA_LIBS}
  ${Boost_LIBRARIES} 
  pthread
  ${CMAKE_THREAD_LIBS_INIT}  # 添加线程库
)

add_executable(buddy-llama-rms-mlp-0-run llama-rms-mlp.cpp)

target_compile_definitions(buddy-llama-rms-mlp-0-run PRIVATE 
  LLAMA_SPLIT_EXAMPLE_PATH="${LLAMA_SPLIT_EXAMPLE_PATH}" 
  LLAMA_EXAMPLE_BUILD_PATH="${LLAMA_EXAMPLE_BUILD_PATH}"
)

target_link_directories(buddy-llama-rms-mlp-0-run PRIVATE ${LLVM_LIBRARY_DIR})

target_link_libraries(buddy-llama-rms-mlp-0-run 
  PRIVATE
  ${BUDDY_SPLIT_LLAMA_LIBS}
  ${Boost_LIBRARIES} 
  pthread
  ${CMAKE_THREAD_LIBS_INIT}  # 添加线程库
)

add_executable(buddy-llama-rms-mlp-1-run llama-rms-mlp-0.cpp)

target_compile_definitions(buddy-llama-rms-mlp-1-run PRIVATE 
  LLAMA_SPLIT_EXAMPLE_PATH="${LLAMA_SPLIT_EXAMPLE_PATH}" 
  LLAMA_EXAMPLE_BUILD_PATH="${LLAMA_EXAMPLE_BUILD_PATH}"
)

target_link_directories(buddy-llama-rms-mlp-1-run PRIVATE ${LLVM_LIBRARY_DIR})

target_link_libraries(buddy-llama-rms-mlp-1-run 
  PRIVATE
  ${BUDDY_SPLIT_LLAMA_LIBS}
  ${Boost_LIBRARIES} 
  pthread
  ${CMAKE_THREAD_LIBS_INIT}  # 添加线程库
)

add_executable( buddy-llama-mlp-0-run llama-mlp.cpp)

target_compile_definitions( buddy-llama-mlp-0-run PRIVATE 
  LLAMA_SPLIT_EXAMPLE_PATH="${LLAMA_SPLIT_EXAMPLE_PATH}" 
  LLAMA_EXAMPLE_BUILD_PATH="${LLAMA_EXAMPLE_BUILD_PATH}"
)

target_link_directories( buddy-llama-mlp-0-run PRIVATE ${LLVM_LIBRARY_DIR})

target_link_libraries( buddy-llama-mlp-0-run 
  PRIVATE
  ${BUDDY_SPLIT_LLAMA_LIBS}
  ${Boost_LIBRARIES} 
  pthread
  ${CMAKE_THREAD_LIBS_INIT}  # 添加线程库
)

add_executable( buddy-llama-mlp-1-run llama-mlp-0.cpp)

target_compile_definitions( buddy-llama-mlp-1-run PRIVATE 
  LLAMA_SPLIT_EXAMPLE_PATH="${LLAMA_SPLIT_EXAMPLE_PATH}" 
  LLAMA_EXAMPLE_BUILD_PATH="${LLAMA_EXAMPLE_BUILD_PATH}"
)

target_link_directories( buddy-llama-mlp-1-run PRIVATE ${LLVM_LIBRARY_DIR})

target_link_libraries( buddy-llama-mlp-1-run 
  PRIVATE
  ${BUDDY_SPLIT_LLAMA_LIBS}
  ${Boost_LIBRARIES} 
  pthread
  ${CMAKE_THREAD_LIBS_INIT}  # 添加线程库
)

add_executable( buddy-llama-add-mlp-0-run llama-add-mlp.cpp)

target_compile_definitions( buddy-llama-add-mlp-0-run PRIVATE 
  LLAMA_SPLIT_EXAMPLE_PATH="${LLAMA_SPLIT_EXAMPLE_PATH}" 
  LLAMA_EXAMPLE_BUILD_PATH="${LLAMA_EXAMPLE_BUILD_PATH}"
)

target_link_directories( buddy-llama-add-mlp-0-run PRIVATE ${LLVM_LIBRARY_DIR})

target_link_libraries( buddy-llama-add-mlp-0-run 
  PRIVATE
  ${BUDDY_SPLIT_LLAMA_LIBS}
  ${Boost_LIBRARIES} 
  pthread
  ${CMAKE_THREAD_LIBS_INIT}  # 添加线程库
)

add_executable( buddy-llama-add-mlp-1-run llama-add-mlp-0.cpp)

target_compile_definitions( buddy-llama-add-mlp-1-run PRIVATE 
  LLAMA_SPLIT_EXAMPLE_PATH="${LLAMA_SPLIT_EXAMPLE_PATH}" 
  LLAMA_EXAMPLE_BUILD_PATH="${LLAMA_EXAMPLE_BUILD_PATH}"
)

target_link_directories( buddy-llama-add-mlp-1-run PRIVATE ${LLVM_LIBRARY_DIR})

target_link_libraries( buddy-llama-add-mlp-1-run 
  PRIVATE
  ${BUDDY_SPLIT_LLAMA_LIBS}
  ${Boost_LIBRARIES} 
  pthread
  ${CMAKE_THREAD_LIBS_INIT}  # 添加线程库
)

add_executable(buddy-llama-output-run llama-output.cpp)

target_compile_definitions(buddy-llama-output-run PRIVATE 
  LLAMA_SPLIT_EXAMPLE_PATH="${LLAMA_SPLIT_EXAMPLE_PATH}" 
  LLAMA_EXAMPLE_BUILD_PATH="${LLAMA_EXAMPLE_BUILD_PATH}"
)

target_link_directories(buddy-llama-output-run PRIVATE ${LLVM_LIBRARY_DIR})


target_link_libraries(buddy-llama-output-run 
  PRIVATE
  ${BUDDY_SPLIT_LLAMA_LIBS}
  ${Boost_LIBRARIES} 
  pthread
  ${CMAKE_THREAD_LIBS_INIT}  # 添加线程库
)

find_package(MPI REQUIRED)

add_executable(buddy-llama-main-run llama-main.cpp)

target_compile_definitions(buddy-llama-main-run PRIVATE 
  LLAMA_SPLIT_EXAMPLE_PATH="${LLAMA_SPLIT_EXAMPLE_PATH}" 
  LLAMA_EXAMPLE_BUILD_PATH="${LLAMA_EXAMPLE_BUILD_PATH}"
)

target_link_directories(buddy-llama-main-run 
  PRIVATE 
  ${LLVM_LIBRARY_DIR}
  ${MPI_INCLUDE_PATH}
  )


target_link_libraries(buddy-llama-main-run 
  PRIVATE
  ${BUDDY_SPLIT_LLAMA_LIBS}
  ${MPI_CXX_LIBRARIES}
  numa
  pthread
  dl
  rt
)
target_link_options(buddy-llama-main-run PRIVATE)
