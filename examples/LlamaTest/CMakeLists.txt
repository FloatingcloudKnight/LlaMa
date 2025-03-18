add_custom_command(
  OUTPUT forward.o
  COMMAND ${LLVM_TOOLS_BINARY_DIR}/mlir-opt ${CMAKE_CURRENT_BINARY_DIR}/forward.mlir 
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
          -o ${CMAKE_CURRENT_BINARY_DIR}/forward.o
  DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward.mlir
  COMMENT "Building forward.o "
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

add_library(DISLLAMA0 STATIC forward0.o subgraph0.o)

add_custom_command(
    OUTPUT subgraph1.o
    COMMAND ${LLVM_TOOLS_BINARY_DIR}/mlir-opt ${CMAKE_CURRENT_SOURCE_DIR}/subgraph1.mlir 
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
            -o ${CMAKE_CURRENT_SOURCE_DIR}/subgraph1.o
    DEPENDS buddy-opt ${CMAKE_CURRENT_SOURCE_DIR}/fake-subgraph1.mlir
    COMMENT "Building subgraph1.o "
    VERBATIM)

add_library(DISLLAMA1 STATIC forward1.o subgraph1.o)

add_custom_command(
  OUTPUT forward2.o
  COMMAND ${LLVM_TOOLS_BINARY_DIR}/mlir-opt ${CMAKE_CURRENT_SOURCE_DIR}/fake-forward2.mlir 
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
          -o ${CMAKE_CURRENT_SOURCE_DIR}/forward2.o
  DEPENDS buddy-opt ${CMAKE_CURRENT_SOURCE_DIR}/fake-forward2.mlir
  COMMENT "Building forward2.o "
  VERBATIM)

add_custom_command(
    OUTPUT subgraph2.o
    COMMAND ${LLVM_TOOLS_BINARY_DIR}/mlir-opt ${CMAKE_CURRENT_SOURCE_DIR}/fake-subgraph2.mlir 
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
            -o ${CMAKE_CURRENT_SOURCE_DIR}/subgraph2.o
    DEPENDS buddy-opt ${CMAKE_CURRENT_SOURCE_DIR}/fake-subgraph2.mlir
    COMMENT "Building subgraph2.o "
    VERBATIM)

add_library(DISLLAMA2 STATIC forward2.o subgraph2.o)

add_custom_command(
  OUTPUT forward3.o
  COMMAND ${LLVM_TOOLS_BINARY_DIR}/mlir-opt ${CMAKE_CURRENT_SOURCE_DIR}/fake-forward3.mlir 
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
          -o ${CMAKE_CURRENT_SOURCE_DIR}/forward3.o
  DEPENDS buddy-opt ${CMAKE_CURRENT_SOURCE_DIR}/fake-forward3.mlir
  COMMENT "Building forward3.o "
  VERBATIM)

add_custom_command(
    OUTPUT subgraph3.o
    COMMAND ${LLVM_TOOLS_BINARY_DIR}/mlir-opt ${CMAKE_CURRENT_SOURCE_DIR}/fake-subgraph3.mlir 
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
            -o ${CMAKE_CURRENT_SOURCE_DIR}/subgraph3.o
    DEPENDS buddy-opt ${CMAKE_CURRENT_SOURCE_DIR}/fake-subgraph3.mlir
    COMMENT "Building subgraph3.o "
    VERBATIM)

add_library(DISLLAMA3 STATIC forward3.o subgraph3.o)

add_custom_command(
  OUTPUT forward4.o
  COMMAND ${LLVM_TOOLS_BINARY_DIR}/mlir-opt ${CMAKE_CURRENT_SOURCE_DIR}/fake-forward4.mlir 
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
          -o ${CMAKE_CURRENT_SOURCE_DIR}/forward4.o
  DEPENDS buddy-opt ${CMAKE_CURRENT_SOURCE_DIR}/fake-forward4.mlir
  COMMENT "Building forward4.o "
  VERBATIM)

add_custom_command(
    OUTPUT subgraph4.o
    COMMAND ${LLVM_TOOLS_BINARY_DIR}/mlir-opt ${CMAKE_CURRENT_SOURCE_DIR}/fake-subgraph4.mlir 
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
            -o ${CMAKE_CURRENT_SOURCE_DIR}/subgraph4.o
    DEPENDS buddy-opt ${CMAKE_CURRENT_SOURCE_DIR}/fake-subgraph4.mlir
    COMMENT "Building subgraph4.o "
    VERBATIM)

add_library(DISLLAMA4 STATIC forward4.o subgraph4.o)

add_custom_command(
  OUTPUT forward5.o
  COMMAND ${LLVM_TOOLS_BINARY_DIR}/mlir-opt ${CMAKE_CURRENT_SOURCE_DIR}/fake-forward5.mlir 
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
          -o ${CMAKE_CURRENT_SOURCE_DIR}/forward5.o
  DEPENDS buddy-opt ${CMAKE_CURRENT_SOURCE_DIR}/fake-forward5.mlir
  COMMENT "Building forward5.o "
  VERBATIM)

add_custom_command(
    OUTPUT subgraph5.o
    COMMAND ${LLVM_TOOLS_BINARY_DIR}/mlir-opt ${CMAKE_CURRENT_SOURCE_DIR}/fake-subgraph5.mlir 
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
            -o ${CMAKE_CURRENT_SOURCE_DIR}/subgraph5.o
    DEPENDS buddy-opt ${CMAKE_CURRENT_SOURCE_DIR}/fake-subgraph5.mlir
    COMMENT "Building subgraph5.o "
    VERBATIM)

add_library(DISLLAMA5 STATIC forward5.o subgraph5.o)

SET_SOURCE_FILES_PROPERTIES(
  template.o
  PROPERTIES
  EXTERNAL_OBJECT true
  GENERATED true)

SET_TARGET_PROPERTIES(
  DISLLAMA0
  DISLLAMA1
  DISLLAMA2
  DISLLAMA3
  DISLLAMA4
  DISLLAMA5
  PROPERTIES
  LINKER_LANGUAGE C)

add_executable(buddy-dis-llama-run llama-main.cpp)

set(LLAMA_DIS_EXAMPLE_PATH ${CMAKE_CURRENT_SOURCE_DIR})

target_compile_definitions(buddy-dis-llama-run PRIVATE 
  LLAMA_DIS_EXAMPLE_PATH="${LLAMA_DIS_EXAMPLE_PATH}" 
)

target_link_directories(buddy-dis-llama-run PRIVATE ${LLVM_LIBRARY_DIR})

set(BUDDY_DIS_LLAMA_LIBS
  DISLLAMA0
  DISLLAMA1
  DISLLAMA2
  DISLLAMA3
  DISLLAMA4
  DISLLAMA5
  mlir_c_runner_utils
  omp
)

target_link_libraries(buddy-dis-llama-run ${BUDDY_DIS_LLAMA_LIBS})
