cmake_minimum_required(VERSION 3.20)
project(fusionpass LANGUAGES CXX C)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Honor LLVM_PREFIX so configure matches Phase 1 without always passing -DMLIR_DIR.
if(NOT MLIR_DIR AND DEFINED ENV{LLVM_PREFIX} AND NOT "$ENV{LLVM_PREFIX}" STREQUAL "")
  file(TO_CMAKE_PATH "$ENV{LLVM_PREFIX}" _llvm_prefix)
  set(MLIR_DIR "${_llvm_prefix}/lib/cmake/mlir" CACHE PATH "Path to MLIRConfig.cmake")
  set(LLVM_DIR "${_llvm_prefix}/lib/cmake/llvm" CACHE PATH "Path to LLVMConfig.cmake")
endif()

find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "LLVM package version: ${LLVM_PACKAGE_VERSION}")
message(STATUS "CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER}")

if(NOT LLVM_VERSION_MAJOR EQUAL 18)
  message(FATAL_ERROR
    "fusionpass requires LLVM/MLIR 18.1.x (found ${LLVM_PACKAGE_VERSION}). "
    "Do not upgrade LLVM; pattern/pass APIs differ across releases.")
endif()

# Mirror mlir/examples/standalone (llvmorg-18.1.8): set output dirs BEFORE AddLLVM.
set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)   # sets -fno-rtti etc. to match the LLVM build

include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include ${PROJECT_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)
