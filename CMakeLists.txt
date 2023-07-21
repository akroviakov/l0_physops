function(add_dpcpp_lib name)
  message(STATUS "Adding dpcpp library ${name}")
  set(sources ${ARGN})
  add_library(${name} SHARED ${sources})
  target_include_directories(${name} PRIVATE ${PROJECT_SOURCE_DIR})
  target_compile_options(${name} PRIVATE -fsycl -O3)
  target_link_options(${name} PRIVATE -fsycl)
endfunction()

function(add_dpcpp_exec name source)
  add_executable(${name} ${source})
  target_include_directories(${name} PRIVATE ${PROJECT_SOURCE_DIR})
  target_compile_options(${name} PRIVATE -fsycl)
  target_link_options(${name} PRIVATE -fsycl)
endfunction()

cmake_minimum_required(VERSION 3.17)
set(CMAKE_CXX_COMPILER "icpx")

project(l0_physops LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

add_subdirectory(hash_table)

add_dpcpp_exec(main main.cpp)
