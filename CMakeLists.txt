cmake_minimum_required(VERSION 3.12)

project(pathtracing_in_5minutes LANGUAGES C CXX)

add_executable(main main.cpp)

target_compile_features(main PUBLIC cxx_std_17)
set_target_properties(main PROPERTIES CXX_EXTENSIONS OFF)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(main PUBLIC OpenMP::OpenMP_CXX)
endif()