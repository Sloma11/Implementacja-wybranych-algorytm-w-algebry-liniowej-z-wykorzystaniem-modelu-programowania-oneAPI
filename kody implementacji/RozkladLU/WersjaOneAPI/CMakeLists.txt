cmake_minimum_required(VERSION 3.23)
project(Rozklad_LU LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(Rozklad_LU Rozklad_LU.cpp)

target_compile_options(Rozklad_LU PRIVATE -fsycl)
target_link_options(Rozklad_LU PRIVATE -fsycl)