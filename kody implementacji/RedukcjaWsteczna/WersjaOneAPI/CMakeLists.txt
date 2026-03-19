cmake_minimum_required(VERSION 3.23)
project(RedukcjaWsteczna LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(RedukcjaWsteczna RedukcjaWsteczna.cpp)

target_compile_options(RedukcjaWsteczna PRIVATE -fsycl)
target_link_options(RedukcjaWsteczna PRIVATE -fsycl)