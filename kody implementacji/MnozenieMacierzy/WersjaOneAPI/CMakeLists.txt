cmake_minimum_required(VERSION 3.23)
project(MnozenieMacierzy LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(MnozenieMacierzy MnozenieMacierzy.cpp)

target_compile_options(MnozenieMacierzy PRIVATE -fsycl)
target_link_options(MnozenieMacierzy PRIVATE -fsycl)