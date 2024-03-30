project(fm)
set(CMAKE_CXX_STANDARD 17)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../pybind11 pybind_build)
pybind11_add_module(fm fm.cpp)