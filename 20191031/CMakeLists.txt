cmake_minimum_required(VERSION 3.5)

# 工程信息
project(DropX)

include_directories(${CMAKE_SOURCE_DIR}/..)

add_subdirectory(SiftStitch)
add_subdirectory(FieldCurveDeblur)
