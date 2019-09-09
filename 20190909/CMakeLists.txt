cmake_minimum_required(VERSION 3.5)

# 工程信息
project(DIP)

include_directories(${CMAKE_SOURCE_DIR}/..)

add_subdirectory(8_Harris)
add_subdirectory(8_Harris2)
add_subdirectory(9_Sobel)
add_subdirectory(10_Shi-Tomasi)
add_subdirectory(13_SRUF)
add_subdirectory(14_LOG)
add_subdirectory(15_CamCali)
add_subdirectory(16_TwoDimGamma)
add_subdirectory(17_AHE)

