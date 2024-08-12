cmake_minimum_required(VERSION 3.10)

set(CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72 75 86 89 90)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

project(depth_anything_cpp LANGUAGES CXX CUDA)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Release)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

# CUDA
find_package(CUDA REQUIRED)
message(STATUS "CUDA Libs: \n${CUDA_LIBRARIES}\n")
get_filename_component(CUDA_LIB_DIR ${CUDA_LIBRARIES} DIRECTORY)
message(STATUS "CUDA Headers: \n${CUDA_INCLUDE_DIRS}\n")


# OpenCV
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV Libs: \n${OpenCV_LIBRARIES}\n") 
message(STATUS "OpenCV Headers: \n${OpenCV_INCLUDE_DIRS}\n")


# RealSense2
find_package(realsense2 REQUIRED)
message(STATUS "RealSense Libs: \n${realsense2_LIBRARY}\n")
message(STATUS "RealSense Headers: \n${realsense2_INCLUDE_DIR}\n")

# TensorRT
set(TensorRT_INCLUDE_DIRS /usr/include /usr/include/aarch-linux-gnu)
set(TensorRT_LIBRARIES /usr/lib/aarch64-linux-gnu)

message(STATUS "TensorRT Libs: \n\n${TensorRT_LIBRARIES}\n")
message(STATUS "TensorRT Headers: \n${TensorRT_INCLUDE_DIRS}\n")

list(APPEND INCLUDE_DIRS
        ${CUDA_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${TensorRT_INCLUDE_DIRS}
        ${realsense2_INCLUDE_DIR}
        include
        )

list(APPEND ALL_LIBS
        ${CUDA_LIBRARIES}
        ${CUDA_LIB_DIR}
        ${OpenCV_LIBRARIES}
        ${TensorRT_LIBRARIES}
        ${realsense2_LIBRARY}
        )

include_directories(${INCLUDE_DIRS}) # 添加头文件搜索路径，用于编译阶段

add_executable(${PROJECT_NAME}
        src/main.cpp
        src/depth_anything.cpp
        src/utils.cpp
        include/depth_anything.hpp
        include/utils.hpp
        )

target_link_directories(${PROJECT_NAME} PUBLIC ${ALL_LIBS}) # 添加链接库搜索路径，用于链接阶段
target_link_libraries(${PROJECT_NAME} PRIVATE nvinfer nvinfer_plugin cudart ${OpenCV_LIBS} ${realsense2_LIBRARY})