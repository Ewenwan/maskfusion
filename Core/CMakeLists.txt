
# cmake 版本要求
cmake_minimum_required(VERSION 3.0.2)

# policy to export a different link interface
# 对于一个指定的策略，那些依赖于旧行为的工程，通过设置策略的状态为OLD，可以禁止策略的警告。
cmake_policy(SET CMP0022 OLD)

# 项目名称
project(libmaskfusion)

message("Evaluating Core/CMAKE")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}")

## 外部依赖库  External packages
find_package(CUDA REQUIRED) # GPU
# 大型稀疏矩阵运算库SuiteSparse
find_package(SuiteSparse REQUIRED)
find_package(OpenCV 3.1 REQUIRED)
find_package(OpenMP)# 多线程
find_package(PythonLibs)

# 目标检测语义分割
set(MASKFUSION_MASK_RCNN_DIR ${CMAKE_SOURCE_DIR}/deps/Mask_RCNN CACHE STRING "Location of MaskRCNN")
set(MASKFUSION_PYTHON_VE_PATH "" CACHE STRING "Python virtual environment to use")
configure_file("Segmentation/MaskRCNN/MaskRCNN.py.in" "${CMAKE_CURRENT_BINARY_DIR}/../GUI/MaskRCNN.py")
configure_file("Segmentation/MaskRCNN/helpers.py" "${CMAKE_CURRENT_BINARY_DIR}/../GUI/helpers.py")

if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

## Options
set(MASKFUSION_GPUS_MASKRCNN "0" CACHE STRING "Select GPU which is used by MaskRCNN")
set(MASKFUSION_GPU_SLAM "-1" CACHE STRING "Select GPU which is running SLAM, this has to be the OpenGL GPU.")
set(MASKFUSION_NUM_GSURFELS "9437184" CACHE STRING "Maximum number of surfels for background")
set(MASKFUSION_NUM_OSURFELS "1048576" CACHE STRING "Maximum number of surfels per object")
add_definitions(-DMASKFUSION_GPUS_MASKRCNN=${MASKFUSION_GPUS_MASKRCNN})
add_definitions(-DMASKFUSION_GPU_SLAM=${MASKFUSION_GPU_SLAM})
add_definitions(-DMASKFUSION_NUM_GSURFELS=${MASKFUSION_NUM_GSURFELS})
add_definitions(-DMASKFUSION_NUM_OSURFELS=${MASKFUSION_NUM_OSURFELS})

# Show debug visualisations
option(SHOW_DEBUG_VISUALISATION "Show debug visualisation" OFF)
if(SHOW_DEBUG_VISUALISATION)
    add_definitions(-DSHOW_DEBUG_VISUALISATION)
endif()

## Directories 条件随机场 库=========
set(DENSECRF_DIR "${CMAKE_SOURCE_DIR}/deps/densecrf")
set(DENSECRF_LIBRARY_DIRS "${DENSECRF_DIR}/build/src/")
set(DENSECRF_INCLUDE_DIRS "${DENSECRF_DIR}/include")
set(DENSECRF_LIBRARIES densecrf)

# 可视化颜色=========================
set(maskfusion_SHADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/Shaders" CACHE PATH "Where the shaders live")

message("Using: ")
message("=======")
message("Using DENSECRF_DIR: ${DENSECRF_DIR}")
message("Using maskfusion_SHADER_DIR: ${maskfusion_SHADER_DIR}")
message("Using python:  ${PYTHONLIBS_VERSION_STRING} ")
message("Using PYTHON_LIBRARIES:  ${PYTHON_LIBRARIES} ")
message("Using PYTHON_INCLUDE_DIRS:  ${PYTHON_INCLUDE_DIRS} ")
message("=======")

## Includes  库文件
include_directories(${Pangolin_INCLUDE_DIRS}) # Pangolin 可视化界面
include_directories(${CUDA_INCLUDE_DIRS})     # CUDA  GPU
include_directories(${EIGEN_INCLUDE_DIRS})    # eigen 矩阵运算
include_directories(${SUITESPARSE_INCLUDE_DIRS})# 大型稀疏矩阵运算库SuiteSparse
include_directories(${DENSECRF_INCLUDE_DIRS}) # 条件随机场
include_directories(${PCL_INCLUDE_DIRS})      # PCL 点云库
include_directories(${OpenCV_INCLUDE_DIRS})   # opencv 2d图像处理等

link_directories(${PCL_LIBRARY_DIRS})      # 链接库
link_directories(${DENSECRF_LIBRARY_DIRS})


# 项目源文件=========================================================
file(GLOB srcs *.cpp)
file(GLOB model_srcs Model/*.cpp)
file(GLOB utils_srcs Utils/*.cpp)
file(GLOB shader_srcs Shaders/*.cpp)
file(GLOB cuda Cuda/*.cu)
file(GLOB containers Cuda/containers/*.cpp)
file(GLOB segmentation Segmentation/*.cpp Segmentation/MaskRCNN/*.cpp)

set(CUDA_ARCH_BIN "50 52 61" CACHE STRING "Specify 'real' GPU arch to build binaries for, BIN(PTX) format is supported. Example: 1.3 2.1(1.3) or 13 21(13)")
set(CUDA_ARCH_PTX "" CACHE STRING "Specify 'virtual' PTX arch to build PTX intermediate code for. Example: 1.0 1.2 or 10 12")

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR})
include(CudaComputeTargetFlags.cmake)
APPEND_TARGET_ARCH_FLAGS()

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}  "-Xcompiler;-fPIC;-D_FORCE_INLINES;")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "--ftz=true;--prec-div=false;--prec-sqrt=false")

if(CMAKE_BUILD_TYPE MATCHES Debug)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}  "-g")
endif(CMAKE_BUILD_TYPE MATCHES Debug)

CUDA_COMPILE(cuda_objs ${cuda})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -msse -msse2 -msse3 -DSHADER_DIR=${maskfusion_SHADER_DIR}")

if(CMAKE_BUILD_TYPE MATCHES Debug)
    message("Debug build.")
else(CMAKE_BUILD_TYPE MATCHES Release)
    message("Release build.")
    add_compile_options(-O3)
    add_compile_options(-Wno-maybe-uninitialized)
endif(CMAKE_BUILD_TYPE MATCHES Debug)


# 添加动态链接库=============================
add_library(${PROJECT_NAME} SHARED
            ${srcs}
            ${model_srcs}
            ${utils_srcs}
            ${shader_srcs}
            ${cuda}
            ${cuda_objs}
            ${containers}
            ${segmentation}
)

# 库依赖的文件
target_include_directories(${PROJECT_NAME} PUBLIC
                            ${CMAKE_CURRENT_SOURCE_DIR}
                            ${GSLICR_DIR}/gSLICr_Lib
                            ${ZLIB_INCLUDE_DIR}
                            ${Pangolin_INCLUDE_DIRS}
                            ${CUDA_INCLUDE_DIRS}
                            ${OPENNI2_INCLUDE_DIR}
                            ${OpenCV_INCLUDE_DIRS}
                            ${BOOST_INCLUDE_DIRS}
                            ${PYTHON_INCLUDE_DIRS}
)
# 动态链接时
target_link_libraries(${PROJECT_NAME} PRIVATE
                      ${DENSECRF_LIBRARIES}
                      ${Eigen_LIBRARIES}
                      ${Pangolin_LIBRARIES}
                      ${CUDA_LIBRARIES}
                      ${SUITESPARSE_LIBRARIES}
                      ${OpenCV_LIBRARIES}
                      gSLICr_lib
                      ${PYTHON_LIBRARIES}
)
# 会把库附加在LINK_INTERFACE_LIBRARIES以及LINK_INTERFACE_LIBRARIES在不同配置下的等价目标属性，而不是用这些库去链接。
set_target_properties(${PROJECT_NAME} PROPERTIES LINK_INTERFACE_LIBRARIES "")



target_include_directories(${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

export(PACKAGE ${PROJECT_NAME})


export(TARGETS ${PROJECT_NAME}
    FILE "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Targets.cmake"
)

configure_file(
    "../CMakeModules/${PROJECT_NAME}Config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
    COPYONLY
)
