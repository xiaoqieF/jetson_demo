cmake_minimum_required(VERSION 3.16)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-13.2/bin/nvcc" CACHE FILEPATH "CUDA compiler")
set(CMAKE_CUDA_ARCHITECTURES 87 CACHE STRING "CUDA architectures")
project(argus_inference LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(ARGUS_ROOT "/usr/src/jetson_multimedia_api/argus" CACHE PATH "LibArgus root")
set(ARGUS_INCLUDE_DIR "${ARGUS_ROOT}/include" CACHE PATH "LibArgus include directory")
set(JETSON_MMAPI_ROOT "/usr/src/jetson_multimedia_api" CACHE PATH "Jetson Multimedia API root")
set(JETSON_MMAPI_INCLUDE_DIR "${JETSON_MMAPI_ROOT}/include" CACHE PATH "Jetson Multimedia API include")

find_package(ament_cmake REQUIRED)
find_package(Threads REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc dnn)
find_package(argus_interfaces REQUIRED)
find_package(argus_transport REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_path(ARGUS_INCLUDE_DIR Argus/Argus.h HINTS "${ARGUS_INCLUDE_DIR}" REQUIRED)
find_path(JETSON_MMAPI_INCLUDE_DIR NvBufSurface.h HINTS "${JETSON_MMAPI_INCLUDE_DIR}" REQUIRED)
find_library(NVINFER_LIBRARY nvinfer REQUIRED)
find_library(NVBUFSURFACE_LIBRARY nvbufsurface HINTS /usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}/nvidia /usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}/tegra REQUIRED)
find_library(NVBUFSURFTRANSFORM_LIBRARY nvbufsurftransform HINTS /usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}/nvidia /usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}/tegra REQUIRED)
find_library(CUDA_DRIVER_LIBRARY cuda HINTS /opt/nvidia/l4t-gpu-libs/nvgpu /usr/local/cuda/lib64 REQUIRED)
find_library(V4L2_LIBRARY v4l2 REQUIRED)

set(JETSON_MMAPI_NVBUFSURFACE_SOURCE "${JETSON_MMAPI_ROOT}/samples/common/classes/NvBufSurface.cpp")
if(NOT EXISTS "${JETSON_MMAPI_NVBUFSURFACE_SOURCE}")
  message(FATAL_ERROR "Jetson Multimedia API NvBufSurface source is missing")
endif()

add_library(${PROJECT_NAME} SHARED
  src/inference_node.cpp
  src/cuda_preprocess.cu
  src/yolov8_segmentation.cpp
  ${JETSON_MMAPI_NVBUFSURFACE_SOURCE})
target_include_directories(${PROJECT_NAME} PRIVATE
  "${ARGUS_INCLUDE_DIR}" "${JETSON_MMAPI_INCLUDE_DIR}" /usr/local/cuda/include)
target_include_directories(${PROJECT_NAME} PUBLIC include)
ament_target_dependencies(${PROJECT_NAME} argus_interfaces argus_transport rclcpp rclcpp_components)
target_link_libraries(${PROJECT_NAME}
  ${OpenCV_LIBS}
  "${NVINFER_LIBRARY}" CUDA::cudart
  "${NVBUFSURFACE_LIBRARY}" "${NVBUFSURFTRANSFORM_LIBRARY}" "${CUDA_DRIVER_LIBRARY}"
  "${V4L2_LIBRARY}" Threads::Threads)
rclcpp_components_register_nodes(${PROJECT_NAME} "argus_inference::InferenceNode")

install(TARGETS ${PROJECT_NAME} DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME})
ament_package()
