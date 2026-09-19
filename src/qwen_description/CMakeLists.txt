cmake_minimum_required(VERSION 3.16)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-13.2/bin/nvcc" CACHE FILEPATH "CUDA compiler")
set(CMAKE_CUDA_ARCHITECTURES 87 CACHE STRING "CUDA architectures")
project(qwen_description LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(EDGE_LLM_ROOT "/home/royfan/edge-llm-v0.10.1" CACHE PATH "TensorRT Edge-LLM source tree")
set(EDGE_LLM_BUILD "${EDGE_LLM_ROOT}/build" CACHE PATH "TensorRT Edge-LLM build tree")
set(ARGUS_ROOT "/usr/src/jetson_multimedia_api/argus" CACHE PATH "LibArgus root")
set(JETSON_MMAPI_ROOT "/usr/src/jetson_multimedia_api" CACHE PATH "Jetson Multimedia API root")

find_package(ament_cmake REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
find_package(argus_interfaces REQUIRED)
find_package(argus_transport REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_path(ARGUS_INCLUDE_DIR Argus/Argus.h HINTS "${ARGUS_ROOT}/include" REQUIRED)
find_path(JETSON_MMAPI_INCLUDE_DIR NvBufSurface.h HINTS "${JETSON_MMAPI_ROOT}/include" REQUIRED)
find_library(NVINFER_LIBRARY nvinfer REQUIRED)
find_library(NVONNXPARSER_LIBRARY nvonnxparser REQUIRED)
find_library(NVBUFSURFACE_LIBRARY nvbufsurface
  HINTS /usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}/nvidia /usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}/tegra REQUIRED)
find_library(NVBUFSURFTRANSFORM_LIBRARY nvbufsurftransform
  HINTS /usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}/nvidia /usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}/tegra REQUIRED)
find_library(CUDA_DEVICE_RUNTIME_LIBRARY cudadevrt
  HINTS
    "${CMAKE_CUDA_COMPILER_TOOLKIT_ROOT}/targets/sbsa-linux/lib"
    "${CMAKE_CUDA_COMPILER_TOOLKIT_ROOT}/lib64"
    "${CMAKE_CUDA_COMPILER}/../targets/sbsa-linux/lib"
  REQUIRED)
find_library(EDGE_LLM_CUTEDSL_LIBRARY cutedsl_aarch64
  HINTS
    "${EDGE_LLM_ROOT}/cpp/kernels/cuteDSLArtifact/aarch64/sm_${CMAKE_CUDA_ARCHITECTURES}"
  REQUIRED)

set(EDGE_LLM_COMMON_INCLUDES
  "${EDGE_LLM_ROOT}/cpp"
  "${EDGE_LLM_ROOT}/examples/multimodal"
  "${EDGE_LLM_ROOT}/3rdParty/nlohmannJson/include"
  "${EDGE_LLM_ROOT}/3rdParty/stb"
  "${EDGE_LLM_ROOT}/3rdParty/miniaudio")

add_library(${PROJECT_NAME} SHARED
  src/qwen_description_node.cpp
  "${JETSON_MMAPI_ROOT}/samples/common/classes/NvBufSurface.cpp")
target_include_directories(${PROJECT_NAME} PRIVATE
  include "${ARGUS_INCLUDE_DIR}" "${JETSON_MMAPI_INCLUDE_DIR}" ${EDGE_LLM_COMMON_INCLUDES})
ament_target_dependencies(${PROJECT_NAME}
  argus_interfaces argus_transport rclcpp rclcpp_components sensor_msgs)
target_link_libraries(${PROJECT_NAME}
  "${EDGE_LLM_BUILD}/cpp/libedgellmCore.a"
  "${EDGE_LLM_CUTEDSL_LIBRARY}"
  ${OpenCV_LIBS}
  "${NVINFER_LIBRARY}" "${NVONNXPARSER_LIBRARY}"
  CUDA::cudart CUDA::cuda_driver "${CUDA_DEVICE_RUNTIME_LIBRARY}" dl rt pthread
  "${NVBUFSURFACE_LIBRARY}" "${NVBUFSURFTRANSFORM_LIBRARY}")
set_target_properties(${PROJECT_NAME} PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_options(${PROJECT_NAME} PRIVATE
  "-Wl,--unresolved-symbols=ignore-in-shared-libs")
rclcpp_components_register_nodes(${PROJECT_NAME} "qwen_description::QwenDescriptionNode")

install(TARGETS ${PROJECT_NAME} DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME})
ament_package()
