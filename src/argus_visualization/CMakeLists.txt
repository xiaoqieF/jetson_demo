cmake_minimum_required(VERSION 3.16)
project(argus_visualization LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(ARGUS_ROOT "/usr/src/jetson_multimedia_api/argus" CACHE PATH "LibArgus root")
set(ARGUS_INCLUDE_DIR "${ARGUS_ROOT}/include" CACHE PATH "LibArgus include directory")
set(JETSON_MMAPI_ROOT "/usr/src/jetson_multimedia_api" CACHE PATH "Jetson Multimedia API root")
set(JETSON_MMAPI_INCLUDE_DIR "${JETSON_MMAPI_ROOT}/include" CACHE PATH "Jetson Multimedia API include")

find_package(ament_cmake REQUIRED)
find_package(Threads REQUIRED)
find_package(argus_interfaces REQUIRED)
find_package(argus_transport REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_path(ARGUS_INCLUDE_DIR Argus/Argus.h HINTS "${ARGUS_INCLUDE_DIR}" REQUIRED)
find_path(JETSON_MMAPI_INCLUDE_DIR NvJpegEncoder.h HINTS "${JETSON_MMAPI_INCLUDE_DIR}" REQUIRED)
find_library(NVMM_JPEG_LIBRARY nvmm_jpeg HINTS /usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}/nvidia /usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}/tegra REQUIRED)
find_library(NVBUFSURFACE_LIBRARY nvbufsurface HINTS /usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}/nvidia /usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}/tegra REQUIRED)
find_library(NVBUFSURFTRANSFORM_LIBRARY nvbufsurftransform HINTS /usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}/nvidia /usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}/tegra REQUIRED)
find_library(CUDA_DRIVER_LIBRARY cuda HINTS /opt/nvidia/l4t-gpu-libs/nvgpu /usr/local/cuda/lib64 REQUIRED)
find_library(V4L2_LIBRARY v4l2 REQUIRED)

set(JETSON_MMAPI_COMMON_SOURCES
  "${JETSON_MMAPI_ROOT}/samples/common/classes/NvJpegEncoder.cpp"
  "${JETSON_MMAPI_ROOT}/samples/common/classes/NvElement.cpp"
  "${JETSON_MMAPI_ROOT}/samples/common/classes/NvElementProfiler.cpp"
  "${JETSON_MMAPI_ROOT}/samples/common/classes/NvBufSurface.cpp"
  "${JETSON_MMAPI_ROOT}/samples/common/classes/NvBuffer.cpp"
  "${JETSON_MMAPI_ROOT}/samples/common/classes/NvLogging.cpp")
foreach(source_file IN LISTS JETSON_MMAPI_COMMON_SOURCES)
  if(NOT EXISTS "${source_file}")
    message(FATAL_ERROR "Jetson Multimedia API sample source is missing: ${source_file}")
  endif()
endforeach()

add_library(${PROJECT_NAME} SHARED
  src/visualization_node.cpp
  src/argus_jpeg.cpp
  ${JETSON_MMAPI_COMMON_SOURCES})
target_include_directories(${PROJECT_NAME} PRIVATE
  src include "${ARGUS_INCLUDE_DIR}" "${JETSON_MMAPI_INCLUDE_DIR}"
  "${JETSON_MMAPI_ROOT}/include/libjpeg-9f" "/usr/local/cuda/include")
ament_target_dependencies(${PROJECT_NAME} argus_interfaces argus_transport rclcpp rclcpp_components sensor_msgs)
target_link_libraries(${PROJECT_NAME}
  "${NVMM_JPEG_LIBRARY}" "${NVBUFSURFACE_LIBRARY}" "${NVBUFSURFTRANSFORM_LIBRARY}"
  "${CUDA_DRIVER_LIBRARY}" "${V4L2_LIBRARY}" Threads::Threads)
rclcpp_components_register_nodes(${PROJECT_NAME} "argus_visualization::VisualizationNode")

install(TARGETS ${PROJECT_NAME} DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME})
ament_package()
