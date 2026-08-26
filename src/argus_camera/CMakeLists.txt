cmake_minimum_required(VERSION 3.16)
project(argus_camera LANGUAGES CXX)

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
find_path(ARGUS_INCLUDE_DIR Argus/Argus.h HINTS "${ARGUS_INCLUDE_DIR}" REQUIRED)
find_path(JETSON_MMAPI_INCLUDE_DIR NvBufSurface.h HINTS "${JETSON_MMAPI_INCLUDE_DIR}" REQUIRED)
find_library(NVARGUS_SOCKETCLIENT_LIBRARY nvargus_socketclient
  HINTS /usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}/nvidia /usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}/tegra
  REQUIRED)
find_library(NVBUFSURFACE_LIBRARY nvbufsurface
  HINTS /usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}/nvidia /usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}/tegra
  REQUIRED)
find_library(EGL_LIBRARY EGL REQUIRED)

add_library(${PROJECT_NAME} SHARED
  src/argus_camera_node.cpp
  src/camera_controls.cpp)
target_include_directories(${PROJECT_NAME} PRIVATE src "${JETSON_MMAPI_INCLUDE_DIR}" PUBLIC include "${ARGUS_INCLUDE_DIR}")
ament_target_dependencies(${PROJECT_NAME} argus_interfaces argus_transport rclcpp rclcpp_components)
target_link_libraries(${PROJECT_NAME}
  "${NVARGUS_SOCKETCLIENT_LIBRARY}" "${NVBUFSURFACE_LIBRARY}" "${EGL_LIBRARY}" Threads::Threads)
rclcpp_components_register_nodes(${PROJECT_NAME} "argus_camera::ArgusCameraNode")

install(TARGETS ${PROJECT_NAME} DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
ament_export_include_directories(include "${ARGUS_INCLUDE_DIR}")
ament_export_libraries(${PROJECT_NAME})
ament_package()
