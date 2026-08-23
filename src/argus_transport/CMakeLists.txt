cmake_minimum_required(VERSION 3.16)
project(argus_transport LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(ARGUS_ROOT "/usr/src/jetson_multimedia_api/argus" CACHE PATH "LibArgus root")
set(ARGUS_INCLUDE_DIR "${ARGUS_ROOT}/include" CACHE PATH "LibArgus include directory")

find_package(ament_cmake REQUIRED)
find_package(argus_interfaces REQUIRED)
find_package(rclcpp REQUIRED)
find_path(ARGUS_INCLUDE_DIR Argus/Argus.h HINTS "${ARGUS_INCLUDE_DIR}" REQUIRED)

add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(${PROJECT_NAME} INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${ARGUS_INCLUDE_DIR}>
  $<INSTALL_INTERFACE:include>
  $<INSTALL_INTERFACE:${ARGUS_INCLUDE_DIR}>)
ament_target_dependencies(${PROJECT_NAME} INTERFACE argus_interfaces rclcpp)

install(TARGETS ${PROJECT_NAME} EXPORT export_${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)
ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(argus_interfaces rclcpp)
ament_export_include_directories(include "${ARGUS_INCLUDE_DIR}")
ament_package()
