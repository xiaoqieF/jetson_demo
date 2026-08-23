cmake_minimum_required(VERSION 3.16)
project(argus_interfaces LANGUAGES C CXX)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ArgusYuvFrame.msg"
  "msg/ArgusInstanceSegmentation.msg"
  "msg/ArgusInferenceResult.msg"
  DEPENDENCIES std_msgs)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
