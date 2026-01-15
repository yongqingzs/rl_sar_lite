cmake_minimum_required(VERSION 3.8)
project(control_input_msgs)

find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
        "msg/Inputs.msg"
        "msg/ContactInfo.msg"
        DEPENDENCIES std_msgs
)

file(GLOB SCRIPTS
  "scripts/*.py"
)

install(PROGRAMS
  ${SCRIPTS}
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
