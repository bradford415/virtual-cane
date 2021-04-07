# License: Apache 2.0. See LICENSE file in root directory.
# Copyright(c) 2019 Intel Corporation. All Rights Reserved.
#  minimum required cmake version: 3.1.0
cmake_minimum_required(VERSION 3.1.0)

project(RealsenseExamples-Distance)

# Save the command line compile commands in the build output
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)


#add_executable(virtual-cane-hello virtual-helloworld.cpp motor_interface.hpp tensorflow/tensorflow/lite/interpreter.h tensorflow/tensorflow/lite/model_builder.h tensorflow/tensorflow/lite/interpreter_builder.h tensorflow/tensorflow/lite/op_resolver.h tensorflow/tensorflow/lite/error_reporter.h)
add_executable(virtual-cane-hello virtual-helloworld.cpp motor_interface.hpp)
target_link_libraries(virtual-cane-hello /home/pi/librealsense/build/librealsense2.so -lopencv_core -lgpiod /usr/local/lib/libtensorflow-lite.a /usr/local/lib/libflatbuffers.a)
target_include_directories(virtual-cane-hello PRIVATE /home/pi/tensorflow)
set_target_properties (virtual-cane-hello PROPERTIES
FOLDER "Examples/C"
)

install(TARGETS virtual-cane-hello RUNTIME DESTINATION ./build)

