# License: Apache 2.0. See LICENSE file in root directory.
# Copyright(c) 2019 Intel Corporation. All Rights Reserved.
#  minimum required cmake version: 3.1.0
cmake_minimum_required(VERSION 3.1.0)

project(RealsenseExamples-Distance)

# Save the command line compile commands in the build output
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)


add_executable(virtual-cane virtual-cane.c example.h)
target_link_libraries(virtual-cane /home/pi/librealsense/build/librealsense2.so -lgpiod)
set_target_properties (virtual-cane PROPERTIES
FOLDER "Examples/C"
)

install(TARGETS virtual-cane RUNTIME DESTINATION ./build)

