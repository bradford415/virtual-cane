#include <iostream>
#include <librealsense2/rs.hpp>
#include <memory>
#include <opencv2/opencv.hpp>

#include "motor_interface.hpp"
#include "tensorflow/tensorflow/lite/error_reporter.h"
#include "tensorflow/tensorflow/lite/interpreter.h"
#include "tensorflow/tensorflow/lite/interpreter_builder.h"
#include "tensorflow/tensorflow/lite/kernels/register.h"
#include "tensorflow/tensorflow/lite/model.h"
#include "tensorflow/tensorflow/lite/model_builder.h"
#include "tensorflow/tensorflow/lite/op_resolver.h"

int main() {
    rs2::pipeline p;
    p.start();

    // TFLite Setup;
    /*
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile("detect.tflite");
    if (model == nullptr) {
        std::cout << "Model load failure" << std::endl;
        exit(EXIT_FAILURE);
    }
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    if (!interpreter) {
        std::cout << "Failed to build interpreter" << std::endl;
        exit(EXIT_FAILURE);
    }
    interpreter->SetAllowFp16PrecisionForFp32(true);
    interpreter->SetNumThreads(-1);
    interpreter->AllocateTensors();
    */
    while (true) {
        rs2::frameset frames = p.wait_for_frames();
        rs2::depth_frame depth = frames.get_depth_frame();
        rs2::video_frame color = frames.get_color_frame();
        float dwidth = depth.get_width();
        float dheight = depth.get_height();
        float cwidth = color.get_width();
        float cheight = color.get_height();
        if (cwidth != dwidth || cheight != dheight) {
            std::cout << "Color and Depth dimensions do not match";
            exit(EXIT_FAILURE);
        }
        float dist_to_center = depth.get_distance(dwidth / 2, dheight / 2);
        std::cout << "Distance to center = " << dist_to_center << " meters\n";
        // Resize here if needed
        // expand dims
    }
    return EXIT_SUCCESS;
}