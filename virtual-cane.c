// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2017 Intel Corporation. All Rights Reserved.

/* Include the librealsense C header files */
#include <librealsense2/rs.h>
#include <librealsense2/h/rs_pipeline.h>
#include <librealsense2/h/rs_option.h>
#include <librealsense2/h/rs_frame.h>
#include "example.h"

#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>
#include <signal.h>

#include <gpiod.h>


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                     These parameters are reconfigurable                                        //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define STREAM          RS2_STREAM_DEPTH  // rs2_stream is a types of data provided by RealSense device           //
#define FORMAT          RS2_FORMAT_Z16    // rs2_format identifies how binary data is encoded within a frame      //
#define WIDTH           640               // Defines the number of columns for each frame or zero for auto resolve//
#define HEIGHT          0                 // Defines the number of lines for each frame or zero for auto resolve  //
#define FPS             30                // Defines the rate of frames per second                                //
#define STREAM_INDEX    0                 // Defines the stream index, used for multiple streams of the same type //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CONSUMER
#define CONSUMER	"Consumer"
#endif

void checkGpioError(int ret, struct gpiod_chip **chip, struct gpiod_line **line);

char *chipname = "gpiochip0";
unsigned int pin1 = 14,pin2 = 15, pin3 = 18;
struct gpiod_chip *chip;
struct gpiod_line *line1, *line2, *line3;

void intHandler(int signum)
{
	int ret;
	printf("Caught signal %d, exiting...\n", signum);

	ret = gpiod_line_set_value(line1, 0);
	checkGpioError(ret, &chip, &line1);
	ret = gpiod_line_set_value(line2, 0);
	checkGpioError(ret, &chip, &line2);
	ret = gpiod_line_set_value(line3, 0);
	checkGpioError(ret, &chip, &line3);
	
	exit(0);
	return; 

}

void checkGpioError(int ret, struct gpiod_chip **chip, struct gpiod_line **line)
{
	if (ret < 0) 
	{
		perror("Failed to output value\n");
		gpiod_line_release(*line);
		gpiod_chip_close(*chip);
		exit(0);
	}

	return;
}

void init_gpio(struct gpiod_chip **chip, struct gpiod_line **line1, struct gpiod_line **line2, struct gpiod_line **line3, char *chipname, unsigned int pin_num1, unsigned int pin_num2, unsigned int pin_num3)
{
	int ret;
	printf("Setting up chip %s...\n", chipname);

	*chip = gpiod_chip_open_by_name(chipname);
	if (!(*chip)) 
	{
		perror("Failed to open chip\n");
		exit(0);
	}

	*line1 = gpiod_chip_get_line(*chip, pin_num1);
	if(!(*line1))
	{
		perror("Failed to get line\n");
		gpiod_chip_close(*chip);
		exit(0);
	}
	*line2 = gpiod_chip_get_line(*chip, pin_num2);
	if(!(*line2))
	{
		perror("Failed to get line\n");
		gpiod_chip_close(*chip);
		exit(0);
	}
	*line3 = gpiod_chip_get_line(*chip, pin_num3);
	if(!(*line3))
	{
		perror("Failed to get line\n");
		gpiod_chip_close(*chip);
		exit(0);
	}

	ret = gpiod_line_request_output(*line1, CONSUMER, 0);
	checkGpioError(ret, chip, line1);
	ret = gpiod_line_request_output(*line2, CONSUMER, 0);
	checkGpioError(ret, chip, line2);
	ret = gpiod_line_request_output(*line3, CONSUMER, 0);
	checkGpioError(ret, chip, line3);


	ret = gpiod_line_set_value(*line1, 0);
	checkGpioError(ret, chip, line1);
	ret = gpiod_line_set_value(*line2, 0);
	checkGpioError(ret, chip, line2);
	ret = gpiod_line_set_value(*line3, 0);
	checkGpioError(ret, chip, line3);

	int val = 1, i;
	for (i = 2; i > 0; i--)
	{
		ret = gpiod_line_set_value(*line1, val);
		checkGpioError(ret, chip, line1);
		printf("Output %u on pin #%u\n", val, pin_num1);
		ret = gpiod_line_set_value(*line2, val);
		checkGpioError(ret, chip, line2);
		printf("Output %u on pin #%u\n", val, pin_num2);
		ret = gpiod_line_set_value(*line3, val);
		checkGpioError(ret, chip, line3);
		printf("Output %u on pin #%u\n", val, pin_num3);
		sleep(1);
		val = !val;
	}

	return;
}

void set_motors(struct gpiod_chip **chip, struct gpiod_line **pin1, struct gpiod_line **pin2, struct gpiod_line **pin3, float distance)
{
	int ret;
	if (distance >= 3.0)
	{
		ret = gpiod_line_set_value(*pin1, 0);
		checkGpioError(ret, chip, pin1);
		ret = gpiod_line_set_value(*pin2, 0);
		checkGpioError(ret, chip, pin2);
		ret = gpiod_line_set_value(*pin3, 0);
		checkGpioError(ret, chip, pin3);
	}
	else if (distance >= 2.0 && distance < 3.0)
	{
		ret = gpiod_line_set_value(*pin1, 1);
		checkGpioError(ret, chip, pin1);
		ret = gpiod_line_set_value(*pin2, 0);
		checkGpioError(ret, chip, pin2);
		ret = gpiod_line_set_value(*pin3, 0);
		checkGpioError(ret, chip, pin3);
	}
	else if (distance >= 1.0 && distance < 2.0)
	{
		ret = gpiod_line_set_value(*pin1, 1);
		checkGpioError(ret, chip, pin1);
		ret = gpiod_line_set_value(*pin2, 1);
		checkGpioError(ret, chip, pin2);
		ret = gpiod_line_set_value(*pin3, 0);
		checkGpioError(ret, chip, pin3);
	}
	else
	{
		ret = gpiod_line_set_value(*pin1, 1);
		checkGpioError(ret, chip, pin1);
		ret = gpiod_line_set_value(*pin2, 1);
		checkGpioError(ret, chip, pin2);
		ret = gpiod_line_set_value(*pin3, 1);
		checkGpioError(ret, chip, pin3);
	}

	return;
}

int main()
{
	signal(SIGINT, intHandler);
/*
	char *chipname = "gpiochip0";
	unsigned int pin1 = 14,pin2 = 15, pin3 = 18;
	struct gpiod_chip *chip;
	struct gpiod_line *line1, *line2, *line3;
	*/

	// initialize gpio files
	init_gpio(&chip, &line1, &line2, &line3, chipname, pin1, pin2, pin3);

	rs2_error* e = 0;

	// Create a context object. This object owns the handles to all connected realsense devices.
	// The returned object should be released with rs2_delete_context(...)
	rs2_context* ctx = rs2_create_context(RS2_API_VERSION, &e);
	check_error(e);

	/* Get a list of all the connected devices. */
	// The returned object should be released with rs2_delete_device_list(...)
	rs2_device_list* device_list = rs2_query_devices(ctx, &e);
	check_error(e);

	int dev_count = rs2_get_device_count(device_list, &e);
	check_error(e);
	printf("There are %d connected RealSense devices.\n", dev_count);
	if (0 == dev_count)
		return EXIT_FAILURE;

	// Get the first connected device
	// The returned object should be released with rs2_delete_device(...)
	rs2_device* dev = rs2_create_device(device_list, 0, &e);
	check_error(e);

	print_device_info(dev);

	// Create a pipeline to configure, start and stop camera streaming
	// The returned object should be released with rs2_delete_pipeline(...)
	rs2_pipeline* pipeline =  rs2_create_pipeline(ctx, &e);
	check_error(e);

	// Create a config instance, used to specify hardware configuration
	// The retunred object should be released with rs2_delete_config(...)
	rs2_config* config = rs2_create_config(&e);
	check_error(e);

	// Request a specific configuration
	rs2_config_enable_stream(config, STREAM, STREAM_INDEX, WIDTH, HEIGHT, FORMAT, FPS, &e);
	check_error(e);

	// Start the pipeline streaming
	// The retunred object should be released with rs2_delete_pipeline_profile(...)
	rs2_pipeline_profile* pipeline_profile = rs2_pipeline_start_with_config(pipeline, config, &e);
	if (e)
	{
		printf("The connected device doesn't support depth streaming!\n");
		exit(EXIT_FAILURE);
	}

	while (1)
	{
		// This call waits until a new composite_frame is available
		// composite_frame holds a set of frames. It is used to prevent frame drops
		// The returned object should be released with rs2_release_frame(...)
		rs2_frame* frames = rs2_pipeline_wait_for_frames(pipeline, RS2_DEFAULT_TIMEOUT, &e);
		check_error(e);

		// Returns the number of frames embedded within the composite frame
		int num_of_frames = rs2_embedded_frames_count(frames, &e);
		//printf("WAITING ON %d FRAMES\n",num_of_frames);
		check_error(e);

		int i;
		for (i = 0; i < num_of_frames; ++i)
		{
			// The retunred object should be released with rs2_release_frame(...)
			rs2_frame* frame = rs2_extract_frame(frames, i, &e);
			check_error(e);

			// Check if the given frame can be extended to depth frame interface
			// Accept only depth frames and skip other frames
			if (0 == rs2_is_frame_extendable_to(frame, RS2_EXTENSION_DEPTH_FRAME, &e))
				continue;

			// Get the depth frame's dimensions
			int width = rs2_get_frame_width(frame, &e);
			check_error(e);
			int height = rs2_get_frame_height(frame, &e);
			check_error(e);

			// Query the distance from the camera to the object in the center of the image
			float dist_to_center = rs2_depth_frame_get_distance(frame, width / 2, height / 2, &e);
			check_error(e);

			// Print the distance
			printf("The camera is facing an object %.3f meters away.\n", dist_to_center);

			set_motors(&chip, &line1, &line2, &line3, dist_to_center);

			rs2_release_frame(frame);
		}

		rs2_release_frame(frames);
	}

	// Stop the pipeline streaming
	rs2_pipeline_stop(pipeline, &e);
	check_error(e);

	// Release resources
	rs2_delete_pipeline_profile(pipeline_profile);
	rs2_delete_config(config);
	rs2_delete_pipeline(pipeline);
	rs2_delete_device(dev);
	rs2_delete_device_list(device_list);
	rs2_delete_context(ctx);

	return EXIT_SUCCESS;
}
