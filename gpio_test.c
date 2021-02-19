#include <gpiod.h>
#include <stdio.h>
#include <unistd.h>

#ifndef	CONSUMER
#define	CONSUMER	"Consumer"
#endif

int main(int argc, char **argv)
{
	char *chipname = "gpiochip0";
	unsigned int line_num = 14;	// GPIO Pin #14
	unsigned int val;
	struct gpiod_chip *chip;
	struct gpiod_line *line;
	int i, ret;

	chip = gpiod_chip_open_by_name(chipname);
	if (!chip) {
		perror("Open chip failed\n");
		return 0;
	}

	line = gpiod_chip_get_line(chip, line_num);
	if (!line) {
		perror("Get line failed\n");
		gpiod_chip_close(chip);
	}

	ret = gpiod_line_request_output(line, CONSUMER, 0);
	if (ret < 0) {
		perror("Request line as output failed\n");
		gpiod_line_release(line);
	}

	/* Blink 20 times */
	val = 0;
	for (i = 20; i > 0; i--) {
		ret = gpiod_line_set_value(line, val);
		if (ret < 0) {
			perror("Set line output failed\n");
			gpiod_line_release(line);
		}
		printf("Output %u on line #%u\n", val, line_num);
		sleep(1);
		val = !val;
	}
	return 0;
}
