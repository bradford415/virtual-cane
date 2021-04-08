import time
import gpiozero

def get_distance(iteration):
    if ((iteration % 2) == 0):
        if ((iteration % 3) == 0):
            if ((iteration % 4) == 0):
                return 2.5
            return 1.5
        return 0.5
    return 0

def turn_on(distance):
    if (distance > 0 and distance < 1):
        print("1 0 0") #one motors on
    elif (distance < 2 and distance > 1):
        print("1 1 0") #two motors on
    elif (distance < 3 and distance > 2):
        print("1 1 1") #all motors on
    else:
        print("0 0 0") #no motors on

def button_pressed(button_object):
    if (button_object % 7 == 0):
        print("Look at that TV")

i = 0
while (True):

    i = i + 1

    # get realsense distance
    dist = get_distance(i)

    # turn on motors
    turn_on(dist)

    if (button_pressed(i)):
        run_inference()

    time.sleep(0.5)
