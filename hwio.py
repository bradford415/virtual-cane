import time 
import random
from gpiozero import LED, Button

def get_distance():
	return random.random() * 5.0
	

def turn_on(distance, l1, l2, l3):
    if (distance < 3 and distance > 2):
        print("1 0 0 , Dist = ",distance) #one motors on
        l1.on()
        l2.off()
        l3.off()
    elif (distance < 2 and distance > 1):
        print("1 1 0 , Dist = ",distance) #two motors on
        l1.on()
        l2.on()
        l3.off()
    elif (distance > 0 and distance < 1):
        print("1 1 1 , Dist = ",distance) #all motors on
        l1.on()
        l2.on()
        l3.on()
    else:
        print("0 0 0 , Dist = ",distance) #no motors on
        l1.off()
        l2.off()
        l3.off()

def run_inference(led):

    #PUT INFERENCE CODE HERE

    led.toggle()
    print("Look at that TV")

    #END INFERENCE CODE

def inference_wrapper(led):

    #WRAPPER NEEDED FOR BUTTON PRESSES

    return lambda: run_inference(led)

#BEGIN SETTING UP HW I/O
led1 = LED(14)
led2 = LED(15)
led3 = LED(18)
bled = LED(23)
button = Button(24,pull_up=False)
#END SETTING UP HW I/O

#MAIN WHILE LOOP
while (True):

    # TO BE REPLACED BY: get realsense distance
    dist = get_distance()

    # turn on motors based on distance
    turn_on(dist,led1,led2,led3)

    button.when_pressed = inference_wrapper(bled)

    time.sleep(2) #distance calc runs every 2 seconds 
                  #but button interrupt still works during 
                  #sleep (THIS SHOULD BE REMOVED PROBABLY)

#END MAIN WHILE LOOP
